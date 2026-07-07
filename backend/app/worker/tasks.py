"""arq task functions. Each opens its own DB session and its own Postgres
checkpointer connection (LangGraph state lives in Postgres, keyed by
thread_id = str(run.id), so a run can be resumed by a later, separate task
invocation after an approval interrupt).

Every event is published to Redis (`run:{run_id}`) as it happens, for the
SSE endpoint to relay live. Non-token events are also persisted to
`run_events` immediately (not buffered), so a client that connects mid-run
can backfill from Postgres and then tail the live channel. Token-level
chunks are streamed live only, never persisted — they're too fine-grained
to be useful in the audit trail.
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from typing import Any

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from agent_runtime import executor
from agent_runtime.spec.models import AgentSpec
from app.core.config import get_settings
from app.core.redis import get_redis, run_channel
from app.db.models import Agent, Run, RunEvent
from app.db.session import async_session_factory


def _now() -> datetime:
    return datetime.now(UTC)


_STATUS_MAP = {
    "succeeded": "succeeded",
    "failed": "failed",
    "waiting_approval": "waiting_approval",
}


def _make_event_sink(session, run: Run):
    seq_counter = {"n": 0}
    redis = get_redis()
    channel = run_channel(str(run.id))

    async def event_sink(event: dict[str, Any]) -> None:
        await redis.publish(channel, json.dumps(event))
        if event["type"] == "token":
            return
        seq_counter["n"] += 1
        session.add(RunEvent(run_id=run.id, seq=seq_counter["n"], type=event["type"], payload=event))
        await session.commit()

    return event_sink


async def _finalize(session, run: Run, outcome) -> None:
    run.status = _STATUS_MAP[outcome.status]
    run.cost_usd = outcome.budget_used.get("cost_usd")
    if outcome.status == "failed":
        run.error = outcome.error
        run.completed_at = _now()
    elif outcome.status == "succeeded":
        run.output = {"text": outcome.output, "terminal_reason": outcome.terminal_reason}
        run.completed_at = _now()
    else:  # waiting_approval — no completed_at, run stays open pending a resume
        run.output = {"interrupt": outcome.interrupt}

    session.add(run)
    await session.commit()

    redis = get_redis()
    await redis.publish(run_channel(str(run.id)), json.dumps({"type": "run_complete", "status": run.status}))


async def run_agent_task(ctx: dict, run_id: str) -> None:
    settings = get_settings()
    async with async_session_factory() as session:
        run = await session.get(Run, uuid.UUID(run_id))
        if run is None:
            return
        agent = await session.get(Agent, run.agent_id)
        if agent is None:
            run.status = "failed"
            run.error = "agent not found"
            session.add(run)
            await session.commit()
            return
        spec = AgentSpec.model_validate(agent.spec)

        run.status = "running"
        run.started_at = _now()
        session.add(run)
        await session.commit()

        event_sink = _make_event_sink(session, run)

        try:
            async with AsyncPostgresSaver.from_conn_string(settings.psycopg_database_url) as checkpointer:
                outcome = await executor.start_run(
                    spec,
                    run.input.get("message", ""),
                    thread_id=str(run.id),
                    checkpointer=checkpointer,
                    event_sink=event_sink,
                )
        except Exception as exc:  # noqa: BLE001 — a run must never get stuck at "running" forever
            outcome = executor.RunOutcome(status="failed", error=str(exc))

        await _finalize(session, run, outcome)


async def resume_run_task(ctx: dict, run_id: str, decision: dict[str, Any]) -> None:
    settings = get_settings()
    async with async_session_factory() as session:
        run = await session.get(Run, uuid.UUID(run_id))
        if run is None or run.status != "waiting_approval":
            return
        agent = await session.get(Agent, run.agent_id)
        spec = AgentSpec.model_validate(agent.spec)

        run.status = "running"
        session.add(run)
        await session.commit()

        event_sink = _make_event_sink(session, run)

        try:
            async with AsyncPostgresSaver.from_conn_string(settings.psycopg_database_url) as checkpointer:
                outcome = await executor.resume_run(
                    spec,
                    decision,
                    thread_id=str(run.id),
                    checkpointer=checkpointer,
                    event_sink=event_sink,
                )
        except Exception as exc:  # noqa: BLE001
            outcome = executor.RunOutcome(status="failed", error=str(exc))

        await _finalize(session, run, outcome)
