import json
import uuid
from collections.abc import AsyncGenerator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from sqlmodel import select

from app.core.redis import get_redis, run_channel
from app.db.models import Agent, Run, RunEvent
from app.db.session import async_session_factory
from app.schemas.runs import ResumeRunRequest, RunCreateRequest, RunDetail, RunSummary
from app.worker.client import enqueue

router = APIRouter(tags=["runs"])

_TERMINAL_STATUSES = {"succeeded", "failed", "waiting_approval"}


@router.post("/agents/{agent_id}/runs", response_model=RunSummary, status_code=202)
async def create_run(agent_id: uuid.UUID, body: RunCreateRequest) -> Run:
    async with async_session_factory() as session:
        agent = await session.get(Agent, agent_id)
        if agent is None:
            raise HTTPException(status_code=404, detail="agent not found")
        run = Run(agent_id=agent_id, thread_id=str(uuid.uuid4()), input={"message": body.message})
        session.add(run)
        await session.commit()
        await session.refresh(run)

    await enqueue("run_agent_task", str(run.id))
    return run


@router.get("/agents/{agent_id}/runs", response_model=list[RunSummary])
async def list_agent_runs(agent_id: uuid.UUID) -> list[Run]:
    async with async_session_factory() as session:
        result = await session.execute(
            select(Run).where(Run.agent_id == agent_id).order_by(Run.created_at.desc())
        )
        return list(result.scalars().all())


@router.get("/runs/{run_id}", response_model=RunDetail)
async def get_run(run_id: uuid.UUID) -> Run:
    async with async_session_factory() as session:
        run = await session.get(Run, run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="run not found")
        return run


@router.get("/runs/{run_id}/events", response_model=list[dict])
async def list_run_events(run_id: uuid.UUID) -> list[dict]:
    async with async_session_factory() as session:
        result = await session.execute(
            select(RunEvent).where(RunEvent.run_id == run_id).order_by(RunEvent.seq)
        )
        return [{"seq": e.seq, "type": e.type, **e.payload} for e in result.scalars().all()]


def _sse(event_type: str, data: dict) -> bytes:
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n".encode()


@router.get("/runs/{run_id}/stream")
async def stream_run(run_id: uuid.UUID) -> StreamingResponse:
    async def event_generator() -> AsyncGenerator[bytes, None]:
        async with async_session_factory() as session:
            run = await session.get(Run, run_id)
            if run is None:
                yield _sse("error", {"message": "run not found"})
                return

            result = await session.execute(
                select(RunEvent).where(RunEvent.run_id == run_id).order_by(RunEvent.seq)
            )
            for ev in result.scalars().all():
                yield _sse(ev.type, ev.payload)

            if run.status in _TERMINAL_STATUSES:
                yield _sse("run_complete", {"status": run.status})
                return

        redis = get_redis()
        pubsub = redis.pubsub()
        channel = run_channel(str(run_id))
        await pubsub.subscribe(channel)
        print(f"[SSE DEBUG] subscribed to {channel}", flush=True)
        try:
            # A run that completed in the gap between the DB backfill above and this
            # subscribe call would otherwise be waited on forever — re-check once.
            async with async_session_factory() as session:
                run = await session.get(Run, run_id)
                if run and run.status in _TERMINAL_STATUSES:
                    print(f"[SSE DEBUG] already terminal after subscribe: {run.status}", flush=True)
                    yield _sse("run_complete", {"status": run.status})
                    return

            i = 0
            while True:
                i += 1
                message = await pubsub.get_message(ignore_subscribe_messages=True, timeout=30.0)
                print(f"[SSE DEBUG] iter={i} message={message!r}", flush=True)
                if message is None:
                    yield b": keepalive\n\n"
                    continue
                payload = json.loads(message["data"])
                yield _sse(payload.get("type", "message"), payload)
                if payload.get("type") == "run_complete":
                    return
        finally:
            print("[SSE DEBUG] cleaning up subscription", flush=True)
            await pubsub.unsubscribe(channel)
            await pubsub.aclose()

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.post("/runs/{run_id}/resume", response_model=RunSummary)
async def resume_run(run_id: uuid.UUID, body: ResumeRunRequest) -> Run:
    async with async_session_factory() as session:
        run = await session.get(Run, run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="run not found")
        if run.status != "waiting_approval":
            raise HTTPException(status_code=409, detail="run is not waiting for approval")

    await enqueue("resume_run_task", str(run_id), {"approved": body.approved})
    return run
