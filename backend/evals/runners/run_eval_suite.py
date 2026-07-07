"""Structural eval smoke suite.

For each fixture in evals/fixtures/, replays a scripted model through the
REAL compiler/executor/tool-registry and asserts on the resulting run status
and tool-call trace shape (e.g. "must call X", "must pause for approval
before Y") — never on exact LLM output text, since that's non-deterministic
even with a real model. Deterministic and free to run (no live LLM calls),
so it's meant to run on every PR; see docs for the separate LLM-judge
quality tier.

Usage (from backend/):
    python -m evals.runners.run_eval_suite
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from unittest.mock import patch

import yaml
from langchain_core.messages import AIMessage

from agent_runtime import executor
from agent_runtime.compiler import custom_dag, graph_builder, pipelines
from agent_runtime.spec.loader import load_builtin_specs
from agent_runtime.testing import ScriptedFakeChatModel
from tools.registry import registry

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"


def _load_fixtures() -> list[dict]:
    fixtures = []
    for path in sorted(FIXTURES_DIR.glob("*.yaml")):
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        data["_name"] = path.stem
        fixtures.append(data)
    return fixtures


def _build_ai_message(entry: dict) -> AIMessage:
    tool_calls = [
        {"name": tc["name"], "args": tc.get("args", {}), "id": tc.get("id", f"call_{i}")}
        for i, tc in enumerate(entry.get("tool_calls", []))
    ]
    return AIMessage(content=entry.get("content", ""), tool_calls=tool_calls)


def _tool_name_to_registry_id() -> dict[str, str]:
    mapping = {}
    for definition in registry.all():
        try:
            instance = definition.factory({})
        except Exception:  # noqa: BLE001 — some factories need config we don't have here
            continue
        mapping[instance.name] = definition.id
    return mapping


async def run_fixture(fixture: dict, specs_by_id: dict, name_to_id: dict[str, str]) -> tuple[bool, str]:
    agent_id = fixture["agent_id"]
    spec = specs_by_id.get(agent_id)
    if spec is None:
        return False, f"unknown agent_id '{agent_id}'"

    responses = [_build_ai_message(e) for e in fixture["scripted_responses"]]
    fake = ScriptedFakeChatModel(responses=responses)
    tool_calls_seen: list[str] = []

    async def event_sink(event: dict) -> None:
        if event["type"] == "tool_call":
            raw_name = event.get("tool_name")
            tool_calls_seen.append(name_to_id.get(raw_name, raw_name))

    with (
        patch.object(graph_builder, "get_chat_model", lambda c: fake),
        patch.object(pipelines, "get_chat_model", lambda c: fake),
        patch.object(custom_dag, "get_chat_model", lambda c: fake),
    ):
        outcome = await executor.start_run(
            spec,
            fixture["input"],
            thread_id=f"eval-{agent_id}-{fixture['_name']}",
            tool_registry=registry,
            event_sink=event_sink,
        )

    expect = fixture.get("expect", {})
    errors = []

    if "status" in expect and outcome.status != expect["status"]:
        errors.append(f"expected status={expect['status']!r}, got {outcome.status!r} (error={outcome.error})")

    if "interrupt_tool_id" in expect:
        actual = (outcome.interrupt or {}).get("tool_id")
        if actual != expect["interrupt_tool_id"]:
            errors.append(f"expected interrupt tool_id={expect['interrupt_tool_id']!r}, got {actual!r}")

    for tool_id in expect.get("tool_calls_include", []):
        if tool_id not in tool_calls_seen:
            errors.append(f"expected tool call {tool_id!r} not observed (saw: {tool_calls_seen})")

    for tool_id in expect.get("must_not_call", []):
        if tool_id in tool_calls_seen:
            errors.append(f"forbidden tool call {tool_id!r} was observed")

    return (not errors), ("; ".join(errors) if errors else "ok")


async def main() -> int:
    registry.discover()
    specs_by_id = {s.id: s for s in load_builtin_specs()}
    name_to_id = _tool_name_to_registry_id()
    fixtures = _load_fixtures()

    if not fixtures:
        print("no eval fixtures found")
        return 1

    failures = 0
    for fixture in fixtures:
        ok, detail = await run_fixture(fixture, specs_by_id, name_to_id)
        print(f"{'PASS' if ok else 'FAIL'}  {fixture['_name']:30s} {detail}")
        if not ok:
            failures += 1

    print(f"\n{len(fixtures) - failures}/{len(fixtures)} fixtures passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
