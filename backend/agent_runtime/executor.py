"""The single entrypoint for running an AgentSpec: used by the worker task,
the eval harness, and any future CLI. No FastAPI import anywhere in this
package — it's usable standalone.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from agent_runtime.compiler.graph_builder import build
from agent_runtime.compiler.state import initial_state
from agent_runtime.guardrails.input_filters import InputRejected, check_input
from agent_runtime.spec.models import AgentSpec
from tools.registry import ToolRegistry
from tools.registry import registry as default_registry

RunStatus = Literal["succeeded", "failed", "waiting_approval"]

EventSink = Callable[[dict[str, Any]], Awaitable[None]]


@dataclass
class RunOutcome:
    status: RunStatus
    output: str | None = None
    terminal_reason: str | None = None
    budget_used: dict[str, Any] = field(default_factory=dict)
    interrupt: dict[str, Any] | None = None
    error: str | None = None


async def _emit(sink: EventSink | None, event: dict[str, Any]) -> None:
    if sink is not None:
        await sink(event)


def _default_checkpointer() -> BaseCheckpointSaver:
    return MemorySaver()


async def _drain_and_summarize(
    graph,
    stream_input: Any,
    config: dict[str, Any],
    event_sink: EventSink | None,
) -> RunOutcome:
    try:
        async for event in graph.astream_events(stream_input, config, version="v2"):
            kind = event.get("event")
            if kind == "on_chat_model_stream":
                chunk = event["data"].get("chunk")
                content = getattr(chunk, "content", "") if chunk else ""
                if content:
                    await _emit(event_sink, {"type": "token", "content": content})
            elif kind == "on_chain_start" and event.get("name") in {"agent", "tools", "budget_exceeded"}:
                await _emit(event_sink, {"type": "node_start", "node": event["name"]})
            elif kind == "on_chain_end" and event.get("name") in {"agent", "tools", "budget_exceeded"}:
                await _emit(event_sink, {"type": "node_end", "node": event["name"]})
            elif kind == "on_tool_start":
                await _emit(
                    event_sink,
                    {
                        "type": "tool_call",
                        "tool_name": event.get("name"),
                        "args": event.get("data", {}).get("input"),
                    },
                )
            elif kind == "on_tool_end":
                output = event.get("data", {}).get("output")
                await _emit(
                    event_sink,
                    {
                        "type": "tool_result",
                        "tool_name": event.get("name"),
                        "output": str(output) if output is not None else None,
                    },
                )
    except Exception as exc:  # noqa: BLE001 — surfaced to the caller as a failed run
        await _emit(event_sink, {"type": "error", "message": str(exc)})
        return RunOutcome(status="failed", error=str(exc))

    state = await graph.aget_state(config)

    if state.next:
        for task in state.tasks:
            if task.interrupts:
                payload = task.interrupts[0].value
                await _emit(event_sink, {"type": "interrupt", "payload": payload})
                return RunOutcome(
                    status="waiting_approval",
                    interrupt=payload,
                    budget_used=state.values.get("budget_used", {}),
                )

    messages = state.values.get("messages", [])
    last_ai = next((m for m in reversed(messages) if isinstance(m, AIMessage)), None)
    terminal_reason = state.values.get("terminal_reason")
    return RunOutcome(
        status="succeeded",
        output=last_ai.content if last_ai and isinstance(last_ai.content, str) else None,
        terminal_reason=terminal_reason,
        budget_used=state.values.get("budget_used", {}),
    )


async def start_run(
    spec: AgentSpec,
    input_text: str,
    *,
    thread_id: str,
    checkpointer: BaseCheckpointSaver | None = None,
    tool_registry: ToolRegistry = default_registry,
    event_sink: EventSink | None = None,
) -> RunOutcome:
    try:
        check_input(input_text, spec.guardrails.input)
    except InputRejected as exc:
        return RunOutcome(status="failed", error=str(exc))

    checkpointer = checkpointer or _default_checkpointer()
    try:
        graph = build(spec, tool_registry=tool_registry, checkpointer=checkpointer)
    except Exception as exc:  # noqa: BLE001 — e.g. missing model credentials, bad tool config
        await _emit(event_sink, {"type": "error", "message": str(exc)})
        return RunOutcome(status="failed", error=str(exc))
    config = {"configurable": {"thread_id": thread_id}}
    stream_input = initial_state([HumanMessage(content=input_text)])
    return await _drain_and_summarize(graph, stream_input, config, event_sink)


async def resume_run(
    spec: AgentSpec,
    decision: dict[str, Any],
    *,
    thread_id: str,
    checkpointer: BaseCheckpointSaver,
    tool_registry: ToolRegistry = default_registry,
    event_sink: EventSink | None = None,
) -> RunOutcome:
    try:
        graph = build(spec, tool_registry=tool_registry, checkpointer=checkpointer)
    except Exception as exc:  # noqa: BLE001
        await _emit(event_sink, {"type": "error", "message": str(exc)})
        return RunOutcome(status="failed", error=str(exc))
    config = {"configurable": {"thread_id": thread_id}}
    return await _drain_and_summarize(graph, Command(resume=decision), config, event_sink)
