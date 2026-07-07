"""linear_pipeline graph type: a fixed sequence of stages (e.g. search ->
synthesize -> cite -> format), each of which is itself a small bounded
tool-calling loop — so a stage can still call tools as needed, it just can't
loop back to an earlier stage. Used for agents with a genuinely staged
workflow rather than an open-ended ReAct loop.
"""

from __future__ import annotations

import time
from collections.abc import Callable

from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, StateGraph
from langgraph.types import interrupt

from agent_runtime.compiler.state import AgentState
from agent_runtime.compiler.tool_resolution import resolve_tools
from agent_runtime.guardrails import budget as budget_guardrails
from agent_runtime.guardrails.output_filters import apply_output_guardrails
from agent_runtime.models.provider_router import get_chat_model
from agent_runtime.spec.models import AgentSpec, GraphNode
from tools.registry import ToolRegistry


def build_linear_pipeline(
    spec: AgentSpec,
    tool_registry: ToolRegistry,
    checkpointer: BaseCheckpointSaver | None,
):
    if not spec.graph.nodes:
        raise ValueError(
            f"agent '{spec.id}': graph.type == 'linear_pipeline' requires at least one node in graph.nodes"
        )

    model = get_chat_model(spec.model)
    tools, name_to_id = resolve_tools(spec, tool_registry)
    tools_by_name = {t.name: t for t in tools}
    model_runnable = model.bind_tools(tools) if tools else model
    approval_ids = set(spec.guardrails.approvals.require_human_approval_for)
    started_at = time.monotonic()

    def make_step(node: GraphNode) -> Callable[[AgentState], dict]:
        step_instructions = node.config.get("instructions", "")
        step_max_iterations = int(node.config.get("max_iterations", 5))
        system_message = SystemMessage(
            content=f"{spec.system_prompt}\n\nCurrent stage: '{node.id}'. {step_instructions}".strip()
        )

        def step_node(state: AgentState) -> dict:
            budget = dict(state["budget_used"])
            new_messages: list = []

            for _ in range(step_max_iterations):
                response = model_runnable.invoke([system_message, *state["messages"], *new_messages])
                cost = (
                    budget_guardrails.estimate_cost_usd(spec.model.model_name, response)
                    if isinstance(response, AIMessage)
                    else 0.0
                )
                budget["cost_usd"] += cost
                budget["seconds"] = budget_guardrails.elapsed_seconds(started_at)
                if isinstance(response, AIMessage) and isinstance(response.content, str):
                    response.content = apply_output_guardrails(response.content, spec.guardrails.output)
                new_messages.append(response)

                tool_calls = getattr(response, "tool_calls", None) or []
                if not tool_calls:
                    break

                for call in tool_calls:
                    ref_id = name_to_id.get(call["name"], call["name"])
                    if ref_id in approval_ids:
                        decision = interrupt(
                            {
                                "type": "tool_approval",
                                "tool_id": ref_id,
                                "tool_name": call["name"],
                                "args": call["args"],
                                "stage": node.id,
                            }
                        )
                        if not (isinstance(decision, dict) and decision.get("approved")):
                            new_messages.append(
                                ToolMessage(
                                    content=f"Tool call to '{ref_id}' was not approved.",
                                    tool_call_id=call["id"],
                                )
                            )
                            continue
                    tool = tools_by_name.get(call["name"])
                    if tool is None:
                        new_messages.append(
                            ToolMessage(content=f"Unknown tool: {call['name']}", tool_call_id=call["id"])
                        )
                        continue
                    try:
                        result = tool.invoke(call["args"])
                    except Exception as exc:  # noqa: BLE001 — tool errors surface to the model
                        result = f"Tool error: {exc}"
                    new_messages.append(ToolMessage(content=str(result), tool_call_id=call["id"]))
                    budget["tool_calls"] += 1

                budget["seconds"] = budget_guardrails.elapsed_seconds(started_at)
                reason = budget_guardrails.budget_exceeded(budget, spec.guardrails.budget)
                if reason:
                    new_messages.append(AIMessage(content=f"Run stopped by a guardrail: {reason}"))
                    return {"messages": new_messages, "budget_used": budget, "terminal_reason": reason}

            return {"messages": new_messages, "budget_used": budget}

        return step_node

    graph = StateGraph(AgentState)
    node_ids = [n.id for n in spec.graph.nodes]
    for node in spec.graph.nodes:
        graph.add_node(node.id, make_step(node))

    graph.set_entry_point(node_ids[0])
    for current_id, next_id in zip(node_ids, node_ids[1:], strict=False):
        graph.add_edge(current_id, next_id)
    graph.add_edge(node_ids[-1], END)

    return graph.compile(checkpointer=checkpointer)
