"""AgentSpec -> LangGraph compiler.

`build()` is the single entrypoint used for both built-in and generated
agents. `graph.type` picks the compilation strategy:

- react: the common case — a system prompt + tool-calling loop, built by
  hand (not LangGraph's prebuilt) so budget/approval guardrails can be woven
  into the loop itself.
- linear_pipeline / custom_dag: see _build_linear_pipeline / _build_custom_dag.
"""

from __future__ import annotations

import time

from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, StateGraph
from langgraph.types import interrupt

from agent_runtime.compiler.state import AgentState
from agent_runtime.compiler.tool_resolution import resolve_tools
from agent_runtime.guardrails import budget as budget_guardrails
from agent_runtime.guardrails.output_filters import apply_output_guardrails
from agent_runtime.models.provider_router import get_chat_model
from agent_runtime.spec.models import AgentSpec
from tools.registry import ToolRegistry
from tools.registry import registry as default_registry


class UnsupportedGraphType(NotImplementedError):
    pass


def _build_react(
    spec: AgentSpec,
    tool_registry: ToolRegistry,
    checkpointer: BaseCheckpointSaver | None,
):
    model = get_chat_model(spec.model)
    tools, name_to_id = resolve_tools(spec, tool_registry)
    tools_by_name = {t.name: t for t in tools}
    model_runnable = model.bind_tools(tools) if tools else model
    system_message = SystemMessage(content=spec.system_prompt)
    approval_ids = set(spec.guardrails.approvals.require_human_approval_for)
    started_at = time.monotonic()

    def agent_node(state: AgentState) -> dict:
        response = model_runnable.invoke([system_message, *state["messages"]])
        cost = (
            budget_guardrails.estimate_cost_usd(spec.model.model_name, response)
            if isinstance(response, AIMessage)
            else 0.0
        )
        if isinstance(response, AIMessage) and isinstance(response.content, str):
            response.content = apply_output_guardrails(response.content, spec.guardrails.output)

        budget = dict(state["budget_used"])
        budget["cost_usd"] += cost
        budget["seconds"] = budget_guardrails.elapsed_seconds(started_at)

        scratchpad = dict(state["scratchpad"])
        scratchpad["iterations"] = scratchpad.get("iterations", 0) + 1

        return {"messages": [response], "budget_used": budget, "scratchpad": scratchpad}

    def tools_node(state: AgentState) -> dict:
        last = state["messages"][-1]
        tool_calls = getattr(last, "tool_calls", None) or []
        tool_messages: list[ToolMessage] = []

        for call in tool_calls:
            ref_id = name_to_id.get(call["name"], call["name"])
            if ref_id in approval_ids:
                decision = interrupt(
                    {
                        "type": "tool_approval",
                        "tool_id": ref_id,
                        "tool_name": call["name"],
                        "args": call["args"],
                    }
                )
                if not (isinstance(decision, dict) and decision.get("approved")):
                    tool_messages.append(
                        ToolMessage(
                            content=f"Tool call to '{ref_id}' was not approved by a human reviewer.",
                            tool_call_id=call["id"],
                        )
                    )
                    continue

            tool = tools_by_name.get(call["name"])
            if tool is None:
                tool_messages.append(
                    ToolMessage(content=f"Unknown tool: {call['name']}", tool_call_id=call["id"])
                )
                continue
            try:
                result = tool.invoke(call["args"])
            except Exception as exc:  # tool errors surface back to the model, not raised
                result = f"Tool error: {exc}"
            tool_messages.append(ToolMessage(content=str(result), tool_call_id=call["id"]))

        budget = dict(state["budget_used"])
        budget["tool_calls"] += len(tool_calls)
        budget["seconds"] = budget_guardrails.elapsed_seconds(started_at)
        return {"messages": tool_messages, "budget_used": budget}

    def budget_exceeded_node(state: AgentState) -> dict:
        reason = (
            budget_guardrails.budget_exceeded(state["budget_used"], spec.guardrails.budget)
            or f"max_iterations exceeded ({spec.graph.max_iterations})"
        )
        return {
            "terminal_reason": reason,
            "messages": [AIMessage(content=f"Run stopped by a guardrail: {reason}")],
        }

    def route_after_agent(state: AgentState) -> str:
        if budget_guardrails.budget_exceeded(state["budget_used"], spec.guardrails.budget):
            return "budget_exceeded"
        if state["scratchpad"].get("iterations", 0) >= spec.graph.max_iterations:
            return "budget_exceeded"
        last = state["messages"][-1]
        if getattr(last, "tool_calls", None):
            return "tools"
        return "end"

    def route_after_tools(state: AgentState) -> str:
        if budget_guardrails.budget_exceeded(state["budget_used"], spec.guardrails.budget):
            return "budget_exceeded"
        return "agent"

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tools_node)
    graph.add_node("budget_exceeded", budget_exceeded_node)
    graph.set_entry_point("agent")
    graph.add_conditional_edges(
        "agent", route_after_agent, {"tools": "tools", "end": END, "budget_exceeded": "budget_exceeded"}
    )
    graph.add_conditional_edges(
        "tools", route_after_tools, {"agent": "agent", "budget_exceeded": "budget_exceeded"}
    )
    graph.add_edge("budget_exceeded", END)

    return graph.compile(checkpointer=checkpointer)


def _build_linear_pipeline(
    spec: AgentSpec,
    tool_registry: ToolRegistry,
    checkpointer: BaseCheckpointSaver | None,
):
    from agent_runtime.compiler.pipelines import build_linear_pipeline

    return build_linear_pipeline(spec, tool_registry, checkpointer)


def _build_custom_dag(
    spec: AgentSpec,
    tool_registry: ToolRegistry,
    checkpointer: BaseCheckpointSaver | None,
):
    from agent_runtime.compiler.custom_dag import build_custom_dag

    return build_custom_dag(spec, tool_registry, checkpointer)


def build(
    spec: AgentSpec,
    *,
    tool_registry: ToolRegistry = default_registry,
    checkpointer: BaseCheckpointSaver | None = None,
):
    if spec.graph.type == "react":
        return _build_react(spec, tool_registry, checkpointer)
    if spec.graph.type == "linear_pipeline":
        return _build_linear_pipeline(spec, tool_registry, checkpointer)
    if spec.graph.type == "custom_dag":
        return _build_custom_dag(spec, tool_registry, checkpointer)
    raise UnsupportedGraphType(spec.graph.type)
