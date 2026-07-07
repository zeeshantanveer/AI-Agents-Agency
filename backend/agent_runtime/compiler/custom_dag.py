"""custom_dag graph type: explicit nodes + edges for agents that need real
branching (not just a fixed sequence). Node kinds:

- llm_call: a bounded tool-calling step, same behavior as a linear_pipeline
  stage (see pipelines.py) — can call tools, can't loop back on its own.
- tool_call: directly invokes one specific tool with static and/or
  scratchpad-sourced args — no LLM involved, for deterministic steps.
- router: an LLM call that picks the next node from a small labeled set of
  options (`node.config["options"]: {node_id: description}`), used for
  conditional edges (`edge.condition == node.id`).
- human_review: pauses the graph (LangGraph interrupt) for a human to
  approve/reject the current state before continuing — the "explicit human
  checkpoint" escape hatch, distinct from per-tool approval guardrails.

This is the escape hatch for hand-built agents with genuinely custom control
flow; the v1 prompt-to-agent generator never emits this graph type.
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


def build_custom_dag(
    spec: AgentSpec,
    tool_registry: ToolRegistry,
    checkpointer: BaseCheckpointSaver | None,
):
    if not spec.graph.nodes:
        raise ValueError(f"agent '{spec.id}': graph.type == 'custom_dag' requires graph.nodes")

    model = get_chat_model(spec.model)
    tools, name_to_id = resolve_tools(spec, tool_registry)
    tools_by_name = {t.name: t for t in tools}
    model_runnable = model.bind_tools(tools) if tools else model
    approval_ids = set(spec.guardrails.approvals.require_human_approval_for)
    started_at = time.monotonic()

    nodes_by_id = {n.id: n for n in spec.graph.nodes}
    outgoing: dict[str, list] = {}
    for edge in spec.graph.edges:
        outgoing.setdefault(edge.source, []).append(edge)

    def _run_bounded_tool_loop(node: GraphNode, state: AgentState) -> dict:
        step_instructions = node.config.get("instructions", "")
        step_max_iterations = int(node.config.get("max_iterations", 5))
        system_message = SystemMessage(
            content=f"{spec.system_prompt}\n\nCurrent stage: '{node.id}'. {step_instructions}".strip()
        )
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
                except Exception as exc:  # noqa: BLE001
                    result = f"Tool error: {exc}"
                new_messages.append(ToolMessage(content=str(result), tool_call_id=call["id"]))
                budget["tool_calls"] += 1

            budget["seconds"] = budget_guardrails.elapsed_seconds(started_at)
            reason = budget_guardrails.budget_exceeded(budget, spec.guardrails.budget)
            if reason:
                new_messages.append(AIMessage(content=f"Run stopped by a guardrail: {reason}"))
                return {"messages": new_messages, "budget_used": budget, "terminal_reason": reason}

        return {"messages": new_messages, "budget_used": budget}

    def _run_tool_call_node(node: GraphNode, state: AgentState) -> dict:
        tool_id = node.config["tool_id"]
        args = dict(node.config.get("args", {}))
        for arg_name, scratchpad_key in node.config.get("args_from_scratchpad", {}).items():
            if scratchpad_key in state["scratchpad"]:
                args[arg_name] = state["scratchpad"][scratchpad_key]

        tool = tools_by_name.get(tool_id) or next(
            (t for t, rid in ((t, name_to_id.get(t.name)) for t in tools) if rid == tool_id), None
        )
        if tool is None:
            return {"messages": [AIMessage(content=f"Node '{node.id}': unknown tool '{tool_id}'")]}

        budget = dict(state["budget_used"])
        if tool_id in approval_ids or node.config.get("requires_approval"):
            decision = interrupt(
                {
                    "type": "tool_approval",
                    "tool_id": tool_id,
                    "tool_name": tool.name,
                    "args": args,
                    "stage": node.id,
                }
            )
            if not (isinstance(decision, dict) and decision.get("approved")):
                return {
                    "messages": [AIMessage(content=f"Node '{node.id}': tool call was not approved.")],
                    "budget_used": budget,
                }
        try:
            result = tool.invoke(args)
        except Exception as exc:  # noqa: BLE001
            result = f"Tool error: {exc}"
        budget["tool_calls"] += 1
        return {"messages": [AIMessage(content=str(result))], "budget_used": budget}

    def _run_router_node(node: GraphNode, state: AgentState) -> dict:
        options: dict[str, str] = node.config.get("options", {})
        option_list = "\n".join(f"- {key}: {desc}" for key, desc in options.items())
        prompt = (
            f"{spec.system_prompt}\n\nBased on the conversation so far, choose exactly one option "
            f"by replying with only its key, nothing else.\nOptions:\n{option_list}"
        )
        response = model.invoke([SystemMessage(content=prompt), *state["messages"]])
        choice = str(response.content).strip().strip('"').strip("'")
        if choice not in options:
            choice = next(iter(options), "end")
        scratchpad = dict(state["scratchpad"])
        scratchpad[f"route_from_{node.id}"] = choice
        return {"scratchpad": scratchpad}

    def _run_human_review_node(node: GraphNode, state: AgentState) -> dict:
        last_content = state["messages"][-1].content if state["messages"] else ""
        decision = interrupt({"type": "human_review", "node_id": node.id, "content": last_content})
        scratchpad = dict(state["scratchpad"])
        approved = isinstance(decision, dict) and decision.get("approved", False)
        scratchpad[f"route_from_{node.id}"] = "approved" if approved else "rejected"
        if isinstance(decision, dict) and decision.get("notes"):
            scratchpad[f"{node.id}_notes"] = decision["notes"]
        return {"scratchpad": scratchpad}

    def make_node_fn(node: GraphNode) -> Callable[[AgentState], dict]:
        if node.kind == "tool_call":
            return lambda state: _run_tool_call_node(node, state)
        if node.kind == "router":
            return lambda state: _run_router_node(node, state)
        if node.kind == "human_review":
            return lambda state: _run_human_review_node(node, state)
        return lambda state: _run_bounded_tool_loop(node, state)

    graph = StateGraph(AgentState)
    for node in spec.graph.nodes:
        graph.add_node(node.id, make_node_fn(node))

    entry = spec.graph.nodes[0].id
    graph.set_entry_point(entry)

    for node_id, edges in outgoing.items():
        branching_kinds = {"router", "human_review"}
        if nodes_by_id[node_id].kind in branching_kinds and len(edges) > 1:
            mapping = {
                edge.condition or edge.target: (END if edge.target == "END" else edge.target)
                for edge in edges
            }

            def route(state: AgentState, _node_id: str = node_id, _mapping: dict = mapping) -> str:
                choice = state["scratchpad"].get(f"route_from_{_node_id}")
                return choice if choice in _mapping else next(iter(_mapping))

            graph.add_conditional_edges(node_id, route, mapping)
        else:
            for edge in edges:
                target = END if edge.target == "END" else edge.target
                graph.add_edge(node_id, target)

    terminal_nodes = [n.id for n in spec.graph.nodes if n.id not in outgoing]
    for node_id in terminal_nodes:
        graph.add_edge(node_id, END)

    return graph.compile(checkpointer=checkpointer)
