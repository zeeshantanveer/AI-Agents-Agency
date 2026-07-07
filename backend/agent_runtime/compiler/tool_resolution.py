"""Shared by every graph type: resolves an AgentSpec's tool refs into live
LangChain tools plus a name->registry-id map (needed because the model calls
tools by their LangChain `.name`, but guardrails/approvals are keyed by the
AgentSpec-level registry id).
"""

from __future__ import annotations

from langchain_core.tools import BaseTool

from agent_runtime.spec.models import AgentSpec
from tools.registry import ToolRegistry


def resolve_tools(spec: AgentSpec, tool_registry: ToolRegistry) -> tuple[list[BaseTool], dict[str, str]]:
    tools: list[BaseTool] = []
    name_to_id: dict[str, str] = {}
    for ref in spec.tools:
        built = tool_registry.build(ref.id, ref.config)
        tools.append(built)
        name_to_id[built.name] = ref.id
    return tools, name_to_id
