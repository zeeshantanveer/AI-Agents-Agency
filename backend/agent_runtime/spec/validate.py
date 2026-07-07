"""Structural + semantic validation for AgentSpec, beyond what pydantic checks alone."""

from __future__ import annotations

from dataclasses import dataclass, field

from agent_runtime.spec.models import AgentSpec


@dataclass
class ValidationResult:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def validate_spec(spec: AgentSpec, *, known_tool_ids: set[str] | None = None) -> ValidationResult:
    result = ValidationResult()

    if spec.graph.type == "custom_dag":
        node_ids = {n.id for n in spec.graph.nodes}
        if not node_ids:
            result.errors.append("custom_dag graph requires at least one node")
        for edge in spec.graph.edges:
            if edge.source not in node_ids:
                result.errors.append(f"edge source '{edge.source}' is not a defined node")
            if edge.target not in node_ids and edge.target != "END":
                result.errors.append(f"edge target '{edge.target}' is not a defined node")

    if known_tool_ids is not None:
        for ref in spec.tools:
            if ref.id not in known_tool_ids:
                result.errors.append(f"unknown tool id: {ref.id}")

    if spec.memory.long_term.enabled and not spec.memory.long_term.collection:
        result.errors.append("memory.long_term.enabled requires a collection name")

    if spec.guardrails.budget.max_tool_calls < 1:
        result.errors.append("guardrails.budget.max_tool_calls must be >= 1")
    if spec.guardrails.budget.max_run_seconds < 1:
        result.errors.append("guardrails.budget.max_run_seconds must be >= 1")

    if len(spec.system_prompt.strip()) < 10:
        result.warnings.append("system_prompt is very short; agent behavior may be poorly defined")

    return result
