"""AgentSpec: the single format both built-in and generated agents conform to.

Pydantic is the source of truth. Built-in agents serialize this as YAML on
disk (agents_library/*/agent.yaml); generated agents serialize it as JSONB
in the `agents` table. Both are loaded through AgentSpec.model_validate(...).
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

SCHEMA_VERSION = "1.0"


class Origin(StrEnum):
    built_in = "built_in"
    generated = "generated"


class Category(StrEnum):
    developer = "developer"
    research = "research"
    business_ops = "business_ops"
    custom = "custom"


class ModelFallback(BaseModel):
    provider: str
    model_name: str


class ModelConfig(BaseModel):
    provider: str = Field(description="e.g. anthropic, openai")
    model_name: str
    temperature: float = 0.2
    max_tokens: int = 4096
    fallback: list[ModelFallback] = Field(default_factory=list)


class ToolRef(BaseModel):
    id: str = Field(description="Tool registry id, e.g. 'web_search.search'")
    config: dict[str, Any] = Field(default_factory=dict)


class ShortTermMemory(BaseModel):
    type: Literal["checkpoint"] = "checkpoint"
    window: int = 20


class LongTermMemory(BaseModel):
    enabled: bool = False
    type: Literal["pgvector"] = "pgvector"
    collection: str | None = None
    top_k: int = 5


class MemoryConfig(BaseModel):
    short_term: ShortTermMemory = Field(default_factory=ShortTermMemory)
    long_term: LongTermMemory = Field(default_factory=LongTermMemory)


class GraphNode(BaseModel):
    id: str
    kind: Literal["llm_call", "tool_call", "router", "human_review"] = "llm_call"
    config: dict[str, Any] = Field(default_factory=dict)


class GraphEdge(BaseModel):
    source: str
    target: str
    condition: str | None = None  # name of a router function, None = unconditional


class GraphConfig(BaseModel):
    type: Literal["react", "linear_pipeline", "custom_dag"] = "react"
    max_iterations: int = 15
    nodes: list[GraphNode] = Field(default_factory=list)
    edges: list[GraphEdge] = Field(default_factory=list)
    interrupts: list[str] = Field(default_factory=list)


class InputGuardrails(BaseModel):
    max_input_tokens: int = 8000
    blocked_patterns: list[str] = Field(default_factory=list)


class OutputGuardrails(BaseModel):
    pii_redaction: bool = False
    max_output_tokens: int = 4096


class BudgetGuardrails(BaseModel):
    max_tool_calls: int = 10
    max_run_seconds: int = 300
    max_cost_usd: float = 2.00


class ApprovalGuardrails(BaseModel):
    require_human_approval_for: list[str] = Field(default_factory=list)


class GuardrailsConfig(BaseModel):
    input: InputGuardrails = Field(default_factory=InputGuardrails)
    output: OutputGuardrails = Field(default_factory=OutputGuardrails)
    budget: BudgetGuardrails = Field(default_factory=BudgetGuardrails)
    approvals: ApprovalGuardrails = Field(default_factory=ApprovalGuardrails)


class Trigger(BaseModel):
    type: Literal["api"] = "api"
    enabled: bool = True


class AgentMetadata(BaseModel):
    icon: str = "bot"
    color: str = "#6366f1"


class AgentSpec(BaseModel):
    schema_version: str = SCHEMA_VERSION
    id: str = Field(description="Stable slug, e.g. 'code-reviewer'")
    name: str
    description: str
    category: Category = Category.custom
    version: str = "1.0.0"
    origin: Origin = Origin.built_in
    created_by: str | None = None
    tags: list[str] = Field(default_factory=list)

    model: ModelConfig
    system_prompt: str
    instructions: dict[str, Any] = Field(default_factory=dict)

    tools: list[ToolRef] = Field(default_factory=list)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    graph: GraphConfig = Field(default_factory=GraphConfig)
    guardrails: GuardrailsConfig = Field(default_factory=GuardrailsConfig)
    triggers: list[Trigger] = Field(default_factory=lambda: [Trigger()])

    output_schema: dict[str, Any] | None = None
    metadata: AgentMetadata = Field(default_factory=AgentMetadata)

    @field_validator("id")
    @classmethod
    def _slug_shape(cls, v: str) -> str:
        if not v or not all(c.isalnum() or c in "-_" for c in v):
            raise ValueError("id must be a slug of [a-zA-Z0-9-_]")
        return v

    @field_validator("tools")
    @classmethod
    def _unique_tool_ids(cls, v: list[ToolRef]) -> list[ToolRef]:
        seen = set()
        for ref in v:
            if ref.id in seen:
                raise ValueError(f"duplicate tool reference: {ref.id}")
            seen.add(ref.id)
        return v
