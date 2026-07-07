"""The contract every tool integration implements.

A tool module (tools/<name>/tool.py) exports a module-level `TOOL_DEFINITION`
built from `ToolDefinition`. The `description` field is what the
prompt-to-agent generator's tool matcher embeds and semantically searches
against, and what the manual tool-picker UI shows — write it for an LLM
audience, not a human API doc.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from langchain_core.tools import BaseTool


@dataclass(frozen=True)
class RateLimit:
    calls_per_minute: int = 30


@dataclass(frozen=True)
class ToolDefinition:
    id: str
    name: str
    description: str
    category: str
    factory: Callable[[dict[str, Any]], BaseTool]
    sensitive: bool = False
    requires_credentials: list[str] = field(default_factory=list)
    rate_limit: RateLimit = field(default_factory=RateLimit)
    sandbox: str = "none"  # none | subprocess | container
