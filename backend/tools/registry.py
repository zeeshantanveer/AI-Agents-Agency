"""Tool discovery and lookup.

Each subpackage under tools/ (other than this file's own helpers) exports a
`tool.py` module with a `TOOL_DEFINITION`. discover() walks the package once
at startup; get()/build() are used by the compiler and the generator's tool
matcher.
"""

from __future__ import annotations

import importlib
import pkgutil
from typing import Any

from langchain_core.tools import BaseTool

from tools.base import ToolDefinition


class UnknownToolError(KeyError):
    pass


class ToolRegistry:
    def __init__(self) -> None:
        self._definitions: dict[str, ToolDefinition] = {}

    def register(self, definition: ToolDefinition) -> None:
        self._definitions[definition.id] = definition

    def get(self, tool_id: str) -> ToolDefinition:
        try:
            return self._definitions[tool_id]
        except KeyError as exc:
            raise UnknownToolError(f"unknown tool id: {tool_id}") from exc

    def build(self, tool_id: str, config: dict[str, Any] | None = None) -> BaseTool:
        return self.get(tool_id).factory(config or {})

    def all(self) -> list[ToolDefinition]:
        return list(self._definitions.values())

    def ids(self) -> set[str]:
        return set(self._definitions)

    def discover(self) -> None:
        import tools as tools_pkg

        for module_info in pkgutil.iter_modules(tools_pkg.__path__):
            if not module_info.ispkg:
                continue
            try:
                module = importlib.import_module(f"tools.{module_info.name}.tool")
            except ModuleNotFoundError:
                continue
            definitions = getattr(module, "TOOL_DEFINITIONS", None)
            if definitions is None:
                single = getattr(module, "TOOL_DEFINITION", None)
                definitions = [single] if single else []
            for definition in definitions:
                self.register(definition)


registry = ToolRegistry()
