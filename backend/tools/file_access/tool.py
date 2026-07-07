"""Scoped file read/write tools.

Every path is resolved against a workspace root and rejected if it would
escape that root (no `..` traversal, no absolute paths outside the sandbox).
The workspace root defaults to `./data/workspaces/default` but is meant to be
overridden per-run via ToolRef.config (`{"workspace_root": "..."}`) so
concurrent runs don't share a workspace.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from tools.base import ToolDefinition

DEFAULT_WORKSPACE_ROOT = Path("./data/workspaces/default")

_READ_DESCRIPTION = "Reads a text file from the agent's workspace and returns its contents."
_WRITE_DESCRIPTION = "Writes text content to a file in the agent's workspace, creating it if needed."


def _resolve(workspace_root: Path, relative_path: str) -> Path:
    root = workspace_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    candidate = (root / relative_path).resolve()
    if root not in candidate.parents and candidate != root:
        raise ValueError(f"path '{relative_path}' escapes the agent workspace")
    return candidate


class FileReadInput(BaseModel):
    path: str = Field(description="Path relative to the agent's workspace.")


class FileReadTool(BaseTool):
    name: str = "file_access_read"
    description: str = _READ_DESCRIPTION
    args_schema: type[BaseModel] = FileReadInput
    workspace_root: Path = DEFAULT_WORKSPACE_ROOT

    def _run(self, path: str) -> str:
        target = _resolve(self.workspace_root, path)
        if not target.exists():
            return f"File not found: {path}"
        return target.read_text(encoding="utf-8", errors="replace")


class FileWriteInput(BaseModel):
    path: str = Field(description="Path relative to the agent's workspace.")
    content: str = Field(description="Text content to write.")


class FileWriteTool(BaseTool):
    name: str = "file_access_write"
    description: str = _WRITE_DESCRIPTION
    args_schema: type[BaseModel] = FileWriteInput
    workspace_root: Path = DEFAULT_WORKSPACE_ROOT

    def _run(self, path: str, content: str) -> str:
        target = _resolve(self.workspace_root, path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return f"Wrote {len(content)} characters to {path}"


def _workspace_root(config: dict[str, Any]) -> Path:
    return Path(config["workspace_root"]) if "workspace_root" in config else DEFAULT_WORKSPACE_ROOT


def build_read_tool(config: dict[str, Any]) -> BaseTool:
    return FileReadTool(workspace_root=_workspace_root(config))


def build_write_tool(config: dict[str, Any]) -> BaseTool:
    return FileWriteTool(workspace_root=_workspace_root(config))


TOOL_DEFINITIONS = [
    ToolDefinition(
        id="file_access.read",
        name="Read File",
        description=_READ_DESCRIPTION,
        category="filesystem",
        factory=build_read_tool,
        sensitive=False,
    ),
    ToolDefinition(
        id="file_access.write",
        name="Write File",
        description=_WRITE_DESCRIPTION,
        category="filesystem",
        factory=build_write_tool,
        sensitive=False,
    ),
]
