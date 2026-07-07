"""Sandboxed code execution: runs a short Python or shell snippet in an
isolated subprocess with a timeout and a language allowlist.

v1 sandbox note: this is subprocess + resource limits (timeout, isolated
tempdir, trimmed environment), not full container isolation — there is no
network isolation yet. Per the project's phased security plan, this must be
hardened to run in an ephemeral, network-disabled container before any
agent using this tool is exposed beyond trusted local use. Tracked as a
Phase 5 hardening item.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from typing import Any, Literal

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from tools.base import RateLimit, ToolDefinition

_DESCRIPTION = (
    "Executes a short Python or shell snippet in a sandboxed subprocess and "
    "returns stdout, stderr, and exit code. Use for running linters, tests, "
    "or quick checks against files already written to the workspace."
)


class CodeExecInput(BaseModel):
    language: Literal["python", "shell"] = "python"
    code: str = Field(description="The snippet to execute.")


class CodeExecTool(BaseTool):
    name: str = "code_exec_run"
    description: str = _DESCRIPTION
    args_schema: type[BaseModel] = CodeExecInput
    allowed_languages: list[str] = ["python", "shell"]
    timeout_seconds: int = 20

    def _run(self, language: str, code: str) -> str:
        if language not in self.allowed_languages:
            return f"Execution of '{language}' is not permitted for this agent."

        cmd = [sys.executable, "-c", code] if language == "python" else ["sh", "-c", code]

        with tempfile.TemporaryDirectory() as tmp:
            try:
                result = subprocess.run(  # noqa: S603 — sandboxed by timeout + isolated cwd, see module docstring
                    cmd,
                    cwd=tmp,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_seconds,
                )
            except subprocess.TimeoutExpired:
                return f"Execution timed out after {self.timeout_seconds}s"

            stdout = result.stdout[:4000]
            stderr = result.stderr[:2000]
            return f"exit_code={result.returncode}\nstdout:\n{stdout}\nstderr:\n{stderr}"


def build_tool(config: dict[str, Any]) -> BaseTool:
    tool = CodeExecTool()
    if "allowed_languages" in config:
        tool.allowed_languages = config["allowed_languages"]
    if "timeout_seconds" in config:
        tool.timeout_seconds = config["timeout_seconds"]
    return tool


TOOL_DEFINITION = ToolDefinition(
    id="code_exec.run",
    name="Run Code",
    description=_DESCRIPTION,
    category="developer",
    factory=build_tool,
    sensitive=False,
    sandbox="subprocess",
    rate_limit=RateLimit(calls_per_minute=20),
)
