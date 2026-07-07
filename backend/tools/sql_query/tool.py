"""Read-only SQL query tool.

Blocks write/DDL keywords as a coarse guardrail. This is not a hardened SQL
injection defense (the "attacker" model here is prompt injection steering
the agent's own tool calls, not an untrusted end user submitting raw SQL) —
it's a blunt but effective backstop appropriate for an agent-configured
internal tool. Point `connection_string` at a read-only DB user/replica for
any real deployment.
"""

from __future__ import annotations

import re
from typing import Any

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from sqlalchemy import text

from tools._db import get_sync_engine
from tools.base import ToolDefinition

_DESCRIPTION = (
    "Runs a read-only SQL SELECT query against a configured database connection and "
    "returns the resulting rows. Only SELECT statements are permitted."
)

_WRITE_KEYWORDS = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|TRUNCATE|GRANT|REVOKE|CREATE)\b", re.IGNORECASE
)


class SqlQueryInput(BaseModel):
    query: str = Field(description="A read-only SQL SELECT statement.")


class SqlQueryTool(BaseTool):
    name: str = "sql_query_run"
    description: str = _DESCRIPTION
    args_schema: type[BaseModel] = SqlQueryInput
    connection_string: str | None = None
    max_rows: int = 50

    def _run(self, query: str) -> str:
        stripped = query.strip().rstrip(";")
        if not stripped.lower().startswith("select"):
            return "Only SELECT statements are permitted for this tool."
        if _WRITE_KEYWORDS.search(stripped):
            return "Query rejected: contains a disallowed write/DDL keyword."

        engine = get_sync_engine(self.connection_string)
        with engine.connect() as conn:
            result = conn.execute(text(f"{stripped} LIMIT {self.max_rows}"))
            rows = result.fetchall()
            columns = list(result.keys())

        if not rows:
            return "Query returned no rows."
        header = " | ".join(columns)
        lines = [header, "-" * len(header)]
        lines.extend(" | ".join(str(v) for v in row) for row in rows)
        return "\n".join(lines)


def build_tool(config: dict[str, Any]) -> BaseTool:
    tool = SqlQueryTool()
    if "connection_string" in config:
        tool.connection_string = config["connection_string"]
    if "max_rows" in config:
        tool.max_rows = config["max_rows"]
    return tool


TOOL_DEFINITION = ToolDefinition(
    id="sql_query.run",
    name="SQL Query",
    description=_DESCRIPTION,
    category="business_ops",
    factory=build_tool,
    sensitive=False,
)
