"""Generic CRM REST adapter — config-driven so one tool implementation can
point at HubSpot, Pipedrive, or any other CRM's REST API via a per-agent
`base_url` + a named credential, rather than hand-rolling a client per CRM.
"""

from __future__ import annotations

import os
from typing import Any

import httpx
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from tools.base import ToolDefinition

_QUERY_DESCRIPTION = (
    "Queries a generic CRM REST API (HubSpot/Pipedrive/etc, configured per-agent) for "
    "records matching a path and query parameters."
)
_WRITE_DESCRIPTION = "Writes or updates a record in a generic CRM REST API (configured per-agent)."


class CrmQueryInput(BaseModel):
    path: str = Field(description="API path relative to the configured base URL, e.g. '/contacts'.")
    params: dict[str, Any] = Field(default_factory=dict)


class CrmGenericQueryTool(BaseTool):
    name: str = "crm_generic_query"
    description: str = _QUERY_DESCRIPTION
    args_schema: type[BaseModel] = CrmQueryInput
    base_url: str = ""
    credential_name: str = "CRM_API_KEY"

    def _run(self, path: str, params: dict[str, Any] | None = None) -> str:
        if not self.base_url:
            return "crm_generic is not configured: missing base_url in tool config."
        api_key = os.environ.get(self.credential_name)
        if not api_key:
            return f"crm_generic is not configured: missing {self.credential_name} credential."
        response = httpx.get(
            f"{self.base_url.rstrip('/')}/{path.lstrip('/')}",
            headers={"Authorization": f"Bearer {api_key}"},
            params=params or {},
            timeout=20.0,
        )
        response.raise_for_status()
        return str(response.json())[:4000]


class CrmWriteInput(BaseModel):
    path: str
    body: dict[str, Any] = Field(default_factory=dict)


class CrmGenericWriteTool(BaseTool):
    name: str = "crm_generic_write"
    description: str = _WRITE_DESCRIPTION
    args_schema: type[BaseModel] = CrmWriteInput
    base_url: str = ""
    credential_name: str = "CRM_API_KEY"

    def _run(self, path: str, body: dict[str, Any] | None = None) -> str:
        if not self.base_url:
            return "crm_generic is not configured: missing base_url in tool config."
        api_key = os.environ.get(self.credential_name)
        if not api_key:
            return f"crm_generic is not configured: missing {self.credential_name} credential."
        response = httpx.post(
            f"{self.base_url.rstrip('/')}/{path.lstrip('/')}",
            headers={"Authorization": f"Bearer {api_key}"},
            json=body or {},
            timeout=20.0,
        )
        response.raise_for_status()
        return str(response.json())[:4000]


def build_query_tool(config: dict[str, Any]) -> BaseTool:
    tool = CrmGenericQueryTool()
    tool.base_url = config.get("base_url", "")
    tool.credential_name = config.get("credential_name", "CRM_API_KEY")
    return tool


def build_write_tool(config: dict[str, Any]) -> BaseTool:
    tool = CrmGenericWriteTool()
    tool.base_url = config.get("base_url", "")
    tool.credential_name = config.get("credential_name", "CRM_API_KEY")
    return tool


TOOL_DEFINITIONS = [
    ToolDefinition(
        id="crm_generic.query",
        name="CRM Query",
        description=_QUERY_DESCRIPTION,
        category="business_ops",
        factory=build_query_tool,
        sensitive=False,
    ),
    ToolDefinition(
        id="crm_generic.write",
        name="CRM Write",
        description=_WRITE_DESCRIPTION,
        category="business_ops",
        factory=build_write_tool,
        sensitive=True,
    ),
]
