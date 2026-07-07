"""Web search tool, backed by the Tavily search API.

Requires TAVILY_API_KEY. Implemented as a direct httpx call rather than an
extra SDK dependency, since Tavily's API is a single simple POST endpoint.
"""

from __future__ import annotations

import os
from typing import Any

import httpx
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from tools.base import RateLimit, ToolDefinition

TAVILY_ENDPOINT = "https://api.tavily.com/search"


_DESCRIPTION = (
    "Searches the public web and returns ranked results with titles, snippets, "
    "and URLs. Use when the request requires current information not already "
    "known or present in provided documents."
)


class WebSearchInput(BaseModel):
    query: str = Field(description="The search query.")
    max_results: int = Field(default=5, ge=1, le=10)


class WebSearchTool(BaseTool):
    name: str = "web_search"
    description: str = _DESCRIPTION
    args_schema: type[BaseModel] = WebSearchInput
    max_results_default: int = 5

    def _run(self, query: str, max_results: int = 5) -> str:
        api_key = os.environ.get("TAVILY_API_KEY")
        if not api_key:
            return "web_search is not configured: missing TAVILY_API_KEY credential."
        response = httpx.post(
            TAVILY_ENDPOINT,
            json={"api_key": api_key, "query": query, "max_results": max_results},
            timeout=15.0,
        )
        response.raise_for_status()
        results = response.json().get("results", [])
        if not results:
            return "No results found."
        lines = []
        for r in results:
            lines.append(f"- {r.get('title')}\n  {r.get('url')}\n  {r.get('content', '')[:300]}")
        return "\n".join(lines)

    async def _arun(self, query: str, max_results: int = 5) -> str:
        api_key = os.environ.get("TAVILY_API_KEY")
        if not api_key:
            return "web_search is not configured: missing TAVILY_API_KEY credential."
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.post(
                TAVILY_ENDPOINT,
                json={"api_key": api_key, "query": query, "max_results": max_results},
            )
        response.raise_for_status()
        results = response.json().get("results", [])
        if not results:
            return "No results found."
        lines = []
        for r in results:
            lines.append(f"- {r.get('title')}\n  {r.get('url')}\n  {r.get('content', '')[:300]}")
        return "\n".join(lines)


def build_tool(config: dict[str, Any]) -> BaseTool:
    tool = WebSearchTool()
    if "max_results" in config:
        tool.max_results_default = config["max_results"]
    return tool


TOOL_DEFINITION = ToolDefinition(
    id="web_search.search",
    name="Web Search",
    description=_DESCRIPTION,
    category="search",
    factory=build_tool,
    sensitive=False,
    requires_credentials=["TAVILY_API_KEY"],
    rate_limit=RateLimit(calls_per_minute=30),
)
