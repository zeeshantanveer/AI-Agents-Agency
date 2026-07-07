"""Google Calendar integration.

v1 constraint: expects a pre-provisioned OAuth access token in
GOOGLE_CALENDAR_ACCESS_TOKEN. A full OAuth authorization-code flow (token
acquisition + refresh) is out of scope for v1 — the token must be obtained
externally and set as a credential. Tracked as a future hardening item
alongside multi-user auth.
"""

from __future__ import annotations

import os
from typing import Any

import httpx
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from tools.base import ToolDefinition

CALENDAR_API = "https://www.googleapis.com/calendar/v3"

_READ_DESCRIPTION = "Reads upcoming events from a Google Calendar."
_CREATE_DESCRIPTION = "Creates a new event on a Google Calendar."


def _headers() -> dict[str, str]:
    token = os.environ.get("GOOGLE_CALENDAR_ACCESS_TOKEN")
    return {"Authorization": f"Bearer {token}"} if token else {}


class CalendarReadInput(BaseModel):
    calendar_id: str = Field(default="primary")
    max_results: int = Field(default=10, ge=1, le=50)


class CalendarReadTool(BaseTool):
    name: str = "calendar_read"
    description: str = _READ_DESCRIPTION
    args_schema: type[BaseModel] = CalendarReadInput

    def _run(self, calendar_id: str = "primary", max_results: int = 10) -> str:
        if not os.environ.get("GOOGLE_CALENDAR_ACCESS_TOKEN"):
            return "calendar tool is not configured: missing GOOGLE_CALENDAR_ACCESS_TOKEN credential."
        response = httpx.get(
            f"{CALENDAR_API}/calendars/{calendar_id}/events",
            headers=_headers(),
            params={"maxResults": max_results, "singleEvents": "true", "orderBy": "startTime"},
            timeout=15.0,
        )
        response.raise_for_status()
        items = response.json().get("items", [])
        if not items:
            return "No upcoming events."
        lines = []
        for item in items:
            start = item.get("start", {}).get("dateTime", item.get("start", {}).get("date"))
            lines.append(f"- {start}: {item.get('summary', '(no title)')}")
        return "\n".join(lines)


class CalendarCreateEventInput(BaseModel):
    calendar_id: str = Field(default="primary")
    summary: str
    start_iso: str = Field(description="ISO 8601 start datetime, e.g. 2026-01-01T10:00:00-05:00")
    end_iso: str = Field(description="ISO 8601 end datetime.")


class CalendarCreateEventTool(BaseTool):
    name: str = "calendar_create_event"
    description: str = _CREATE_DESCRIPTION
    args_schema: type[BaseModel] = CalendarCreateEventInput

    def _run(self, summary: str, start_iso: str, end_iso: str, calendar_id: str = "primary") -> str:
        if not os.environ.get("GOOGLE_CALENDAR_ACCESS_TOKEN"):
            return "calendar tool is not configured: missing GOOGLE_CALENDAR_ACCESS_TOKEN credential."
        response = httpx.post(
            f"{CALENDAR_API}/calendars/{calendar_id}/events",
            headers=_headers(),
            json={"summary": summary, "start": {"dateTime": start_iso}, "end": {"dateTime": end_iso}},
            timeout=15.0,
        )
        response.raise_for_status()
        return f"Created event: {response.json().get('htmlLink')}"


def build_read_tool(config: dict[str, Any]) -> BaseTool:
    return CalendarReadTool()


def build_create_tool(config: dict[str, Any]) -> BaseTool:
    return CalendarCreateEventTool()


TOOL_DEFINITIONS = [
    ToolDefinition(
        id="calendar.read",
        name="Read Calendar",
        description=_READ_DESCRIPTION,
        category="productivity",
        factory=build_read_tool,
        sensitive=False,
        requires_credentials=["GOOGLE_CALENDAR_ACCESS_TOKEN"],
    ),
    ToolDefinition(
        id="calendar.create_event",
        name="Create Calendar Event",
        description=_CREATE_DESCRIPTION,
        category="productivity",
        factory=build_create_tool,
        sensitive=True,
        requires_credentials=["GOOGLE_CALENDAR_ACCESS_TOKEN"],
    ),
]
