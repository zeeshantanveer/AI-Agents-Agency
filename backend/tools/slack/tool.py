from __future__ import annotations

import os
from typing import Any

import httpx
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from tools.base import ToolDefinition

SLACK_API = "https://slack.com/api"

_POST_DESCRIPTION = "Posts a message to a Slack channel."
_READ_DESCRIPTION = "Reads recent messages from a Slack channel."


def _headers() -> dict[str, str]:
    token = os.environ.get("SLACK_BOT_TOKEN")
    return {"Authorization": f"Bearer {token}"} if token else {}


class SlackPostInput(BaseModel):
    channel: str = Field(description="Channel ID or name, e.g. '#support'.")
    text: str = Field(description="Message text.")


class SlackPostTool(BaseTool):
    name: str = "slack_post_message"
    description: str = _POST_DESCRIPTION
    args_schema: type[BaseModel] = SlackPostInput

    def _run(self, channel: str, text: str) -> str:
        if not os.environ.get("SLACK_BOT_TOKEN"):
            return "slack tool is not configured: missing SLACK_BOT_TOKEN credential."
        response = httpx.post(
            f"{SLACK_API}/chat.postMessage",
            headers=_headers(),
            json={"channel": channel, "text": text},
            timeout=15.0,
        )
        response.raise_for_status()
        data = response.json()
        if not data.get("ok"):
            return f"Slack API error: {data.get('error')}"
        return f"Posted to {channel}."


class SlackReadInput(BaseModel):
    channel: str = Field(description="Channel ID to read from.")
    limit: int = Field(default=20, ge=1, le=100)


class SlackReadTool(BaseTool):
    name: str = "slack_read_channel"
    description: str = _READ_DESCRIPTION
    args_schema: type[BaseModel] = SlackReadInput

    def _run(self, channel: str, limit: int = 20) -> str:
        if not os.environ.get("SLACK_BOT_TOKEN"):
            return "slack tool is not configured: missing SLACK_BOT_TOKEN credential."
        response = httpx.get(
            f"{SLACK_API}/conversations.history",
            headers=_headers(),
            params={"channel": channel, "limit": limit},
            timeout=15.0,
        )
        response.raise_for_status()
        data = response.json()
        if not data.get("ok"):
            return f"Slack API error: {data.get('error')}"
        messages = data.get("messages", [])
        if not messages:
            return "No messages."
        return "\n".join(f"[{m.get('user', 'unknown')}] {m.get('text', '')}" for m in messages)


def build_post_tool(config: dict[str, Any]) -> BaseTool:
    return SlackPostTool()


def build_read_tool(config: dict[str, Any]) -> BaseTool:
    return SlackReadTool()


TOOL_DEFINITIONS = [
    ToolDefinition(
        id="slack.post_message",
        name="Post Slack Message",
        description=_POST_DESCRIPTION,
        category="communication",
        factory=build_post_tool,
        sensitive=True,
        requires_credentials=["SLACK_BOT_TOKEN"],
    ),
    ToolDefinition(
        id="slack.read_channel",
        name="Read Slack Channel",
        description=_READ_DESCRIPTION,
        category="communication",
        factory=build_read_tool,
        sensitive=False,
        requires_credentials=["SLACK_BOT_TOKEN"],
    ),
]
