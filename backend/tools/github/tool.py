"""GitHub integration: read a PR's diff, and post a review comment.

Requires GITHUB_TOKEN. `github.post_review_comment` is sensitive — posting
to a real PR should always go through the approval guardrail.
"""

from __future__ import annotations

import os
from typing import Any

import httpx
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from tools.base import ToolDefinition

GITHUB_API = "https://api.github.com"

_READ_DESCRIPTION = "Fetches the unified diff for a GitHub pull request, given 'owner/repo' and a PR number."
_COMMENT_DESCRIPTION = "Posts a top-level review comment on a GitHub pull request."


def _headers() -> dict[str, str]:
    token = os.environ.get("GITHUB_TOKEN")
    headers = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


class ReadPRDiffInput(BaseModel):
    repo: str = Field(description="owner/repo, e.g. 'anthropics/claude-code'")
    pr_number: int = Field(description="Pull request number")


class ReadPRDiffTool(BaseTool):
    name: str = "github_read_pr_diff"
    description: str = _READ_DESCRIPTION
    args_schema: type[BaseModel] = ReadPRDiffInput
    max_diff_size_kb: int = 500

    def _run(self, repo: str, pr_number: int) -> str:
        if not os.environ.get("GITHUB_TOKEN"):
            return "github tool is not configured: missing GITHUB_TOKEN credential."
        url = f"{GITHUB_API}/repos/{repo}/pulls/{pr_number}"
        headers = {**_headers(), "Accept": "application/vnd.github.v3.diff"}
        response = httpx.get(url, headers=headers, timeout=20.0)
        response.raise_for_status()
        diff = response.text
        limit = self.max_diff_size_kb * 1024
        if len(diff) > limit:
            diff = diff[:limit] + "\n...[diff truncated]"
        return diff


class PostReviewCommentInput(BaseModel):
    repo: str = Field(description="owner/repo")
    pr_number: int
    body: str = Field(description="Review comment body (markdown).")


class PostReviewCommentTool(BaseTool):
    name: str = "github_post_review_comment"
    description: str = _COMMENT_DESCRIPTION
    args_schema: type[BaseModel] = PostReviewCommentInput

    def _run(self, repo: str, pr_number: int, body: str) -> str:
        if not os.environ.get("GITHUB_TOKEN"):
            return "github tool is not configured: missing GITHUB_TOKEN credential."
        url = f"{GITHUB_API}/repos/{repo}/issues/{pr_number}/comments"
        response = httpx.post(url, headers=_headers(), json={"body": body}, timeout=20.0)
        response.raise_for_status()
        return f"Posted comment: {response.json().get('html_url')}"


def build_read_tool(config: dict[str, Any]) -> BaseTool:
    tool = ReadPRDiffTool()
    if "max_diff_size_kb" in config:
        tool.max_diff_size_kb = config["max_diff_size_kb"]
    return tool


def build_comment_tool(config: dict[str, Any]) -> BaseTool:
    return PostReviewCommentTool()


TOOL_DEFINITIONS = [
    ToolDefinition(
        id="github.read_pr_diff",
        name="Read PR Diff",
        description=_READ_DESCRIPTION,
        category="developer",
        factory=build_read_tool,
        sensitive=False,
        requires_credentials=["GITHUB_TOKEN"],
    ),
    ToolDefinition(
        id="github.post_review_comment",
        name="Post PR Review Comment",
        description=_COMMENT_DESCRIPTION,
        category="developer",
        factory=build_comment_tool,
        sensitive=True,
        requires_credentials=["GITHUB_TOKEN"],
    ),
]
