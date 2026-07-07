from __future__ import annotations

import re
from typing import Any

import httpx
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from sqlalchemy import text as sql_text

from tools._db import get_sync_engine
from tools.base import ToolDefinition
from tools.vector_search.embeddings import EmbeddingUnavailable, embed_text

_DESCRIPTION = (
    "Ingests a document (raw text or a URL) into the agent's knowledge base collection "
    "so it can later be found via vector_search. Use when asked to remember, index, or "
    "add a document/page to the knowledge base."
)

_TAG_RE = re.compile(r"<[^>]+>")


def _chunk(content: str, size: int = 1000, overlap: int = 100) -> list[str]:
    chunks = []
    start = 0
    while start < len(content):
        chunks.append(content[start : start + size])
        start += size - overlap
    return [c for c in chunks if c.strip()]


class DocumentLoaderInput(BaseModel):
    text: str | None = Field(default=None, description="Raw text to ingest.")
    url: str | None = Field(default=None, description="A URL to fetch and ingest.")


class DocumentLoaderTool(BaseTool):
    name: str = "document_loader_ingest"
    description: str = _DESCRIPTION
    args_schema: type[BaseModel] = DocumentLoaderInput
    collection: str = "default"

    def _run(self, text: str | None = None, url: str | None = None) -> str:
        if not text and not url:
            return "Provide either 'text' or 'url' to ingest."

        content = text or ""
        if url:
            try:
                response = httpx.get(url, timeout=20.0, follow_redirects=True)
                response.raise_for_status()
                content = _TAG_RE.sub(" ", response.text)
            except httpx.HTTPError as exc:
                return f"Failed to fetch {url}: {exc}"

        chunks = _chunk(content)
        if not chunks:
            return "Nothing to ingest (empty content)."

        engine = get_sync_engine()
        ingested = 0
        with engine.begin() as conn:
            for chunk in chunks:
                try:
                    embedding = embed_text(chunk)
                except EmbeddingUnavailable as exc:
                    return f"document_loader is not configured: {exc}"
                conn.execute(
                    sql_text(
                        "INSERT INTO vector_documents "
                        "(id, collection, content, doc_metadata, embedding, created_at) "
                        "VALUES "
                        "(gen_random_uuid(), :collection, :content, '{}'::jsonb, (:embedding)::vector, now())"
                    ),
                    {"collection": self.collection, "content": chunk, "embedding": str(embedding)},
                )
                ingested += 1
        return f"Ingested {ingested} chunk(s) into collection '{self.collection}'."


def build_tool(config: dict[str, Any]) -> BaseTool:
    tool = DocumentLoaderTool()
    if "collection" in config:
        tool.collection = config["collection"]
    return tool


TOOL_DEFINITION = ToolDefinition(
    id="document_loader.ingest",
    name="Document Loader",
    description=_DESCRIPTION,
    category="research",
    factory=build_tool,
    sensitive=False,
    requires_credentials=["OPENAI_API_KEY"],
)
