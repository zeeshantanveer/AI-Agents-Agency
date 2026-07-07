from __future__ import annotations

from typing import Any

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from sqlalchemy import text

from tools._db import get_sync_engine
from tools.base import ToolDefinition
from tools.vector_search.embeddings import EmbeddingUnavailable, embed_text

_DESCRIPTION = (
    "Searches a collection of previously ingested documents by semantic similarity and "
    "returns the most relevant chunks. Use when the request requires information from "
    "the agent's own knowledge base/documents rather than the public web."
)


class VectorSearchInput(BaseModel):
    query: str = Field(description="What to search for.")
    top_k: int = Field(default=5, ge=1, le=20)


class VectorSearchTool(BaseTool):
    name: str = "vector_search_query"
    description: str = _DESCRIPTION
    args_schema: type[BaseModel] = VectorSearchInput
    collection: str = "default"

    def _run(self, query: str, top_k: int = 5) -> str:
        try:
            embedding = embed_text(query)
        except EmbeddingUnavailable as exc:
            return f"vector_search is not configured: {exc}"

        engine = get_sync_engine()
        with engine.connect() as conn:
            rows = conn.execute(
                text(
                    "SELECT content, embedding <=> (:embedding)::vector AS distance "
                    "FROM vector_documents WHERE collection = :collection "
                    "ORDER BY distance ASC LIMIT :top_k"
                ),
                {"embedding": str(embedding), "collection": self.collection, "top_k": top_k},
            ).fetchall()

        if not rows:
            return f"No documents found in collection '{self.collection}'."
        return "\n\n".join(f"[distance={row.distance:.4f}] {row.content[:500]}" for row in rows)


def build_tool(config: dict[str, Any]) -> BaseTool:
    tool = VectorSearchTool()
    if "collection" in config:
        tool.collection = config["collection"]
    return tool


TOOL_DEFINITION = ToolDefinition(
    id="vector_search.query",
    name="Vector Search",
    description=_DESCRIPTION,
    category="research",
    factory=build_tool,
    sensitive=False,
    requires_credentials=["OPENAI_API_KEY"],
)
