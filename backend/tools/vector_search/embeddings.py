"""Shared OpenAI embeddings helper, used by both vector_search and
document_loader. Anthropic has no public embeddings API, so OpenAI is the
practical default here regardless of which provider an agent's chat model
uses — documented as a v1 constraint, not a bug.
"""

from __future__ import annotations

import os

import httpx

OPENAI_EMBEDDINGS_URL = "https://api.openai.com/v1/embeddings"
EMBEDDING_MODEL = "text-embedding-3-small"


class EmbeddingUnavailable(RuntimeError):
    pass


def embed_text(text: str) -> list[float]:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EmbeddingUnavailable("missing OPENAI_API_KEY credential (required for embeddings)")
    response = httpx.post(
        OPENAI_EMBEDDINGS_URL,
        headers={"Authorization": f"Bearer {api_key}"},
        json={"model": EMBEDDING_MODEL, "input": text},
        timeout=30.0,
    )
    response.raise_for_status()
    return response.json()["data"][0]["embedding"]
