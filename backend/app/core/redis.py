"""Shared async Redis client, used for the run-event pub/sub relay
(worker publishes, the SSE endpoint subscribes) and, later, rate limiting.
"""

from __future__ import annotations

from redis import asyncio as redis_asyncio

from app.core.config import get_settings

_client: redis_asyncio.Redis | None = None


def get_redis() -> redis_asyncio.Redis:
    global _client
    if _client is None:
        _client = redis_asyncio.from_url(get_settings().redis_url, decode_responses=True)
    return _client


def run_channel(run_id: str) -> str:
    return f"run:{run_id}"
