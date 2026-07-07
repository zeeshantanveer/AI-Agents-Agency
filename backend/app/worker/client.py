"""Enqueues jobs onto the arq/Redis queue from the FastAPI process.

The API process never executes an agent graph inline — it enqueues here and
the worker process picks the job up.
"""

from __future__ import annotations

from arq import create_pool
from arq.connections import ArqRedis, RedisSettings

from app.core.config import get_settings

_pool: ArqRedis | None = None


async def get_pool() -> ArqRedis:
    global _pool
    if _pool is None:
        _pool = await create_pool(RedisSettings.from_dsn(get_settings().redis_url))
    return _pool


async def enqueue(function_name: str, *args: object) -> None:
    pool = await get_pool()
    await pool.enqueue_job(function_name, *args)
