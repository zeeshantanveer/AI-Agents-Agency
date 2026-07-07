"""Shared sync SQLAlchemy engine for tools that need direct DB access
(vector_search, sql_query). Deliberately synchronous — tool `_run` methods
execute inside a LangGraph node function called via a thread-pool executor,
not the main async event loop, so a sync engine avoids any async/sync
bridging complexity.

Not a tool package itself (no tool.py), so `ToolRegistry.discover()`'s
pkgutil walk skips it.
"""

from __future__ import annotations

from functools import lru_cache

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

from app.core.config import get_settings


@lru_cache
def get_sync_engine(connection_string: str | None = None) -> Engine:
    dsn = connection_string or get_settings().database_url.replace(
        "postgresql+asyncpg://", "postgresql+psycopg://"
    )
    return create_engine(dsn, pool_pre_ping=True)
