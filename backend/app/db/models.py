"""SQLModel table definitions. Everything here is auto-discovered by Alembic
via `target_metadata = SQLModel.metadata` in alembic/env.py.

Enum-shaped fields (status, origin, category, ...) are plain `str` columns
rather than DB-level enums, deliberately — adding a new status value should
never require a migration.

All timestamps are timezone-aware (`TIMESTAMPTZ`) and stored in UTC — asyncpg
rejects binding an aware `datetime` to a naive `TIMESTAMP` column, so every
datetime column below is explicit about `timezone=True`.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import Column, DateTime
from sqlalchemy.dialects.postgresql import JSONB
from sqlmodel import Field, SQLModel

EMBEDDING_DIM = 1536  # matches OpenAI text-embedding-3-small


def _uuid() -> uuid.UUID:
    return uuid.uuid4()


def _now() -> datetime:
    return datetime.now(UTC)


def _timestamp_column(*, nullable: bool = False) -> Column:
    return Column(DateTime(timezone=True), nullable=nullable)


class User(SQLModel, table=True):
    __tablename__ = "users"

    id: uuid.UUID = Field(default_factory=_uuid, primary_key=True)
    email: str = Field(unique=True, index=True)
    hashed_password: str | None = None
    role: str = Field(default="admin")  # admin | member
    created_at: datetime = Field(default_factory=_now, sa_column=_timestamp_column())


class Agent(SQLModel, table=True):
    __tablename__ = "agents"

    id: uuid.UUID = Field(default_factory=_uuid, primary_key=True)
    slug: str = Field(unique=True, index=True)
    name: str
    description: str
    category: str  # developer | research | business_ops | custom
    origin: str  # built_in | generated
    version: str = Field(default="1.0.0")
    spec: dict[str, Any] = Field(sa_column=Column(JSONB, nullable=False))
    created_by: uuid.UUID | None = Field(default=None, foreign_key="users.id")
    created_at: datetime = Field(default_factory=_now, sa_column=_timestamp_column())
    updated_at: datetime = Field(default_factory=_now, sa_column=_timestamp_column())
    is_active: bool = Field(default=True)


class Run(SQLModel, table=True):
    __tablename__ = "runs"

    id: uuid.UUID = Field(default_factory=_uuid, primary_key=True)
    agent_id: uuid.UUID = Field(foreign_key="agents.id", index=True)
    thread_id: str = Field(index=True)
    status: str = Field(default="queued", index=True)
    # queued | running | waiting_approval | succeeded | failed | cancelled
    input: dict[str, Any] = Field(sa_column=Column(JSONB, nullable=False))
    output: dict[str, Any] | None = Field(default=None, sa_column=Column(JSONB))
    error: str | None = None
    started_at: datetime | None = Field(default=None, sa_column=_timestamp_column(nullable=True))
    completed_at: datetime | None = Field(default=None, sa_column=_timestamp_column(nullable=True))
    cost_usd: float | None = None
    tokens_used: int | None = None
    triggered_by: uuid.UUID | None = Field(default=None, foreign_key="users.id")
    created_at: datetime = Field(default_factory=_now, sa_column=_timestamp_column())


class RunEvent(SQLModel, table=True):
    __tablename__ = "run_events"

    id: int | None = Field(default=None, primary_key=True)
    run_id: uuid.UUID = Field(foreign_key="runs.id", index=True)
    seq: int
    # node_start | node_end | tool_call | tool_result | token | error | interrupt | approval_resolved
    type: str
    payload: dict[str, Any] = Field(sa_column=Column(JSONB, nullable=False))
    created_at: datetime = Field(default_factory=_now, sa_column=_timestamp_column())


class ToolInvocation(SQLModel, table=True):
    __tablename__ = "tool_invocations"

    id: uuid.UUID = Field(default_factory=_uuid, primary_key=True)
    run_id: uuid.UUID = Field(foreign_key="runs.id", index=True)
    tool_id: str
    args: dict[str, Any] = Field(sa_column=Column(JSONB, nullable=False))
    result_summary: dict[str, Any] | None = Field(default=None, sa_column=Column(JSONB))
    status: str = Field(default="ok")  # ok | error | denied
    created_at: datetime = Field(default_factory=_now, sa_column=_timestamp_column())


class ToolCredential(SQLModel, table=True):
    __tablename__ = "tool_credentials"

    id: uuid.UUID = Field(default_factory=_uuid, primary_key=True)
    owner_id: uuid.UUID | None = Field(default=None, foreign_key="users.id")
    credential_name: str = Field(index=True)
    encrypted_value: bytes
    created_at: datetime = Field(default_factory=_now, sa_column=_timestamp_column())
    updated_at: datetime = Field(default_factory=_now, sa_column=_timestamp_column())


class Generation(SQLModel, table=True):
    __tablename__ = "generations"

    id: uuid.UUID = Field(default_factory=_uuid, primary_key=True)
    status: str = Field(default="extracting")
    # extracting | matching | assembling | validating | ready_for_review | confirmed | discarded
    prompt: str
    intent_spec: dict[str, Any] | None = Field(default=None, sa_column=Column(JSONB))
    draft_spec: dict[str, Any] | None = Field(default=None, sa_column=Column(JSONB))
    generation_report: dict[str, Any] | None = Field(default=None, sa_column=Column(JSONB))
    validation_result: dict[str, Any] | None = Field(default=None, sa_column=Column(JSONB))
    confirmed_agent_id: uuid.UUID | None = Field(default=None, foreign_key="agents.id")
    created_by: uuid.UUID | None = Field(default=None, foreign_key="users.id")
    created_at: datetime = Field(default_factory=_now, sa_column=_timestamp_column())


class VectorDocument(SQLModel, table=True):
    """Backs the vector_search tool and document_loader.ingest — a flat
    chunk store, partitioned by `collection` (an AgentSpec's
    memory.long_term.collection name)."""

    __tablename__ = "vector_documents"

    id: uuid.UUID = Field(default_factory=_uuid, primary_key=True)
    collection: str = Field(index=True)
    content: str
    doc_metadata: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSONB))
    embedding: list[float] = Field(sa_column=Column(Vector(EMBEDDING_DIM)))
    created_at: datetime = Field(default_factory=_now, sa_column=_timestamp_column())
