import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class RunCreateRequest(BaseModel):
    message: str = Field(description="The input message/task for the agent.")


class RunSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    agent_id: uuid.UUID
    status: str
    cost_usd: float | None
    created_at: datetime
    started_at: datetime | None
    completed_at: datetime | None


class RunDetail(RunSummary):
    input: dict[str, Any]
    output: dict[str, Any] | None
    error: str | None


class ResumeRunRequest(BaseModel):
    approved: bool
