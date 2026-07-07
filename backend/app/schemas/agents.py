import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict


class AgentSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    slug: str
    name: str
    description: str
    category: str
    origin: str
    version: str
    is_active: bool


class AgentDetail(AgentSummary):
    spec: dict[str, Any]
    created_at: datetime
    updated_at: datetime
