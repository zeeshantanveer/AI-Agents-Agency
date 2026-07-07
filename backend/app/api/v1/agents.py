import uuid

from fastapi import APIRouter, HTTPException
from sqlmodel import select

from app.db.models import Agent
from app.db.session import async_session_factory
from app.schemas.agents import AgentDetail, AgentSummary

router = APIRouter(prefix="/agents", tags=["agents"])


@router.get("", response_model=list[AgentSummary])
async def list_agents(category: str | None = None) -> list[Agent]:
    async with async_session_factory() as session:
        stmt = select(Agent).where(Agent.is_active == True)  # noqa: E712
        if category:
            stmt = stmt.where(Agent.category == category)
        result = await session.execute(stmt.order_by(Agent.name))
        return list(result.scalars().all())


@router.get("/{agent_id}", response_model=AgentDetail)
async def get_agent(agent_id: uuid.UUID) -> Agent:
    async with async_session_factory() as session:
        agent = await session.get(Agent, agent_id)
        if agent is None:
            raise HTTPException(status_code=404, detail="agent not found")
        return agent
