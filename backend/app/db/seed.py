"""Upserts agents_library/*/agent.yaml into the `agents` table, keyed by slug.

Called on backend startup (so a fresh `docker compose up` always has the
built-in agents available) and from infra/scripts/seed_agents.py for manual
re-seeding.
"""

from __future__ import annotations

from sqlmodel import select

from agent_runtime.spec.loader import load_builtin_specs
from app.db.models import Agent
from app.db.session import async_session_factory


async def seed_builtin_agents() -> None:
    specs = load_builtin_specs()
    async with async_session_factory() as session:
        for spec in specs:
            existing = (
                await session.execute(select(Agent).where(Agent.slug == spec.id))
            ).scalar_one_or_none()
            spec_dict = spec.model_dump(mode="json")
            if existing:
                existing.name = spec.name
                existing.description = spec.description
                existing.category = spec.category.value
                existing.version = spec.version
                existing.spec = spec_dict
                session.add(existing)
            else:
                session.add(
                    Agent(
                        slug=spec.id,
                        name=spec.name,
                        description=spec.description,
                        category=spec.category.value,
                        origin="built_in",
                        version=spec.version,
                        spec=spec_dict,
                    )
                )
        await session.commit()
