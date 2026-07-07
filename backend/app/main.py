from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.agents import router as agents_router
from app.api.v1.runs import router as runs_router
from app.core.config import get_settings
from app.db.seed import seed_builtin_agents
from tools.registry import registry


@asynccontextmanager
async def lifespan(app: FastAPI):
    registry.discover()
    await seed_builtin_agents()
    yield


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title=settings.app_name, lifespan=lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    app.include_router(agents_router, prefix="/api/v1")
    app.include_router(runs_router, prefix="/api/v1")

    return app


app = create_app()
