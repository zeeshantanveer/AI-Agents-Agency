from arq.connections import RedisSettings
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from app.core.config import get_settings
from app.worker.tasks import resume_run_task, run_agent_task
from tools.registry import registry

functions: list = [run_agent_task, resume_run_task]


async def ping(ctx: dict) -> str:
    return "pong"


functions.append(ping)


async def startup(ctx: dict) -> None:
    registry.discover()
    async with AsyncPostgresSaver.from_conn_string(get_settings().psycopg_database_url) as saver:
        await saver.setup()


async def shutdown(ctx: dict) -> None:
    pass


class WorkerSettings:
    functions = functions
    on_startup = startup
    on_shutdown = shutdown
    redis_settings = RedisSettings.from_dsn(get_settings().redis_url)
