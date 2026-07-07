from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "AI Agents Agency"
    auth_mode: str = "none"  # none | basic | jwt
    secret_key: str = "dev-insecure-secret-change-me"

    database_url: str = "postgresql+asyncpg://agency:agency@localhost:5432/agency"
    redis_url: str = "redis://localhost:6379/0"

    anthropic_api_key: str | None = None
    openai_api_key: str | None = None

    @property
    def psycopg_database_url(self) -> str:
        """Plain `postgresql://` DSN for psycopg (langgraph's Postgres checkpointer),
        derived from the SQLAlchemy-style `postgresql+asyncpg://` database_url."""
        return self.database_url.replace("postgresql+asyncpg://", "postgresql://")


@lru_cache
def get_settings() -> Settings:
    return Settings()
