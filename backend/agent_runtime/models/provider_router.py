"""Provider-agnostic chat model resolution.

Swapping Anthropic for OpenAI (or adding a fallback chain) is a config
change on AgentSpec.model, never a code change. New providers are added by
extending _PROVIDER_FACTORIES.
"""

from __future__ import annotations

from collections.abc import Callable

from langchain_core.language_models.chat_models import BaseChatModel

from agent_runtime.spec.models import ModelConfig
from app.core.config import get_settings


def _anthropic(model_name: str, temperature: float, max_tokens: int) -> BaseChatModel:
    from langchain_anthropic import ChatAnthropic

    settings = get_settings()
    return ChatAnthropic(
        model=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        api_key=settings.anthropic_api_key,
    )


def _openai(model_name: str, temperature: float, max_tokens: int) -> BaseChatModel:
    from langchain_openai import ChatOpenAI

    settings = get_settings()
    return ChatOpenAI(
        model=model_name,
        temperature=temperature,
        max_completion_tokens=max_tokens,
        api_key=settings.openai_api_key,
    )


_PROVIDER_FACTORIES: dict[str, Callable[[str, float, int], BaseChatModel]] = {
    "anthropic": _anthropic,
    "openai": _openai,
}


class UnknownProviderError(ValueError):
    pass


def _build_one(provider: str, model_name: str, temperature: float, max_tokens: int) -> BaseChatModel:
    factory = _PROVIDER_FACTORIES.get(provider)
    if factory is None:
        raise UnknownProviderError(
            f"unknown model provider '{provider}'; known providers: {sorted(_PROVIDER_FACTORIES)}"
        )
    return factory(model_name, temperature, max_tokens)


def get_chat_model(config: ModelConfig) -> BaseChatModel:
    """Resolve an AgentSpec model block into a LangChain chat model.

    If `config.fallback` is set, the returned model tries the primary first
    and falls back to each entry in order on error (LangChain's built-in
    `with_fallbacks`), so a provider outage doesn't take an agent down.
    """
    primary = _build_one(config.provider, config.model_name, config.temperature, config.max_tokens)
    if not config.fallback:
        return primary

    fallbacks = [
        _build_one(f.provider, f.model_name, config.temperature, config.max_tokens)
        for f in config.fallback
    ]
    return primary.with_fallbacks(fallbacks)
