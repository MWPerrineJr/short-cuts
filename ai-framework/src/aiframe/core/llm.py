"""
Provider-agnostic model factory.

WHY THIS PATTERN
----------------
Workflows never import ``ChatOpenAI`` or ``ChatAnthropic`` directly. They call
``get_llm()`` and receive a LangChain chat model with the same interface
(``invoke``, ``stream``, ``batch``, ``ainvoke``, ``with_structured_output``)
no matter which provider is behind it. That gives you:

- Provider switching through ``.env`` with no code changes.
- Easy A/B comparisons: ``get_llm(provider="ollama", model="llama3.2")``.
- One place to add cross-cutting behavior later (retries, caching, rate limits).

Keys are passed explicitly from settings, so nothing depends on variables
being exported to ``os.environ``.
"""

from __future__ import annotations

from typing import Any

from langchain.chat_models import init_chat_model
from langchain.embeddings import init_embeddings
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

from config.settings import (
    EMBEDDING_PROVIDERS,
    PROVIDER_KEYS,
    ConfigError,
    Settings,
    get_settings,
)


def _credentials(provider: str, settings: Settings) -> dict[str, Any]:
    """Provider-specific connection kwargs (each integration names them differently)."""
    if provider == "ollama":
        return {"base_url": settings.ollama_base_url}
    key = settings.api_key_for(provider)
    if key is None:
        field = PROVIDER_KEYS.get(provider)
        hint = f"Set {field.upper()} in .env." if field else "Check the provider name."
        raise ConfigError(f"No credentials for provider '{provider}'. {hint}")
    if provider == "google_genai":
        return {"google_api_key": key}
    return {"api_key": key}


def get_llm(
    *,
    provider: str | None = None,
    model: str | None = None,
    temperature: float | None = None,
    settings: Settings | None = None,
    **kwargs: Any,
) -> BaseChatModel:
    """Return a configured chat model.

    Defaults come from ``.env``; any argument overrides it for this call.
    Extra ``kwargs`` go to the provider class (e.g. ``max_tokens=512``).

    Examples:
        llm = get_llm()                                    # from .env
        creative = get_llm(temperature=0.9)                # override one setting
        local = get_llm(provider="ollama", model="llama3.2")
    """
    settings = settings or get_settings()
    if provider is None or model is None:
        default_provider, default_model = settings.require_llm()
        provider = provider or default_provider
        model = model or default_model
    provider = provider.strip().lower()
    if provider not in PROVIDER_KEYS:
        raise ConfigError(
            f"Unknown provider '{provider}'. Use one of: {', '.join(PROVIDER_KEYS)}."
        )

    return init_chat_model(
        model=model,
        model_provider=provider,
        temperature=settings.llm_temperature if temperature is None else temperature,
        **_credentials(provider, settings),
        **kwargs,
    )


def get_embeddings(
    *,
    provider: str | None = None,
    model: str | None = None,
    settings: Settings | None = None,
    **kwargs: Any,
) -> Embeddings:
    """Return a configured embeddings model (used by RAG and semantic search).

    Example:
        emb = get_embeddings()
        vectors = emb.embed_documents(["first chunk", "second chunk"])
    """
    settings = settings or get_settings()
    if provider is None or model is None:
        default_provider, default_model = settings.require_embeddings()
        provider = provider or default_provider
        model = model or default_model
    provider = provider.strip().lower()
    if provider not in EMBEDDING_PROVIDERS:
        raise ConfigError(
            f"'{provider}' has no embeddings support here. Use one of: {', '.join(EMBEDDING_PROVIDERS)}."
        )

    return init_embeddings(
        model, provider=provider, **_credentials(provider, settings), **kwargs
    )
