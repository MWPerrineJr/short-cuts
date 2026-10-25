"""
Typed, validated configuration for the AI framework.

WHY THIS PATTERN
----------------
Hard-coding model names and reading ``os.environ`` all over a codebase makes
experiments hard to reproduce and bugs hard to trace. Instead, every setting
lives in ONE typed object loaded from ``.env``:

- Pydantic validates types at startup (temperature must be a float in [0, 2]).
- Secrets are wrapped in ``SecretStr`` so they never leak into logs or tracebacks.
- Missing config fails FAST with a clear message, not deep inside LangChain.

Usage
-----
    from config.settings import get_settings

    settings = get_settings()
    provider, model = settings.require_llm()   # raises ConfigError if incomplete
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = PROJECT_ROOT / ".env"

# Provider names match LangChain's `init_chat_model(model_provider=...)` values.
# Maps provider -> the Settings field that holds its API key (None = no key needed).
PROVIDER_KEYS: dict[str, str | None] = {
    "openai": "openai_api_key",
    "anthropic": "anthropic_api_key",
    "google_genai": "google_api_key",
    "ollama": None,  # local models: needs a running Ollama server, not a key
}

# Anthropic offers chat models but no embeddings API.
EMBEDDING_PROVIDERS = ("openai", "google_genai", "ollama")


class ConfigError(RuntimeError):
    """Raised when required configuration is missing or invalid."""


class Settings(BaseSettings):
    """All runtime configuration, loaded from environment variables and `.env`."""

    model_config = SettingsConfigDict(
        env_file=ENV_FILE,
        env_file_encoding="utf-8",
        env_ignore_empty=True,  # `LLM_PROVIDER=` (blank) falls back to the default
        extra="ignore",  # unrelated variables in .env are allowed
        validate_default=True,  # run validators on defaults too (path resolution)
    )

    # --- Chat model (no defaults on purpose: you choose per project) ---
    llm_provider: str | None = None
    llm_model: str | None = None
    llm_temperature: float = Field(default=0.2, ge=0.0, le=2.0)

    # --- Embeddings (used by RAG) ---
    embedding_provider: str | None = None
    embedding_model: str | None = None

    # --- Provider credentials ---
    openai_api_key: SecretStr | None = None
    anthropic_api_key: SecretStr | None = None
    google_api_key: SecretStr | None = None
    ollama_base_url: str = "http://localhost:11434"

    # --- Observability ---
    langsmith_tracing: bool = False
    langsmith_api_key: SecretStr | None = None
    langsmith_project: str = "ai-framework"

    # --- Paths & runtime ---
    data_dir: Path = Path("data")
    vectorstore_dir: Path = Path("vectorstore")
    max_concurrency: int = Field(default=5, ge=1, le=100)

    # ------------------------------------------------------------------ #
    # Validators
    # ------------------------------------------------------------------ #
    @field_validator("llm_provider", "embedding_provider")
    @classmethod
    def _normalize_provider(cls, value: str | None) -> str | None:
        return value.strip().lower() if value else value

    @field_validator("data_dir", "vectorstore_dir")
    @classmethod
    def _resolve_path(cls, value: Path) -> Path:
        # Anchor relative paths to the project root, not the current working
        # directory, so scripts behave the same wherever they are launched from.
        return value if value.is_absolute() else PROJECT_ROOT / value

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def api_key_for(self, provider: str) -> SecretStr | None:
        """Return the API key for a provider, or None if it needs/has none."""
        field = PROVIDER_KEYS.get(provider)
        return getattr(self, field) if field else None

    def configured_providers(self) -> list[str]:
        """Providers with credentials available (Ollama needs none, so always listed)."""
        return [
            p
            for p, field in PROVIDER_KEYS.items()
            if field is None or getattr(self, field)
        ]

    def _require(
        self,
        kind: str,
        provider: str | None,
        model: str | None,
        provider_var: str,
        model_var: str,
        allowed: tuple[str, ...],
    ) -> tuple[str, str]:
        problems: list[str] = []
        if not provider:
            problems.append(
                f"Set {provider_var} in .env (one of: {', '.join(allowed)})."
            )
        elif provider not in allowed:
            problems.append(
                f"{provider_var}='{provider}' is not supported here. Use one of: {', '.join(allowed)}."
            )
        if not model:
            problems.append(
                f"Set {model_var} in .env (a model name from your provider's docs)."
            )
        if provider in allowed:
            key_field = PROVIDER_KEYS.get(provider or "")
            if key_field and not getattr(self, key_field):
                problems.append(f"Set {key_field.upper()} in .env to use '{provider}'.")
        if problems:
            raise ConfigError(
                f"{kind} configuration incomplete:\n  - " + "\n  - ".join(problems)
            )
        assert provider and model  # narrowed for type checkers
        return provider, model

    def require_llm(self) -> tuple[str, str]:
        """Return (provider, model) for the chat model or raise ConfigError."""
        return self._require(
            "Chat model",
            self.llm_provider,
            self.llm_model,
            "LLM_PROVIDER",
            "LLM_MODEL",
            tuple(PROVIDER_KEYS),
        )

    def require_embeddings(self) -> tuple[str, str]:
        """Return (provider, model) for embeddings or raise ConfigError."""
        return self._require(
            "Embeddings",
            self.embedding_provider,
            self.embedding_model,
            "EMBEDDING_PROVIDER",
            "EMBEDDING_MODEL",
            EMBEDDING_PROVIDERS,
        )

    def apply_tracing(self) -> bool:
        """Export LangSmith settings to os.environ (LangChain reads them from there).

        Returns True if tracing was enabled.
        """
        if not self.langsmith_tracing:
            return False
        if not self.langsmith_api_key:
            raise ConfigError(
                "LANGSMITH_TRACING=true but LANGSMITH_API_KEY is not set."
            )
        os.environ["LANGSMITH_TRACING"] = "true"
        os.environ["LANGSMITH_API_KEY"] = self.langsmith_api_key.get_secret_value()
        os.environ["LANGSMITH_PROJECT"] = self.langsmith_project
        return True


@lru_cache
def get_settings() -> Settings:
    """Load settings once and reuse them.

    In tests, call ``get_settings.cache_clear()`` after changing env vars.
    """
    return Settings()
