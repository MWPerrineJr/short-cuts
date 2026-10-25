"""
Shared test fixtures.

WHY THESE FIXTURES
------------------
Tests must be fast, free, and deterministic, so no test ever calls a real
model. Instead:

- ``isolated_settings`` (autouse) hides your real ``.env`` and API keys, so
  tests behave the same on every machine and in CI.
- ``configured`` sets fake provider settings for tests that need them.
- ``fake_llm`` / ``FakeStructured`` stand in for chat models; LangChain's
  ``DeterministicFakeEmbedding`` stands in for embeddings.

Run:  pytest              (all)
      pytest -m "not slow" (skip app/UI tests)
      pytest --cov=src     (with coverage)
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from config.settings import Settings, get_settings

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DIR = ROOT / "data" / "sample"
GOLDEN_SET = ROOT / "data" / "eval" / "golden_set.jsonl"

_ENV_PREFIXES = (
    "LLM_",
    "EMBEDDING_",
    "OPENAI_",
    "ANTHROPIC_",
    "GOOGLE_",
    "LANGSMITH_",
    "OLLAMA_",
)


# ---------------------------------------------------------------------- #
# Fakes
# ---------------------------------------------------------------------- #
class FakeChatModel(GenericFakeChatModel):
    """Fake chat model that also accepts tools (for agent tests)."""

    def bind_tools(self, tools, **kwargs):
        return self


class FakeStructured:
    """Stands in for a model whose ``with_structured_output`` returns fixed objects."""

    def __init__(self, output) -> None:
        self.output = output
        self.calls = 0

    def with_structured_output(self, schema):
        def respond(_):
            self.calls += 1
            return self.output(_) if callable(self.output) else self.output

        return RunnableLambda(respond)


def make_fake_llm(*replies: str | AIMessage, loop: bool = True) -> FakeChatModel:
    """Fake model returning ``replies`` in order (cycling forever when ``loop``)."""
    messages = [
        r if isinstance(r, AIMessage) else AIMessage(content=r) for r in replies
    ]

    def stream() -> Iterator[AIMessage]:
        while True:
            yield from messages
            if not loop:
                return

    return FakeChatModel(messages=stream())


# ---------------------------------------------------------------------- #
# Fixtures
# ---------------------------------------------------------------------- #
@pytest.fixture(autouse=True)
def isolated_settings(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Ignore the developer's .env and provider env vars in every test."""
    for key in list(os.environ):
        if key.startswith(_ENV_PREFIXES):
            monkeypatch.delenv(key)
    monkeypatch.setitem(Settings.model_config, "env_file", None)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def configured(monkeypatch: pytest.MonkeyPatch) -> Settings:
    """Valid (fake) chat + embeddings configuration."""
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("LLM_MODEL", "gpt-test")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-real")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "openai")
    monkeypatch.setenv("EMBEDDING_MODEL", "embed-test")
    get_settings.cache_clear()
    return get_settings()


@pytest.fixture
def fake_llm() -> Callable[..., FakeChatModel]:
    return make_fake_llm


@pytest.fixture
def fake_embeddings() -> DeterministicFakeEmbedding:
    return DeterministicFakeEmbedding(size=32)


@pytest.fixture
def rag(fake_embeddings):
    """An ingested RAG pipeline over the sample docs, with a fake LLM."""
    from aiframe.workflows.rag import RAGPipeline

    pipeline = RAGPipeline(
        llm=make_fake_llm("15 days of PTO [company_handbook.md]"),
        embeddings=fake_embeddings,
        k=3,
    )
    pipeline.ingest(SAMPLE_DIR)
    return pipeline
