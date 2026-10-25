"""Tests for configuration (Step 1) and the core layer (Step 2)."""

from __future__ import annotations

import pytest
from langchain_core.output_parsers import StrOutputParser
from pydantic import ValidationError

from aiframe.core import get_embeddings, get_llm, get_prompt, list_prompts
from aiframe.core.schemas import Answer, Contact, JudgeScore, Sentiment
from config.settings import PROJECT_ROOT, ConfigError, Settings

# ---------------------------------------------------------------------- #
# Settings
# ---------------------------------------------------------------------- #


def test_missing_provider_and_model_reported_together():
    with pytest.raises(ConfigError) as err:
        Settings().require_llm()
    assert "LLM_PROVIDER" in str(err.value)
    assert "LLM_MODEL" in str(err.value)


def test_missing_key_names_the_variable(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("LLM_MODEL", "m")
    with pytest.raises(ConfigError, match="OPENAI_API_KEY"):
        Settings().require_llm()


def test_ollama_needs_no_key(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "Ollama")  # case-insensitive
    monkeypatch.setenv("LLM_MODEL", "llama3.2")
    assert Settings().require_llm() == ("ollama", "llama3.2")


def test_anthropic_rejected_for_embeddings(monkeypatch):
    monkeypatch.setenv("EMBEDDING_PROVIDER", "anthropic")
    monkeypatch.setenv("EMBEDDING_MODEL", "x")
    with pytest.raises(ConfigError, match="not supported"):
        Settings().require_embeddings()


@pytest.mark.parametrize("value", ["-0.1", "2.5"])
def test_temperature_out_of_range_rejected(monkeypatch, value):
    monkeypatch.setenv("LLM_TEMPERATURE", value)
    with pytest.raises(ValidationError):
        Settings()


def test_blank_values_fall_back_to_defaults(monkeypatch):
    monkeypatch.setenv("LLM_TEMPERATURE", "")
    assert Settings().llm_temperature == 0.2


def test_secrets_are_masked(configured):
    assert "sk-test" not in repr(configured)
    assert str(configured.openai_api_key) == "**********"


def test_relative_paths_anchor_to_project_root(monkeypatch):
    monkeypatch.setenv("DATA_DIR", "my_data")
    assert Settings().data_dir == PROJECT_ROOT / "my_data"


def test_tracing_requires_key(monkeypatch):
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    with pytest.raises(ConfigError, match="LANGSMITH_API_KEY"):
        Settings().apply_tracing()


# ---------------------------------------------------------------------- #
# Model factory
# ---------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("provider", "model", "cls_name"),
    [
        ("openai", "gpt-test", "ChatOpenAI"),
        ("anthropic", "claude-test", "ChatAnthropic"),
        ("google_genai", "gemini-test", "ChatGoogleGenerativeAI"),
        ("ollama", "llama3.2", "ChatOllama"),
    ],
)
def test_get_llm_builds_each_provider(monkeypatch, provider, model, cls_name):
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    monkeypatch.setenv("GOOGLE_API_KEY", "k")
    llm = get_llm(provider=provider, model=model, settings=Settings())
    assert type(llm).__name__ == cls_name


def test_get_llm_uses_settings_and_overrides(configured):
    assert get_llm().temperature == configured.llm_temperature
    assert get_llm(temperature=0.9).temperature == 0.9
    assert get_llm(max_tokens=42).max_tokens == 42


def test_get_llm_unknown_provider(configured):
    with pytest.raises(ConfigError, match="Unknown provider"):
        get_llm(provider="bogus", model="x")


def test_get_embeddings(configured):
    assert type(get_embeddings()).__name__ == "OpenAIEmbeddings"
    with pytest.raises(ConfigError):
        get_embeddings(provider="anthropic", model="x")


# ---------------------------------------------------------------------- #
# Prompts
# ---------------------------------------------------------------------- #


def test_prompt_registry_contents():
    assert {p.key for p in list_prompts()} >= {
        "assistant/v1",
        "rag/v1",
        "extract/v1",
        "judge/v1",
    }


def test_rag_prompt_includes_context():
    messages = get_prompt("rag").format_messages(
        context="Paris [geo.md]", question="Where?"
    )
    assert "Paris [geo.md]" in messages[0].content
    assert messages[1].content == "Where?"


def test_assistant_history_is_optional():
    assert len(get_prompt("assistant").format_messages(input="hi")) == 2


def test_unknown_prompt_version_raises():
    with pytest.raises(KeyError):
        get_prompt("rag", "v99")


def test_prompt_model_parser_chain(fake_llm):
    chain = (
        get_prompt("assistant")
        | fake_llm("Overfitting is memorizing noise.")
        | StrOutputParser()
    )
    assert chain.invoke({"input": "x"}) == "Overfitting is memorizing noise."


# ---------------------------------------------------------------------- #
# Schemas
# ---------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "build",
    [
        lambda: Answer(answer="x", confidence=1.5),
        lambda: Contact(name="A", email="not-an-email"),
        lambda: Sentiment(label="angry", score=0, reason="r"),
        lambda: JudgeScore(score=6, reasoning="r"),
    ],
    ids=["confidence>1", "bad-email", "bad-label", "score>5"],
)
def test_schemas_reject_invalid(build):
    with pytest.raises(ValidationError):
        build()


def test_judge_pass_threshold():
    assert JudgeScore(score=4, reasoning="").passed
    assert not JudgeScore(score=3, reasoning="").passed
