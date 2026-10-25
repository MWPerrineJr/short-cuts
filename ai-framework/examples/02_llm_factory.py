"""
Example 02 - The model factory
==============================

CONCEPT: Code against an interface, not a provider.
Every chat model from ``get_llm()`` exposes the same methods, so the four
patterns below work unchanged with OpenAI, Anthropic, Google, or Ollama:

    1. invoke      one request -> one response
    2. stream      tokens as they are generated (better UX for chat apps)
    3. chain       prompt template | model | parser  (LangChain Expression Language)
    4. structured  response parsed into a validated Pydantic object
    5. compare     the same prompt across several providers

Run from the ai-framework folder (after 01_check_setup.py passes):
    python examples/02_llm_factory.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make `config` and `aiframe` importable when this file is run directly.
# The project imports below come after this on purpose.
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import time

from langchain_core.output_parsers import StrOutputParser
from rich.console import Console

from aiframe.core import get_llm, get_prompt
from aiframe.core.schemas import Answer
from config.settings import ConfigError, get_settings

console = Console()

# TRY: fill in model names for providers you have keys for, then re-run.
# Providers left blank are skipped in the comparison.
COMPARE_MODELS: dict[str, str] = {
    "openai": "",
    "anthropic": "",
    "google_genai": "",
    "ollama": "",
}


def demo_invoke() -> None:
    console.rule("1. invoke")
    llm = get_llm()
    response = llm.invoke("In one sentence, what is retrieval-augmented generation?")
    console.print(response.content)
    # Token usage lives in metadata: the basis for cost tracking.
    console.print(f"[dim]usage: {getattr(response, 'usage_metadata', None)}[/]")


def demo_stream() -> None:
    console.rule("2. stream")
    llm = get_llm()
    for chunk in llm.stream("List three uses of vector embeddings, one line each."):
        console.print(chunk.content, end="")
    console.print()


def demo_chain() -> None:
    console.rule("3. chain (prompt | model | parser)")
    # The registry returns a versioned template; `|` composes runnables.
    chain = get_prompt("assistant") | get_llm() | StrOutputParser()
    # TRY: get_llm(temperature=1.0) and run twice to see output variance.
    console.print(
        chain.invoke(
            {"input": "Explain overfitting to a new analyst in two sentences."}
        )
    )


def demo_structured() -> None:
    console.rule("4. structured output")
    structured_llm = get_llm(temperature=0).with_structured_output(Answer)
    result: Answer = structured_llm.invoke(
        "What is the capital of Australia? Report your confidence and sources."
    )
    console.print(result.model_dump())
    # A real Python object, not a string: no regex parsing needed.
    console.print(
        f"[dim]type={type(result).__name__}, confidence={result.confidence:.2f}[/]"
    )


def demo_compare() -> None:
    console.rule("5. compare providers")
    settings = get_settings()
    question = "In 15 words or fewer: why do we split data into train and test sets?"
    targets = {
        p: m
        for p, m in COMPARE_MODELS.items()
        if m and p in settings.configured_providers()
    }
    if not targets:
        console.print(
            "[yellow]Fill in COMPARE_MODELS at the top of this file to compare.[/]"
        )
        return
    for provider, model in targets.items():
        start = time.perf_counter()
        try:
            reply = get_llm(provider=provider, model=model).invoke(question).content
        except Exception as err:  # noqa: BLE001 - one provider failing shouldn't stop the rest
            reply = f"[red]error: {err}[/]"
        elapsed = time.perf_counter() - start
        console.print(f"[bold]{provider}/{model}[/] ({elapsed:.1f}s): {reply}")


def main() -> int:
    try:
        get_settings().require_llm()
    except ConfigError as err:
        console.print(
            f"[red]{err}[/]\nRun examples/01_check_setup.py for a full report."
        )
        return 1

    for demo in (demo_invoke, demo_stream, demo_chain, demo_structured, demo_compare):
        demo()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
