"""
Example 01 - Check your setup
=============================

CONCEPT: Fail fast.
Before spending money on API calls, verify the environment: Python version,
packages installed, `.env` present, a provider configured with its key.
Production services do the same thing at startup (a "readiness check").

Run from the ai-framework folder:
    python examples/01_check_setup.py

Exit code is 0 when the chat model is ready and 1 otherwise, so this script
can also gate a CI job or a Docker HEALTHCHECK.
"""

from __future__ import annotations

import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

# Make `config` and `aiframe` importable when this file is run directly.
# The project imports below come after this on purpose.
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from pydantic import ValidationError
from rich.console import Console
from rich.table import Table

from config.settings import (
    ENV_FILE,
    PROVIDER_KEYS,
    ConfigError,
    Settings,
    get_settings,
)

console = Console()

# TRY: add a package your project depends on and re-run.
KEY_PACKAGES = [
    "langchain",
    "langchain-core",
    "langgraph",
    "langsmith",
    "pydantic",
    "pydantic-settings",
    "gradio",
    "streamlit",
    "fastapi",
    "chromadb",
    "ipython",
]

OK, FAIL, WARN = "[green]OK[/]", "[red]MISSING[/]", "[yellow]WARN[/]"


def check_environment() -> bool:
    """Python version and .env presence."""
    py_ok = sys.version_info >= (3, 11)
    py = ".".join(map(str, sys.version_info[:3]))
    console.print(f"Python {py:<10} {OK if py_ok else FAIL + ' (need 3.11+)'}")
    if ENV_FILE.exists():
        console.print(f".env file        {OK}")
    else:
        console.print(f".env file        {WARN} not found - run: cp .env.example .env")
    return py_ok


def check_packages() -> bool:
    """Report installed versions of the key packages."""
    table = Table(title="Packages", show_lines=False)
    table.add_column("Package")
    table.add_column("Version")
    all_ok = True
    for name in KEY_PACKAGES:
        try:
            table.add_row(name, version(name))
        except PackageNotFoundError:
            table.add_row(name, FAIL)
            all_ok = False
    console.print(table)
    if not all_ok:
        console.print("Install with: [bold]pip install -r ../requirements.txt[/]")
    return all_ok


def check_providers(settings: Settings) -> None:
    """Show which providers have credentials. Keys are NEVER printed, not even partially."""
    table = Table(title="Provider credentials")
    table.add_column("Provider")
    table.add_column("Credential")
    for provider, key_field in PROVIDER_KEYS.items():
        if key_field is None:
            status = f"none needed (server: {settings.ollama_base_url})"
        else:
            status = (
                OK
                if getattr(settings, key_field)
                else f"[dim]{key_field.upper()} not set[/]"
            )
        table.add_row(provider, status)
    console.print(table)


def check_models(settings: Settings) -> bool:
    """Validate chat + embedding config using the same checks the framework uses."""
    llm_ok = True
    try:
        provider, model = settings.require_llm()
        console.print(f"Chat model       {OK} {provider} / {model}")
    except ConfigError as err:
        llm_ok = False
        console.print(f"Chat model       {FAIL}\n[red]{err}[/]")

    # Embeddings are only needed for RAG, so a gap here is a warning, not a failure.
    try:
        provider, model = settings.require_embeddings()
        console.print(f"Embeddings       {OK} {provider} / {model}")
    except ConfigError as err:
        console.print(
            f"Embeddings       {WARN} (only needed for RAG)\n[yellow]{err}[/]"
        )

    tracing = "on" if settings.langsmith_tracing else "off"
    console.print(f"LangSmith        tracing {tracing}")
    return llm_ok


def main() -> int:
    console.rule("[bold]AI Framework - setup check")
    env_ok = check_environment()
    pkgs_ok = check_packages()

    try:
        settings = get_settings()
    except ValidationError as err:  # a value in .env has the wrong type or range
        console.print(f"[red]Invalid value in .env:[/]\n{err}")
        return 1

    check_providers(settings)
    llm_ok = check_models(settings)

    ready = env_ok and pkgs_ok and llm_ok
    console.rule("[green]Ready to build[/]" if ready else "[red]Fix the items above[/]")
    # TRY: set LLM_TEMPERATURE=5 in .env and re-run to watch validation catch it.
    return 0 if ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
