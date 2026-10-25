"""
Example 03 - Chat with memory and streaming
===========================================

CONCEPT: The model is stateless; the session is not.
Every turn re-sends recent history so the model can "remember". Watch the
demo: the second question only works because turn 1 is replayed.

Run from the ai-framework folder:
    python examples/03_chat.py           # scripted demo of memory
    python examples/03_chat.py --live    # interactive chat (/reset, /history, /quit)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make `config` and `aiframe` importable when this file is run directly.
# The project imports below come after this on purpose.
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from rich.console import Console

from aiframe.workflows.chat import ChatSession
from config.settings import ConfigError, get_settings

console = Console()


def demo() -> None:
    # TRY: set max_turns=0 and re-run: the model "forgets" your name.
    session = ChatSession(max_turns=10)
    for turn in [
        "Hi! My name is Michael and I'm a data scientist learning MLOps.",
        "What's my name, and what am I learning?",
        "Suggest one project that would build that skill. One sentence.",
    ]:
        console.print(f"\n[bold cyan]you>[/] {turn}")
        console.print("[bold green]bot>[/] ", end="")
        for chunk in session.stream(turn):
            console.print(chunk, end="")
        console.print()
    console.print(f"\n[dim]{session.turns} turns stored in history[/]")


def live() -> None:
    session = ChatSession()
    console.print("[dim]Commands: /reset  /history  /quit[/]")
    while True:
        try:
            text = console.input("\n[bold cyan]you>[/] ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if text in {"/quit", "/exit"}:
            break
        if text == "/reset":
            session.reset()
            console.print("[dim]history cleared[/]")
            continue
        if text == "/history":
            for msg in session.history:
                console.print(f"[dim]{msg.type}:[/] {msg.content}")
            continue
        if not text:
            continue
        console.print("[bold green]bot>[/] ", end="")
        for chunk in session.stream(text):
            console.print(chunk, end="")
        console.print()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--live", action="store_true", help="interactive chat")
    args = parser.parse_args()
    try:
        get_settings().require_llm()
    except ConfigError as err:
        console.print(
            f"[red]{err}[/]\nRun examples/01_check_setup.py for a full report."
        )
        return 1
    live() if args.live else demo()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
