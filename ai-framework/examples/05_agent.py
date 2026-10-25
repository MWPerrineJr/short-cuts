"""
Example 05 - Tool-calling agent
===============================

CONCEPT: The model plans; your code executes.
The agent decides which tools to call and in what order. Your code only
defines the tools and runs them. Read the trace after each answer: it shows
exactly what the agent did, which is how you debug agents.

The questions are chosen so each needs a different tool path:
    1. search_notes only
    2. search_notes then calculator (multi-step)
    3. current_datetime
    4. a follow-up that relies on memory from question 2 (same thread_id)

Run from the ai-framework folder:
    python examples/05_agent.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make `config` and `aiframe` importable when this file is run directly.
# The project imports below come after this on purpose.
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from rich.console import Console

from aiframe.workflows.agent import build_agent, run_agent
from config.settings import ConfigError, get_settings

console = Console()

QUESTIONS = [
    "What does error code NX-500 mean in Nimbus Analytics?",
    "A team of 12 is on the Pro plan. What is their annual cost?",
    "What day of the week is it today?",
    "And how much would they save per year on the Starter plan instead?",
]


def main() -> int:
    try:
        get_settings().require_llm()
    except ConfigError as err:
        console.print(
            f"[red]{err}[/]\nRun examples/01_check_setup.py for a full report."
        )
        return 1

    # TRY: build_agent(memory=False) and watch question 4 fail without context.
    agent = build_agent()
    for question in QUESTIONS:
        result = run_agent(agent, question, thread_id="demo")
        console.print(f"\n[bold cyan]Q:[/] {question}")
        for call in result.tool_calls:
            preview = call.result.replace("\n", " ")[:90]
            console.print(f"  [dim]tool {call.name}({call.args}) -> {preview}[/]")
        console.print(f"[bold green]A:[/] {result.answer}")
        console.print(
            f"[dim]{len(result.tool_calls)} tool calls, {result.steps} messages[/]"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
