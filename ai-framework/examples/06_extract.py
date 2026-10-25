"""
Example 06 - Structured extraction into a DataFrame
===================================================

CONCEPT: The schema is the contract.
Define WHAT you want as a Pydantic model; the LLM fills it in; Pydantic
validates it; pandas takes over. Part 2 shows how to define your own
schema in a few lines.

Run from the ai-framework folder:
    python examples/06_extract.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make `config` and `aiframe` importable when this file is run directly.
# The project imports below come after this on purpose.
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from typing import Literal

from pydantic import BaseModel, Field
from rich.console import Console

from aiframe.core.schemas import ContactList
from aiframe.workflows.extract import extract, extract_many, to_dataframe
from config.settings import ConfigError, get_settings

console = Console()

EMAILS = [
    (
        "Hi team, looping in Dana Reyes (dana.reyes@acme.io), our new Head of Data at Acme. "
        "She'll work with Priya Shah, Acme's ML lead, on the pilot."
    ),
    (
        "Thanks for the demo! I'm Tom Becker, procurement manager at Globex. "
        "Reach me at tbecker@globex.com."
    ),
    "Quick note: the contract is on hold until legal reviews it. No names to add.",
]

TICKETS = [
    "Dashboard won't load since this morning, getting NX-404. Whole sales team is blocked!",
    "Could you add a dark mode option sometime? Not urgent.",
    "Scheduled report failed overnight with NX-500. Need it fixed before Friday's board meeting.",
]


# Part 2: your own schema. Field descriptions are instructions to the model.
class SupportTicket(BaseModel):
    category: Literal["bug", "feature_request", "question"]
    priority: Literal["low", "medium", "high"] = Field(
        description="high if users are blocked or there is a deadline"
    )
    error_code: str | None = Field(
        default=None, description="e.g. NX-404, if mentioned"
    )
    summary: str = Field(description="Ten words or fewer.")


def main() -> int:
    try:
        get_settings().require_llm()
    except ConfigError as err:
        console.print(
            f"[red]{err}[/]\nRun examples/01_check_setup.py for a full report."
        )
        return 1

    console.rule("1. One document -> one object")
    result = extract(EMAILS[0], ContactList)
    console.print(result.model_dump())

    console.rule("2. Many documents -> DataFrame (one row per contact)")
    results = extract_many(EMAILS, ContactList)
    df = to_dataframe(results, explode="contacts")
    console.print(df.to_string(index=False))
    # Email 3 has no contacts: it still gets a row, so nothing silently disappears.

    console.rule("3. Your own schema -> analysis")
    tickets = to_dataframe(extract_many(TICKETS, SupportTicket))
    console.print(tickets.to_string(index=False))
    if "priority" in tickets:
        console.print(
            "\nPriority counts:\n", tickets["priority"].value_counts().to_string()
        )
    # TRY: add a `sentiment` field to SupportTicket and re-run. No other code changes.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
