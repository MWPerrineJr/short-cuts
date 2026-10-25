"""
Example 08 - Evaluate RAG against a golden set
==============================================

CONCEPT: Measure, then change one thing, then measure again.
Runs the golden set (data/eval/golden_set.jsonl) through two RAG
configurations that differ only in ``k`` (chunks retrieved), then compares
retrieval and answer metrics side by side. Reports are saved to reports/.

Read the confidence interval before declaring a winner: with 11 questions,
a 10-point accuracy difference is usually noise.

Run from the ai-framework folder (needs chat + embeddings configured):
    python examples/08_evaluate.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make `config` and `aiframe` importable when this file is run directly.
# The project imports below come after this on purpose.
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from rich.console import Console

from aiframe.core import get_embeddings
from aiframe.eval import compare_reports, evaluate_rag, load_golden, save_report
from aiframe.workflows.rag import RAGPipeline
from config.settings import ConfigError, get_settings

console = Console()
GOLDEN = ROOT / "data" / "eval" / "golden_set.jsonl"
REPORTS = ROOT / "reports"

# TRY: change one variable at a time, e.g. chunk_size 300 vs 800, or k 2 vs 6.
CONFIGS = {
    "k2": {"k": 2, "chunk_size": 800},
    "k4": {"k": 4, "chunk_size": 800},
}


def main() -> int:
    settings = get_settings()
    try:
        settings.require_llm()
        settings.require_embeddings()
    except ConfigError as err:
        console.print(
            f"[red]{err}[/]\nRun examples/01_check_setup.py for a full report."
        )
        return 1

    examples = load_golden(GOLDEN)
    console.print(f"Golden set: {len(examples)} questions ({GOLDEN.name})")
    embeddings = get_embeddings()  # shared, so only `k` differs between runs

    reports = []
    for name, cfg in CONFIGS.items():
        rag = RAGPipeline(k=cfg["k"], embeddings=embeddings)
        rag.ingest(ROOT / "data" / "sample", chunk_size=cfg["chunk_size"])
        # TRY: pass judge_llm=get_llm(provider=..., model=...) to grade with a different model.
        report = evaluate_rag(rag, examples, name=name)
        save_report(report, REPORTS)
        reports.append(report)

    console.rule("Comparison")
    table = compare_reports(reports)
    shown = [
        "accuracy",
        "accuracy_ci95",
        "refusal_accuracy",
        "false_refusals",
        "hit_rate",
        "mrr",
    ]
    console.print(table[shown].to_string())

    console.rule("Misses (last config)")
    misses = reports[-1].rows.loc[~reports[-1].rows["correct"]]
    if misses.empty:
        console.print("[green]No misses.[/]")
    for _, row in misses.iterrows():
        console.print(f"[bold]{row['id']}[/]: {row['answer'][:120]}")
        console.print(f"  [dim]judge: {row['judge_reasoning']}[/]")

    console.print(f"\nReports saved to {REPORTS.relative_to(ROOT)}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
