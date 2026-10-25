"""
Example 04 - RAG with hybrid retrieval
======================================

CONCEPT: Ground answers in your documents, and inspect retrieval first.
Most bad RAG answers are retrieval failures, not generation failures. So
this example shows what each retriever returns BEFORE generating answers:

- An exact code ("NX-404") is easy for keyword search, hard for vectors.
- A paraphrase ("vacation for new hires" vs. "paid time off") is the opposite.
- Hybrid (RRF) fuses both rankings and usually wins on both.

Then it asks questions, including one the documents can't answer, to check
the model says "I don't know" instead of inventing something.

Run from the ai-framework folder (needs EMBEDDING_PROVIDER/MODEL in .env):
    python examples/04_rag.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make `config` and `aiframe` importable when this file is run directly.
# The project imports below come after this on purpose.
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from rich.console import Console
from rich.table import Table

from aiframe.workflows.rag import RAGPipeline
from config.settings import ConfigError, get_settings

console = Console()
SAMPLE_DIR = ROOT / "data" / "sample"

# TRY: add your own .md/.txt/.pdf files to data/sample and new queries here.
RETRIEVAL_QUERIES = [
    "What does error NX-404 mean?",
    "How much vacation do new hires get?",
]
QUESTIONS = [
    "How many PTO days do new employees get, and how many roll over?",
    "Which Nimbus plan includes forecasting, and what does it cost?",
    "What is data leakage? Give the example from the glossary.",
    "What is the company's parental leave policy?",  # not in the docs
]


def compare_retrievers(rag: RAGPipeline) -> None:
    console.rule("Retrieval comparison (top 3 sources)")
    table = Table(show_lines=True)
    for col in ("query", "vector", "keyword (BM25)", "hybrid (RRF)"):
        table.add_column(col)

    def fmt(docs) -> str:
        return "\n".join(
            f"{d.metadata['source']}#{d.metadata['chunk_id']}" for d in docs[:3]
        )

    for q in RETRIEVAL_QUERIES:
        table.add_row(
            q,
            fmt(rag.vector_search(q)),
            fmt(rag.keyword_search(q)),
            fmt(rag.retrieve(q)),
        )
    console.print(table)


def ask_questions(rag: RAGPipeline) -> None:
    console.rule("Answers")
    for q in QUESTIONS:
        result = rag.ask(q)
        console.print(f"\n[bold cyan]Q:[/] {q}")
        console.print(f"[bold green]A:[/] {result.answer}")
        console.print(f"[dim]sources: {', '.join(result.sources)}[/]")


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

    # TRY: chunk_size=300 vs 1500, and k=2 vs 6. Watch answers and sources change.
    rag = RAGPipeline(k=4)
    n = rag.ingest(SAMPLE_DIR, chunk_size=800, chunk_overlap=100)
    console.print(f"Indexed [bold]{n}[/] chunks from {SAMPLE_DIR.name}/")

    compare_retrievers(rag)
    ask_questions(rag)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
