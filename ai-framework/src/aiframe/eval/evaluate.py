"""
Golden-set evaluation for RAG.

WHY THIS PATTERN
----------------
"It looked good when I tried it" is not evidence. Treat the LLM system like
any predictive model: a held-out labeled set, metrics, and comparisons
between configurations (prompt v1 vs v2, k=2 vs k=4, model A vs model B).

A RAG system can fail in two separate places, so it gets two kinds of metric:

Retrieval (did we find the right document?)
    hit_rate   share of questions where an expected source was retrieved
    mrr        mean reciprocal rank of the first relevant chunk (1.0 = always first)

Generation (did we answer correctly?)
    accuracy          share judged correct (score >= 4 of 5 by an LLM judge)
    refusal_accuracy  unanswerable questions correctly answered "I don't know"
    false_refusals    answerable questions the system wrongly refused

Two caveats every evaluator should state:
- Small golden sets give wide confidence intervals. The report includes a
  95% Wilson interval for accuracy. With 10 questions it spans roughly
  +/-25 points, so don't over-read small differences.
- An LLM judge can be biased toward its own outputs. Use a different (ideally
  stronger) model as judge when you can, and spot-check its verdicts by hand.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, Field
from tqdm import tqdm

from aiframe.core import get_llm, get_prompt
from aiframe.core.schemas import JudgeScore

REFUSAL_MARKERS = ("i don't know", "i do not know", "i don’t know")


class GoldenExample(BaseModel):
    """One labeled evaluation question."""

    id: str
    question: str
    reference: str = Field(
        default="", description="The correct answer, in plain words."
    )
    expected_sources: list[str] = Field(default_factory=list)
    answerable: bool = True


@dataclass
class EvalReport:
    name: str
    rows: Any  # pandas DataFrame, one row per question
    summary: dict[str, Any]


def load_golden(path: str | Path) -> list[GoldenExample]:
    """Read a JSONL golden set (one JSON object per line; blank lines ignored)."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [GoldenExample.model_validate_json(line) for line in lines if line.strip()]


def is_refusal(answer: str) -> bool:
    text = answer.lower()
    return any(marker in text for marker in REFUSAL_MARKERS)


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a proportion; better than +/-2SE at small n."""
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (round(max(0.0, center - margin), 3), round(min(1.0, center + margin), 3))


def _judge_chain(judge_llm: BaseChatModel | None):
    return get_prompt("judge") | (
        judge_llm or get_llm(temperature=0)
    ).with_structured_output(JudgeScore)


def evaluate_rag(
    rag: Any,
    examples: Sequence[GoldenExample],
    judge_llm: BaseChatModel | None = None,
    name: str = "run",
    progress: bool = True,
) -> EvalReport:
    """Run every golden question through a RAGPipeline and score it.

    Args:
        rag: An ingested ``RAGPipeline`` (anything with ``retrieve`` and ``ask``).
        examples: Golden set from ``load_golden``.
        judge_llm: Model used as grader. Defaults to the configured model.
        name: Label for this configuration, used when comparing runs.
    """
    import pandas as pd

    judge = _judge_chain(judge_llm)
    rows: list[dict[str, Any]] = []
    for ex in tqdm(examples, desc=f"eval {name}", disable=not progress):
        ranked = [d.metadata.get("source") for d in rag.retrieve(ex.question)]
        start = time.perf_counter()
        result = rag.ask(ex.question)
        latency = time.perf_counter() - start
        refused = is_refusal(result.answer)

        row: dict[str, Any] = {
            "id": ex.id,
            "question": ex.question,
            "answerable": ex.answerable,
            "answer": result.answer,
            "retrieved_sources": ranked,
            "refused": refused,
            "latency_s": round(latency, 3),
        }

        if ex.expected_sources:
            ranks = [i for i, src in enumerate(ranked, 1) if src in ex.expected_sources]
            row["hit"] = bool(ranks)
            row["reciprocal_rank"] = 1.0 / ranks[0] if ranks else 0.0

        if not ex.answerable:
            row["correct"] = refused  # the right answer is "I don't know"
            row["judge_score"] = None
            row["judge_reasoning"] = "unanswerable: correct only if the system refuses"
        elif refused:
            row["correct"] = False
            row["judge_score"] = 1
            row["judge_reasoning"] = "refused an answerable question"
        else:
            verdict: JudgeScore = judge.invoke(
                {
                    "question": ex.question,
                    "reference": ex.reference,
                    "candidate": result.answer,
                }
            )
            row["correct"] = verdict.passed
            row["judge_score"] = verdict.score
            row["judge_reasoning"] = verdict.reasoning
        rows.append(row)

    df = pd.DataFrame(rows)
    return EvalReport(name=name, rows=df, summary=_summarize(df))


def _summarize(df: Any) -> dict[str, Any]:
    n = len(df)
    correct = int(df["correct"].sum()) if n else 0
    answerable = df[df["answerable"]] if n else df
    unanswerable = df[~df["answerable"]] if n else df
    with_sources = df.dropna(subset=["hit"]) if "hit" in df else df.iloc[0:0]
    latencies = sorted(df["latency_s"]) if n else []
    scores = df["judge_score"].dropna() if n else []

    def mean(series: Any) -> float | None:
        return round(float(series.mean()), 3) if len(series) else None

    return {
        "questions": n,
        "accuracy": round(correct / n, 3) if n else None,
        "accuracy_ci95": wilson_interval(correct, n),
        "answerable_accuracy": mean(answerable["correct"]),
        "refusal_accuracy": mean(unanswerable["correct"]),
        "false_refusals": int(answerable["refused"].sum()) if len(answerable) else 0,
        "hit_rate": mean(with_sources["hit"].astype(float))
        if len(with_sources)
        else None,
        "mrr": mean(with_sources["reciprocal_rank"]) if len(with_sources) else None,
        "mean_judge_score": mean(scores),
        "p50_latency_s": latencies[n // 2] if n else None,
        "p95_latency_s": latencies[min(n - 1, int(n * 0.95))] if n else None,
    }


def compare_reports(reports: Sequence[EvalReport]) -> Any:
    """One row per configuration: the table you'd put in a write-up."""
    import pandas as pd

    return pd.DataFrame([{"config": r.name, **r.summary} for r in reports]).set_index(
        "config"
    )


def save_report(report: EvalReport, out_dir: str | Path) -> tuple[Path, Path]:
    """Write per-question rows (CSV) and the summary (JSON). Returns both paths."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows_path = out / f"{report.name}_rows.csv"
    summary_path = out / f"{report.name}_summary.json"
    report.rows.to_csv(rows_path, index=False)
    summary_path.write_text(json.dumps(report.summary, indent=2), encoding="utf-8")
    return rows_path, summary_path
