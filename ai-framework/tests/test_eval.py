"""Tests for the evaluation harness (Step 5)."""

from __future__ import annotations

import json

import pytest

from aiframe.core.schemas import JudgeScore
from aiframe.eval import (
    GoldenExample,
    compare_reports,
    evaluate_rag,
    is_refusal,
    load_golden,
    save_report,
    wilson_interval,
)
from aiframe.workflows.rag import RAGPipeline
from tests.conftest import GOLDEN_SET, SAMPLE_DIR, FakeStructured, make_fake_llm

IDK = "I don't know based on the provided documents."


def test_golden_set_file_is_valid():
    examples = load_golden(GOLDEN_SET)
    assert len(examples) >= 10
    assert len({e.id for e in examples}) == len(examples)  # unique ids
    assert any(not e.answerable for e in examples)  # includes refusal checks
    sample_names = {p.name for p in SAMPLE_DIR.glob("*.md")}
    for ex in examples:
        assert set(ex.expected_sources) <= sample_names, ex.id
        assert ex.answerable == bool(ex.reference), ex.id


@pytest.mark.parametrize(
    ("answer", "expected"),
    [(IDK, True), ("I do not know.", True), ("15 days [handbook]", False)],
)
def test_is_refusal(answer, expected):
    assert is_refusal(answer) is expected


def test_wilson_interval():
    low, high = wilson_interval(8, 10)
    assert 0.45 < low < 0.5 and 0.94 < high < 0.97  # wide at n=10
    assert wilson_interval(0, 0) == (0.0, 0.0)
    narrow = wilson_interval(800, 1000)
    assert narrow[1] - narrow[0] < high - low  # more data, tighter interval


def _pipeline(reply: str, fake_embeddings) -> RAGPipeline:
    pipeline = RAGPipeline(llm=make_fake_llm(reply), embeddings=fake_embeddings, k=3)
    pipeline.ingest(SAMPLE_DIR)
    return pipeline


EXAMPLES = [
    GoldenExample(
        id="pto",
        question="How many PTO days do new employees receive?",
        reference="15 days",
        expected_sources=["company_handbook.md"],
    ),
    GoldenExample(id="leave", question="Parental leave policy?", answerable=False),
]


def test_evaluate_scores_answers_and_refusals(fake_embeddings):
    judge = FakeStructured(JudgeScore(score=5, reasoning="matches reference"))
    report = evaluate_rag(
        _pipeline("15 days [company_handbook.md]", fake_embeddings),
        EXAMPLES,
        judge_llm=judge,
        progress=False,
    )
    rows = report.rows.set_index("id")
    assert rows.loc["pto", "correct"] and rows.loc["pto", "judge_score"] == 5
    assert not rows.loc["leave", "correct"]  # answered when it should have refused
    assert judge.calls == 1  # judge only runs on answerable, non-refused questions
    assert report.summary["accuracy"] == 0.5
    assert report.summary["refusal_accuracy"] == 0.0
    assert 0 < report.summary["hit_rate"] <= 1


def test_evaluate_flags_false_refusals(fake_embeddings):
    judge = FakeStructured(JudgeScore(score=5, reasoning=""))
    report = evaluate_rag(
        _pipeline(IDK, fake_embeddings), EXAMPLES, judge_llm=judge, progress=False
    )
    assert report.summary["false_refusals"] == 1
    assert report.summary["refusal_accuracy"] == 1.0
    assert judge.calls == 0


def test_compare_and_save(tmp_path, fake_embeddings):
    judge = FakeStructured(JudgeScore(score=4, reasoning=""))
    reports = [
        evaluate_rag(
            _pipeline(reply, fake_embeddings),
            EXAMPLES,
            judge_llm=judge,
            name=name,
            progress=False,
        )
        for name, reply in [("answers", "15 days"), ("refuses", IDK)]
    ]
    table = compare_reports(reports)
    assert list(table.index) == ["answers", "refuses"]

    rows_path, summary_path = save_report(reports[0], tmp_path / "reports")
    assert rows_path.exists()
    assert json.loads(summary_path.read_text())["questions"] == 2
