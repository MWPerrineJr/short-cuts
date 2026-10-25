"""Evaluation layer: measure LLM systems the way you'd measure any model.

from aiframe.eval import load_golden, evaluate_rag, compare_reports
"""

from aiframe.eval.evaluate import (
    EvalReport,
    GoldenExample,
    compare_reports,
    evaluate_rag,
    is_refusal,
    load_golden,
    save_report,
    wilson_interval,
)

__all__ = [
    "EvalReport",
    "GoldenExample",
    "compare_reports",
    "evaluate_rag",
    "is_refusal",
    "load_golden",
    "save_report",
    "wilson_interval",
]
