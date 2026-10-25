"""
Sample agent tools.

WHY THIS PATTERN
----------------
A tool is a plain Python function the model may decide to call. LangChain's
``@tool`` decorator turns the function name, type hints, and DOCSTRING into a
schema sent to the model, so the docstring is effectively a prompt: say what
the tool does, when to use it, and what the arguments mean.

Two rules for safe tools:
1. Never ``eval()`` model-generated input. The calculator below parses the
   expression into a syntax tree and allows only arithmetic.
2. Return short strings, including errors. A tool that raises crashes the
   agent; a tool that returns "error: ..." lets the model recover.
"""

from __future__ import annotations

import ast
import operator
import re
from datetime import UTC, datetime
from pathlib import Path

from langchain_core.tools import tool

SAMPLE_DIR = Path(__file__).resolve().parents[3] / "data" / "sample"

_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _safe_eval(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPERATORS:
        if isinstance(node.op, ast.Pow) and abs(_safe_eval(node.right)) > 100:
            raise ValueError("exponent too large")
        return _OPERATORS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPERATORS:
        return _OPERATORS[type(node.op)](_safe_eval(node.operand))
    raise ValueError(f"unsupported expression: {ast.dump(node)[:40]}")


@tool
def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression exactly. Use for ANY math instead of mental arithmetic.

    Args:
        expression: Arithmetic using numbers, + - * / // % ** and parentheses,
            e.g. "(1250 * 0.07) / 12".
    """
    try:
        result = _safe_eval(ast.parse(expression, mode="eval"))
    except (SyntaxError, ValueError, ZeroDivisionError) as err:
        return f"error: {err}"
    return f"{result:.6g}" if isinstance(result, float) else str(result)


@tool
def current_datetime() -> str:
    """Return the current local date and time. Use when a question depends on today's date."""
    # Timezone-aware: take UTC, convert to the machine's local zone, show the zone.
    return datetime.now(UTC).astimezone().strftime("%A, %Y-%m-%d %H:%M %Z")


@tool
def search_notes(query: str) -> str:
    """Search the internal sample documents (company handbook, product FAQ, ML glossary).

    Use for questions about company policies, the Nimbus Analytics product,
    or definitions of machine-learning terms.

    Args:
        query: A few keywords, e.g. "PTO new employees".
    """
    terms = set(re.findall(r"[a-z0-9]+", query.lower()))
    if not terms:
        return "error: empty query"
    hits: list[tuple[int, str]] = []
    for file in sorted(SAMPLE_DIR.glob("*.md")):
        for paragraph in file.read_text(encoding="utf-8").split("\n\n"):
            words = set(re.findall(r"[a-z0-9]+", paragraph.lower()))
            score = len(terms & words)
            if score:
                hits.append((score, f"[{file.name}] {paragraph.strip()}"))
    if not hits:
        return "No matching notes found."
    hits.sort(key=lambda h: h[0], reverse=True)
    return "\n\n".join(text for _, text in hits[:3])


DEFAULT_TOOLS = [calculator, current_datetime, search_notes]
