"""
Extraction workflow: unstructured text -> validated Pydantic objects -> DataFrame.

WHY THIS PATTERN
----------------
A lot of business data is locked in emails, PDFs, notes and tickets.
Structured extraction turns that text into rows you can analyze:

    text --(prompt + with_structured_output(Schema))--> Schema --> DataFrame

The schema is the contract. Its field types and descriptions tell the
model what to return, and Pydantic rejects anything that doesn't fit (an
invalid email, a missing required field) instead of passing bad data along.

For a data scientist this is feature engineering from text: once the
results are in a DataFrame, everything you already know applies.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, TypeVar

from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel

from aiframe.core import get_llm, get_prompt
from config.settings import get_settings

SchemaT = TypeVar("SchemaT", bound=BaseModel)


def _chain(schema: type[SchemaT], llm: BaseChatModel | None):
    # temperature=0: extraction should be deterministic, not creative.
    model = (llm or get_llm(temperature=0)).with_structured_output(schema)
    return get_prompt("extract") | model


def extract(
    text: str, schema: type[SchemaT], llm: BaseChatModel | None = None
) -> SchemaT:
    """Extract one schema instance from one piece of text."""
    return _chain(schema, llm).invoke({"text": text})


def extract_many(
    texts: Sequence[str],
    schema: type[SchemaT],
    llm: BaseChatModel | None = None,
    max_concurrency: int | None = None,
) -> list[SchemaT | Exception]:
    """Extract from many texts in parallel. Failures come back as exceptions, in order.

    Uses LangChain's built-in ``batch``, which runs calls concurrently and
    preserves input order. For progress bars, retries and latency tracking,
    see ``aiframe.workflows.batch``.
    """
    concurrency = max_concurrency or get_settings().max_concurrency
    return _chain(schema, llm).batch(
        [{"text": t} for t in texts],
        config={"max_concurrency": concurrency},
        return_exceptions=True,
    )


def to_dataframe(
    results: Sequence[BaseModel | Exception], explode: str | None = None
) -> Any:
    """Turn extraction results into a pandas DataFrame.

    Args:
        results: Output of ``extract_many`` (exceptions become rows with an ``error``).
        explode: Name of a list field to flatten into one row per item,
            e.g. ``"contacts"`` for ``ContactList``.
    """
    import pandas as pd

    rows: list[dict[str, Any]] = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            rows.append({"doc_id": i, "error": f"{type(result).__name__}: {result}"})
            continue
        data = result.model_dump()
        if explode:
            items = data.pop(explode) or [{}]
            rows.extend({"doc_id": i, **data, **item} for item in items)
        else:
            rows.append({"doc_id": i, **data})
    return pd.DataFrame(rows)
