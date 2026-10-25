"""
Batch workflow: async, rate-limited processing of many items.

WHY THIS PATTERN
----------------
LLM calls spend most of their time waiting on the network. Running them one
at a time wastes that time; firing them all at once triggers provider rate
limits (HTTP 429). The answer is bounded concurrency:

- ``asyncio.Semaphore(n)`` allows at most n calls in flight.
- ``tenacity`` retries transient failures with exponential backoff.
- Each item records output, error, attempts and latency, so the run can be
  analyzed like any other dataset (error rate, p95 latency, cost).

Typical speedup over a sequential loop is roughly ``n``x until you hit the
provider's limits. Tune ``max_concurrency`` in ``.env``.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Generic, TypeVar

from tenacity import AsyncRetrying, RetryError, stop_after_attempt, wait_exponential
from tqdm.asyncio import tqdm_asyncio

from aiframe.core import get_llm
from aiframe.core.schemas import Sentiment
from config.settings import get_settings

InT = TypeVar("InT")
OutT = TypeVar("OutT")


@dataclass
class BatchItem(Generic[InT, OutT]):
    index: int
    input: InT
    output: OutT | None = None
    error: str | None = None
    attempts: int = 0
    latency_s: float = 0.0

    @property
    def ok(self) -> bool:
        return self.error is None


async def run_batch(
    items: Sequence[InT],
    worker: Callable[[InT], Awaitable[OutT]],
    max_concurrency: int | None = None,
    max_attempts: int = 3,
    progress: bool = True,
) -> list[BatchItem[InT, OutT]]:
    """Run ``worker`` over every item with bounded concurrency and retries.

    Results are returned in input order. One item failing never stops the batch.
    """
    semaphore = asyncio.Semaphore(max_concurrency or get_settings().max_concurrency)

    async def process(index: int, item: InT) -> BatchItem[InT, OutT]:
        record: BatchItem[InT, OutT] = BatchItem(index=index, input=item)
        async with semaphore:
            start = time.perf_counter()
            try:
                async for attempt in AsyncRetrying(
                    stop=stop_after_attempt(max_attempts),
                    wait=wait_exponential(multiplier=0.5, max=8),
                    reraise=True,
                ):
                    with attempt:
                        record.attempts = attempt.retry_state.attempt_number
                        record.output = await worker(item)
            except (Exception, RetryError) as err:  # noqa: BLE001 - recorded, not swallowed
                record.error = f"{type(err).__name__}: {err}"
            record.latency_s = round(time.perf_counter() - start, 3)
        return record

    tasks = [process(i, item) for i, item in enumerate(items)]
    if progress:
        return list(await tqdm_asyncio.gather(*tasks, desc="batch"))
    return list(await asyncio.gather(*tasks))


def summarize(results: Sequence[BatchItem]) -> dict[str, Any]:
    """Run-level metrics: the same numbers you'd monitor in production."""
    latencies = sorted(r.latency_s for r in results)
    n = len(results)
    failed = sum(not r.ok for r in results)
    return {
        "items": n,
        "failed": failed,
        "error_rate": round(failed / n, 3) if n else 0.0,
        "retries": sum(max(r.attempts - 1, 0) for r in results),
        "p50_latency_s": latencies[n // 2] if n else 0.0,
        "p95_latency_s": latencies[min(n - 1, int(n * 0.95))] if n else 0.0,
    }


def to_dataframe(results: Sequence[BatchItem]) -> Any:
    """Flatten results (including Pydantic outputs) into a pandas DataFrame."""
    import pandas as pd

    rows = []
    for r in results:
        row = asdict(r)
        output = row.pop("output")
        if hasattr(r.output, "model_dump"):
            row.update(r.output.model_dump())
        else:
            row["output"] = output
        rows.append(row)
    return pd.DataFrame(rows)


async def classify_sentiment(
    texts: Sequence[str], **kwargs: Any
) -> list[BatchItem[str, Sentiment]]:
    """Ready-made batch job: sentiment for each text, as validated ``Sentiment`` objects."""
    model = get_llm(temperature=0).with_structured_output(Sentiment)

    async def worker(text: str) -> Sentiment:
        return await model.ainvoke(f"Classify the sentiment of this text:\n\n{text}")

    return await run_batch(texts, worker, **kwargs)
