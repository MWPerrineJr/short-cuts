"""
Example 07 - Async batch processing
===================================

CONCEPT: Bounded concurrency.
Classifies a set of product reviews concurrently (at most MAX_CONCURRENCY
calls in flight), with retries, then treats the run itself as data:
error rate, latency percentiles, label distribution.

Run from the ai-framework folder:
    python examples/07_batch.py              # concurrent run
    python examples/07_batch.py --compare    # also time concurrency=1 (costs 2x calls)
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

# Make `config` and `aiframe` importable when this file is run directly.
# The project imports below come after this on purpose.
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from rich.console import Console

from aiframe.workflows.batch import classify_sentiment, summarize, to_dataframe
from config.settings import ConfigError, get_settings

console = Console()

REVIEWS = [
    "Setup took five minutes and the dashboards look great.",
    "Forecasts are way off for our seasonal products. Disappointed.",
    "It's fine. Does what it says, nothing special.",
    "Support fixed our NX-404 issue within an hour. Impressed!",
    "Pricing jumped at renewal with no warning. Considering alternatives.",
    "The Snowflake connector is fast and reliable.",
    "Scheduled reports keep timing out (NX-500). Very frustrating.",
    "Decent tool, but the learning curve for new analysts is steep.",
    "Our execs actually read the weekly reports now. Huge win.",
    "Average experience overall.",
    "Single sign-on setup was painless on Enterprise.",
    "Exports to CSV drop columns randomly. Unusable for finance.",
]


async def run(concurrency: int) -> tuple[list, float]:
    start = time.perf_counter()
    results = await classify_sentiment(REVIEWS, max_concurrency=concurrency)
    return results, time.perf_counter() - start


async def amain(compare: bool) -> None:
    concurrency = get_settings().max_concurrency
    results, elapsed = await run(concurrency)
    console.print(
        f"\nconcurrency={concurrency}: [bold]{elapsed:.1f}s[/] for {len(REVIEWS)} items"
    )
    console.print(summarize(results))

    df = to_dataframe(results)
    cols = [c for c in ("input", "label", "score", "latency_s", "error") if c in df]
    console.print(df[cols].to_string(index=False, max_colwidth=50))
    if "label" in df:
        console.print("\nLabel distribution:\n", df["label"].value_counts().to_string())

    if compare:
        _, sequential = await run(1)
        console.print(
            f"\nconcurrency=1: [bold]{sequential:.1f}s[/]  ->  speedup {sequential / elapsed:.1f}x"
        )
    # TRY: set MAX_CONCURRENCY=1, 5, 20 in .env and plot elapsed time vs concurrency.


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--compare", action="store_true", help="also run sequentially")
    args = parser.parse_args()
    try:
        get_settings().require_llm()
    except ConfigError as err:
        console.print(
            f"[red]{err}[/]\nRun examples/01_check_setup.py for a full report."
        )
        return 1
    asyncio.run(amain(args.compare))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
