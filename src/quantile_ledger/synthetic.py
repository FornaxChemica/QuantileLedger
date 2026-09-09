"""Deterministic synthetic market path for offline demo and tests."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class SyntheticSeries:
    ticker: str
    bar_ends: list[str]
    closes: list[float]
    is_synthetic: bool = True


def make_synthetic_hourly_closes(
    *,
    ticker: str = "SYN",
    n: int = 400,
    start_price: float = 100.0,
    seed: int = 7,
) -> SyntheticSeries:
    """Labeled synthetic GBM-like path; not real market data."""
    # Simple LCG for determinism without numpy dependency.
    state = seed % 2147483647 or 1
    closes = [start_price]
    bar_ends: list[str] = []
    # Fake hourly stamps from a fixed UTC epoch day (weekday-ish sequence).
    year, month, day, hour = 2024, 1, 2, 14
    for _i in range(n):
        bar_ends.append(f"{year:04d}-{month:02d}-{day:02d}T{hour:02d}:00:00Z")
        hour += 1
        if hour >= 21:
            hour = 14
            day += 1
            if day > 28:
                day = 1
                month += 1
                if month > 12:
                    month = 1
                    year += 1
        state = (state * 48271) % 2147483647
        shock = (state / 2147483647) - 0.5
        ret = 0.0002 + 0.01 * shock
        closes.append(closes[-1] * math.exp(ret))
    # closes has n+1 points; align bar_ends to n closes used as completed bars
    return SyntheticSeries(
        ticker=ticker,
        bar_ends=bar_ends,
        closes=closes[1:],
    )


def one_step_log_returns(closes: list[float]) -> list[float]:
    out: list[float] = []
    for i in range(len(closes) - 1):
        out.append(math.log(closes[i + 1] / closes[i]))
    return out
