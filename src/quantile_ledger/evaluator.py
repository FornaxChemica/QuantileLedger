"""Walk-forward calibration and paired comparison summaries."""

from __future__ import annotations

from collections.abc import Sequence

from quantile_ledger.compare import (
    PairedComparison,
    SettledForecast,
    build_paired_cohort,
)


def summarize_calibration(
    forecasts: Sequence[SettledForecast],
    *,
    experiment_ids: Sequence[str],
    baseline_id: str = "B1",
) -> PairedComparison:
    """Build a strict paired cohort and score each experiment."""
    return build_paired_cohort(forecasts, experiment_ids=experiment_ids)


__all__ = [
    "PairedComparison",
    "SettledForecast",
    "build_paired_cohort",
    "summarize_calibration",
]
