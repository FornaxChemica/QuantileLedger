"""Probabilistic forecasting: baselines and M0 adapter surface."""

from __future__ import annotations

from collections.abc import Sequence

from quantile_ledger.baselines import issue_b0_persistence, issue_b1_empirical
from quantile_ledger.contract import ForecastContract
from quantile_ledger.mamba_quantile import (
    issue_m0_forecast,
    train_m0_model,
)


def run_baselines(
    *,
    mode: str,
    ticker: str,
    issued_at: str,
    origin_bar_at: str,
    target_at: str,
    spot_at_issue: float,
    horizon_hours: int,
    training_cutoff: str,
    data_as_of: str,
    closes_known_by_issue: Sequence[float] | None = None,
    bar_horizon: int = 1,
    is_synthetic: bool = False,
) -> ForecastContract:
    """Issue B0 or B1 under the common forecast contract."""
    if mode == "B0":
        return issue_b0_persistence(
            ticker=ticker,
            issued_at=issued_at,
            origin_bar_at=origin_bar_at,
            target_at=target_at,
            spot_at_issue=spot_at_issue,
            horizon_hours=horizon_hours,
            training_cutoff=training_cutoff,
            data_as_of=data_as_of,
            is_synthetic=is_synthetic,
        )
    if mode == "B1":
        if closes_known_by_issue is None:
            msg = "B1 requires closes_known_by_issue"
            raise ValueError(msg)
        return issue_b1_empirical(
            ticker=ticker,
            issued_at=issued_at,
            origin_bar_at=origin_bar_at,
            target_at=target_at,
            spot_at_issue=spot_at_issue,
            horizon_hours=horizon_hours,
            closes_known_by_issue=closes_known_by_issue,
            bar_horizon=bar_horizon,
            training_cutoff=training_cutoff,
            data_as_of=data_as_of,
            is_synthetic=is_synthetic,
        )
    msg = f"unknown baseline mode: {mode}"
    raise ValueError(msg)


__all__ = [
    "issue_b0_persistence",
    "issue_b1_empirical",
    "issue_m0_forecast",
    "run_baselines",
    "train_m0_model",
]
