"""Probabilistic forecasting baselines and TFT challenger."""

from __future__ import annotations

from quantile_ledger.errors import NotImplementedMilestoneError


def run_baselines(*_args: object, **_kwargs: object) -> None:
    raise NotImplementedMilestoneError("baseline forecasts", "Milestone 1")
