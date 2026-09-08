"""Local walk-forward job orchestration (Milestone 9)."""

from __future__ import annotations

from quantile_ledger.errors import NotImplementedMilestoneError


def run_nightly(*_args: object, **_kwargs: object) -> None:
    raise NotImplementedMilestoneError("nightly walk-forward job", "Milestone 9")
