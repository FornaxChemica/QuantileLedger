"""Read-only market data provider boundaries (implemented in Milestone 2)."""

from __future__ import annotations

from quantile_ledger.errors import NotImplementedMilestoneError


def fetch_bars(*_args: object, **_kwargs: object) -> None:
    raise NotImplementedMilestoneError("keyless price fetch", "Milestone 2")
