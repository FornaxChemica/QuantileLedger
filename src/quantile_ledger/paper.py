"""Paper-only long-option ledger (Milestone 6). Never places real orders."""

from __future__ import annotations

from quantile_ledger.errors import NotImplementedMilestoneError


def paper_buy(*_args: object, **_kwargs: object) -> None:
    raise NotImplementedMilestoneError("manual paper buy", "Milestone 6")
