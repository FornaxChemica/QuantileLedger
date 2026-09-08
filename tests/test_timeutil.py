"""UTC timestamp helper tests."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from quantile_ledger.timeutil import ensure_utc, parse_iso_utc, to_iso_utc


def test_rejects_naive_datetime() -> None:
    with pytest.raises(ValueError, match="Naive"):
        ensure_utc(datetime(2024, 1, 2, 15, 0, 0))


def test_roundtrip_iso_z() -> None:
    value = datetime(2024, 6, 3, 14, 30, 0, tzinfo=UTC)
    text = to_iso_utc(value)
    assert text.endswith("Z")
    assert parse_iso_utc(text) == value
