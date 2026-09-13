"""Phase H realized-vol regime tests."""

from __future__ import annotations

import math

import pytest

from quantile_ledger.compare import SettledForecast, score_by_regime
from quantile_ledger.errors import ForecastValidationError
from quantile_ledger.regimes import (
    REGIME_HIGH,
    REGIME_LOW,
    REGIME_MID,
    assign_vol_regime,
    fit_vol_tercile_thresholds,
    realized_vol,
    regime_for_closes,
)


def test_realized_vol_and_terciles() -> None:
    closes = [100.0]
    for i in range(40):
        closes.append(closes[-1] * math.exp(0.001 * ((-1) ** i)))
    vol = realized_vol(closes, lookback=20)
    assert vol > 0
    vols = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06]
    thr = fit_vol_tercile_thresholds(vols, fit_as_of_max="2024-01-01T00:00:00Z")
    low, _ = assign_vol_regime(0.01, thr)
    mid, _ = assign_vol_regime(thr.low_max + 1e-9, thr)
    high, reasons = assign_vol_regime(thr.mid_max + 1.0, thr)
    assert low == REGIME_LOW
    assert mid in {REGIME_MID, REGIME_HIGH}
    assert high == REGIME_HIGH
    assert reasons["n_fit"] == 6


def test_regime_pit_cutoff() -> None:
    closes = [100.0 + i * 0.1 for i in range(30)]
    ends = [f"2024-01-01T{h:02d}:00:00Z" for h in range(30)]
    thr = fit_vol_tercile_thresholds(
        [0.01, 0.02, 0.03], fit_as_of_max=ends[20], lookback=20
    )
    label, _ = regime_for_closes(
        closes[:21],
        thresholds=thr,
        bar_ends=ends[:21],
        issued_at=ends[20],
    )
    assert label in {REGIME_LOW, REGIME_MID, REGIME_HIGH}
    with pytest.raises(ForecastValidationError, match="PIT"):
        regime_for_closes(
            closes[:21],
            thresholds=thr,
            bar_ends=ends[:21],
            issued_at=ends[10],
        )


def test_score_by_regime_includes_n() -> None:
    levels = (0.1, 0.5, 0.9)
    rows = [
        SettledForecast(
            forecast_id=f"f{i}",
            experiment_id="M0",
            ticker="SYN",
            issued_at=f"2024-01-{i + 1:02d}T15:00:00Z",
            origin_bar_at=f"2024-01-{i + 1:02d}T15:00:00Z",
            target_at=f"2024-01-{i + 1:02d}T21:00:00Z",
            horizon_hours=6,
            spot_at_issue=100.0,
            quantile_levels=levels,
            quantile_values=(-0.02, 0.0, 0.02),
            actual_return=0.0,
            regime="vol_low" if i < 2 else "vol_high",
        )
        for i in range(4)
    ]
    by_reg = score_by_regime(rows, experiment_id="M0")
    assert by_reg["vol_low"]["sample_count"] == 2
    assert by_reg["vol_high"]["sample_count"] == 2
    assert "coverage_p10_p90" in by_reg["vol_low"]
    assert "mean_width_return" in by_reg["vol_low"]
