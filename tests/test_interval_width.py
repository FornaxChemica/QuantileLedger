"""Regression tests locking non-degenerate M0/T0 quantile intervals.

These guard against the interval-collapse bug where M0/T0 quantile heads
produced ~zero-width P10-P90 intervals (and therefore 0% coverage). A trained
model must emit a strictly positive, genuinely fanned predictive interval.
"""

from __future__ import annotations

import math

from quantile_ledger.contract import DEFAULT_QUANTILES
from quantile_ledger.mamba_quantile import MambaQuantileModel, train_m0_model
from quantile_ledger.synthetic import make_synthetic_hourly_closes, one_step_log_returns
from quantile_ledger.tft_quantile import TemporalFusionQuantileModel, train_t0_model

_P10 = DEFAULT_QUANTILES.index(0.10)
_P50 = DEFAULT_QUANTILES.index(0.50)
_P90 = DEFAULT_QUANTILES.index(0.90)
_MIN_WIDTH = 1e-4


def _trained_m0(n: int = 400) -> MambaQuantileModel:
    series = make_synthetic_hourly_closes(n=n, seed=11)
    model, _ = train_m0_model(
        series.closes,
        bar_horizon=1,
        lookback=32,
        epochs=40,
        seed=11,
        min_samples=40,
    )
    return model


def _trained_t0(n: int = 400) -> TemporalFusionQuantileModel:
    series = make_synthetic_hourly_closes(n=n, seed=11)
    model, _ = train_t0_model(
        series.closes,
        bar_horizon=1,
        lookback=32,
        epochs=40,
        seed=11,
        min_samples=40,
    )
    return model


def test_m0_interval_width_is_positive_and_fanned() -> None:
    series = make_synthetic_hourly_closes(n=400, seed=11)
    rets = one_step_log_returns(series.closes)
    model = _trained_m0()

    q = model.predict_quantiles(rets[-32:])
    assert all(q[i] <= q[i + 1] + 1e-9 for i in range(len(q) - 1)), "quantiles cross"
    width = q[_P90] - q[_P10]
    assert width > _MIN_WIDTH, f"M0 P10-P90 width collapsed: {width}"
    # Genuinely fanned: the interior spread is not a single degenerate point.
    assert q[_P90] - q[_P50] > _MIN_WIDTH / 2
    assert q[_P50] - q[_P10] > _MIN_WIDTH / 2


def test_t0_interval_width_is_positive_and_fanned() -> None:
    series = make_synthetic_hourly_closes(n=400, seed=11)
    rets = one_step_log_returns(series.closes)
    model = _trained_t0()

    q = model.predict_quantiles(rets[-32:])
    assert all(q[i] <= q[i + 1] + 1e-9 for i in range(len(q) - 1)), "quantiles cross"
    width = q[_P90] - q[_P10]
    assert width > _MIN_WIDTH, f"T0 P10-P90 width collapsed: {width}"
    assert q[_P90] - q[_P50] > _MIN_WIDTH / 2
    assert q[_P50] - q[_P10] > _MIN_WIDTH / 2


def test_m0_interval_covers_many_outcomes() -> None:
    """A non-degenerate interval should capture most realized returns."""
    n = 500
    series = make_synthetic_hourly_closes(n=n, seed=11)
    rets = one_step_log_returns(series.closes)
    model = _trained_m0(n=n)

    hits = 0
    total = 0
    for t in range(32, len(rets)):
        q = model.predict_quantiles(rets[t - 32 : t])
        y = rets[t]
        if q[_P10] <= y <= q[_P90]:
            hits += 1
        total += 1
    assert total > 0
    # Degenerate (zero-width) intervals score ~0 coverage; a real interval
    # should be well above that. Loose bound to stay deterministic-robust —
    # the regression we guard against drives coverage to ~0, not merely low.
    assert hits / total > 0.3, f"M0 P10-P90 coverage collapsed: {hits}/{total}"


def test_t0_interval_covers_many_outcomes() -> None:
    n = 500
    series = make_synthetic_hourly_closes(n=n, seed=11)
    rets = one_step_log_returns(series.closes)
    model = _trained_t0(n=n)

    hits = 0
    total = 0
    for t in range(32, len(rets)):
        q = model.predict_quantiles(rets[t - 32 : t])
        y = rets[t]
        if q[_P10] <= y <= q[_P90]:
            hits += 1
        total += 1
    assert total > 0
    assert hits / total > 0.3, f"T0 P10-P90 coverage collapsed: {hits}/{total}"


def test_width_matches_synthetic_scale_order_of_magnitude() -> None:
    """Sanity: interval width is in the right ballpark for the synthetic sigma.

    The synthetic path uses ~1% one-step shocks; a sane P10-P90 width is on the
    order of 1e-2, not 1e-7 (collapsed) or 1e+1 (diverged).
    """
    series = make_synthetic_hourly_closes(n=400, seed=11)
    rets = one_step_log_returns(series.closes)
    for model in (_trained_m0(), _trained_t0()):
        q = model.predict_quantiles(rets[-32:])
        width = q[_P90] - q[_P10]
        assert 1e-3 < width < 1.0, f"width out of sane range: {width}"
        assert math.isfinite(width)
