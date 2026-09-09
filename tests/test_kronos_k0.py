"""K0 Kronos sample → quantile tests (fake adapter; no torch)."""

from __future__ import annotations

from pathlib import Path

import pytest

from quantile_ledger.db import connection, initialize_database
from quantile_ledger.errors import InsufficientDataError, ModelUnavailableError
from quantile_ledger.experiments import K0
from quantile_ledger.forecast_store import insert_forecast, upsert_experiment
from quantile_ledger.kronos_quantile import (
    DISTRIBUTION_CLAIM,
    QUANTILE_METHOD,
    FakeKronosSampler,
    KronosLocalPaths,
    RealKronosSampler,
    closes_to_ohlc_bars,
    empirical_quantile,
    issue_k0_forecast,
    kronos_bundle_ready,
    require_kronos_runtime_deps,
    terminal_closes_to_return_quantiles,
)
from quantile_ledger.synthetic import make_synthetic_hourly_closes


def test_empirical_quantile_hand_calculated() -> None:
    # Sorted sample [1,2,3,4]; q=0.5 → midpoint between 2 and 3 = 2.5
    assert empirical_quantile([1.0, 2.0, 3.0, 4.0], 0.5) == pytest.approx(2.5)
    assert empirical_quantile([10.0], 0.1) == pytest.approx(10.0)


def test_terminal_closes_to_return_quantiles_monotonic() -> None:
    spot = 100.0
    terminals = [90.0, 95.0, 100.0, 105.0, 110.0, 120.0, 130.0, 140.0]
    levels = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)
    values = terminal_closes_to_return_quantiles(
        terminals, spot_at_issue=spot, quantile_levels=levels
    )
    assert all(values[i] <= values[i + 1] + 1e-12 for i in range(len(values) - 1))
    mid = levels.index(0.5)
    # Median of 8 samples is mean of 4th and 5th → log(105/100) and log(110/100)
    import math

    expected = 0.5 * (math.log(105 / 100) + math.log(110 / 100))
    assert values[mid] == pytest.approx(expected)


def test_fake_k0_issues_contract(tmp_path: Path) -> None:
    series = make_synthetic_hourly_closes(n=200, seed=11)
    bar_horizon = 6
    issue_idx = 150
    closes = series.closes[: issue_idx + 1]
    bars = closes_to_ohlc_bars(closes)
    issued = series.bar_ends[issue_idx]
    target = series.bar_ends[issue_idx + bar_horizon]
    future = series.bar_ends[issue_idx + 1 : issue_idx + bar_horizon + 1]
    sampler = FakeKronosSampler(lookback=64)
    fc = issue_k0_forecast(
        sampler,
        bars=bars,
        bar_ends=series.bar_ends[: issue_idx + 1],
        future_bar_ends=future,
        ticker=series.ticker,
        issued_at=issued,
        origin_bar_at=issued,
        target_at=target,
        spot_at_issue=closes[-1],
        horizon_hours=bar_horizon,
        pred_len=bar_horizon,
        training_cutoff=issued,
        data_as_of=issued,
        seed=11,
        is_synthetic=True,
    )
    assert fc.experiment_id == "K0"
    assert fc.model_family == "kronos"
    assert fc.generation_metadata["quantile_method"] == QUANTILE_METHOD
    assert fc.generation_metadata["distribution_claim"] == DISTRIBUTION_CLAIM
    assert fc.generation_metadata["sample_count"] == K0.hyperparameters["sample_count"]
    assert fc.generation_metadata["sampler_metadata"]["is_fake"] is True
    vals = fc.quantile_values
    assert all(vals[i] <= vals[i + 1] + 1e-9 for i in range(len(vals) - 1))

    db = tmp_path / "k0.db"
    initialize_database(db)
    with connection(db) as conn:
        upsert_experiment(conn, K0)
        insert_forecast(conn, fc)
        row = conn.execute(
            "SELECT experiment_id, feature_set FROM forecasts WHERE forecast_id = ?",
            (fc.forecast_id,),
        ).fetchone()
        assert row is not None
        assert row["experiment_id"] == "K0"
        assert row["feature_set"] == "market_only"


def test_fake_k0_insufficient_lookback() -> None:
    sampler = FakeKronosSampler(lookback=64)
    bars = closes_to_ohlc_bars([100.0, 101.0, 102.0])
    with pytest.raises(InsufficientDataError):
        sampler.sample_terminal_closes(
            bars,
            bar_ends=["2024-01-01T14:00:00Z"] * 3,
            future_bar_ends=["2024-01-01T15:00:00Z"],
            spot_at_issue=102.0,
            pred_len=1,
            sample_count=8,
            temperature=1.0,
            top_p=0.9,
            seed=0,
        )


def test_real_kronos_refuses_missing_weights(tmp_path: Path) -> None:
    paths = KronosLocalPaths(
        weights_dir=tmp_path / "missing_w",
        source_dir=tmp_path / "missing_s",
    )
    sampler = RealKronosSampler(paths=paths, allow_hub_download=False)
    # Missing deps or missing weights both surface as ModelUnavailableError.
    bars = closes_to_ohlc_bars([100.0 + i * 0.1 for i in range(80)])
    ends = [f"2024-01-02T{14 + (i % 7):02d}:00:00Z" for i in range(80)]
    with pytest.raises(ModelUnavailableError):
        sampler.sample_terminal_closes(
            bars,
            bar_ends=ends,
            future_bar_ends=["2024-01-10T14:00:00Z"] * 6,
            spot_at_issue=bars[-1].close,
            pred_len=6,
            sample_count=4,
            temperature=1.0,
            top_p=0.9,
            seed=0,
        )


def test_kronos_bundle_ready_false_when_empty(tmp_path: Path) -> None:
    paths = KronosLocalPaths.under(tmp_path)
    assert kronos_bundle_ready(paths) is False


def test_require_deps_message_when_missing() -> None:
    # Soft check: function either passes (deps present) or raises clearly.
    try:
        require_kronos_runtime_deps()
    except ModelUnavailableError as exc:
        assert "uv sync --extra ml" in str(exc)
