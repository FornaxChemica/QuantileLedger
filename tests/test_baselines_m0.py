"""Baseline and M0 issuance tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from quantile_ledger.baselines import issue_b0_persistence, issue_b1_empirical
from quantile_ledger.db import connection, initialize_database
from quantile_ledger.errors import InsufficientDataError
from quantile_ledger.experiments import B0, B1, M0
from quantile_ledger.forecast_store import insert_forecast, upsert_experiment
from quantile_ledger.mamba_quantile import issue_m0_forecast, train_m0_model
from quantile_ledger.synthetic import make_synthetic_hourly_closes, one_step_log_returns


def test_b0_median_zero() -> None:
    fc = issue_b0_persistence(
        ticker="SYN",
        issued_at="2024-01-10T15:00:00Z",
        origin_bar_at="2024-01-10T15:00:00Z",
        target_at="2024-01-10T21:00:00Z",
        spot_at_issue=100.0,
        horizon_hours=6,
        training_cutoff="2024-01-10T15:00:00Z",
        data_as_of="2024-01-10T15:00:00Z",
        is_synthetic=True,
    )
    assert fc.experiment_id == "B0"
    mid = fc.quantile_levels.index(0.5)
    assert fc.quantile_values[mid] == 0.0


def test_b1_insufficient_history() -> None:
    with pytest.raises(InsufficientDataError):
        issue_b1_empirical(
            ticker="SYN",
            issued_at="2024-01-10T15:00:00Z",
            origin_bar_at="2024-01-10T15:00:00Z",
            target_at="2024-01-10T21:00:00Z",
            spot_at_issue=100.0,
            horizon_hours=6,
            closes_known_by_issue=[100.0, 101.0, 102.0],
            bar_horizon=1,
            training_cutoff="2024-01-10T15:00:00Z",
            data_as_of="2024-01-10T15:00:00Z",
            min_samples=30,
        )


def test_m0_issues_monotonic_contract(tmp_path: Path) -> None:
    series = make_synthetic_hourly_closes(n=300, seed=3)
    bar_horizon = 6
    model, _ = train_m0_model(
        series.closes,
        bar_horizon=bar_horizon,
        lookback=32,
        epochs=15,
        seed=3,
        min_samples=40,
    )
    issue_idx = 200
    closes = series.closes[: issue_idx + 1]
    rets = one_step_log_returns(closes)
    issued = series.bar_ends[issue_idx]
    target = series.bar_ends[issue_idx + bar_horizon]
    fc = issue_m0_forecast(
        model,
        ticker=series.ticker,
        issued_at=issued,
        origin_bar_at=issued,
        target_at=target,
        spot_at_issue=closes[-1],
        horizon_hours=bar_horizon,
        recent_one_step_log_returns=rets,
        training_cutoff=issued,
        data_as_of=issued,
        is_synthetic=True,
    )
    assert fc.experiment_id == "M0"
    assert fc.generation_metadata["monotonicity_method"] == "isotonic_pav_v1"
    vals = fc.quantile_values
    assert all(vals[i] <= vals[i + 1] + 1e-9 for i in range(len(vals) - 1))

    db = tmp_path / "t.db"
    initialize_database(db)
    with connection(db) as conn:
        for spec in (B0, B1, M0):
            upsert_experiment(conn, spec)
        insert_forecast(conn, fc)
        row = conn.execute(
            "SELECT experiment_id, feature_set FROM forecasts WHERE forecast_id = ?",
            (fc.forecast_id,),
        ).fetchone()
        assert row is not None
        assert row["experiment_id"] == "M0"
        assert row["feature_set"] == "market_only"
