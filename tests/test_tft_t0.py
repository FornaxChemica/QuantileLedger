"""T0 TFT-style quantile issuance tests."""

from __future__ import annotations

from pathlib import Path

from quantile_ledger.db import connection, initialize_database
from quantile_ledger.experiments import T0
from quantile_ledger.forecast_store import insert_forecast, upsert_experiment
from quantile_ledger.synthetic import make_synthetic_hourly_closes, one_step_log_returns
from quantile_ledger.tft_quantile import issue_t0_forecast, train_t0_model


def test_t0_issues_monotonic_contract(tmp_path: Path) -> None:
    series = make_synthetic_hourly_closes(n=300, seed=5)
    bar_horizon = 6
    model, metrics = train_t0_model(
        series.closes,
        bar_horizon=bar_horizon,
        lookback=32,
        epochs=15,
        seed=5,
        min_samples=40,
    )
    assert metrics["n"] >= 40
    issue_idx = 200
    closes = series.closes[: issue_idx + 1]
    rets = one_step_log_returns(closes)
    issued = series.bar_ends[issue_idx]
    target = series.bar_ends[issue_idx + bar_horizon]
    fc = issue_t0_forecast(
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
    assert fc.experiment_id == "T0"
    assert fc.model_family == "tft_quantile"
    assert fc.generation_metadata["backbone"] == "local_tft_attention_numpy_v1"
    assert fc.generation_metadata["monotonicity_method"] == "isotonic_pav_v1"
    vals = fc.quantile_values
    assert all(vals[i] <= vals[i + 1] + 1e-9 for i in range(len(vals) - 1))

    db = tmp_path / "t0.db"
    initialize_database(db)
    with connection(db) as conn:
        upsert_experiment(conn, T0)
        insert_forecast(conn, fc)
        row = conn.execute(
            "SELECT experiment_id, feature_set FROM forecasts WHERE forecast_id = ?",
            (fc.forecast_id,),
        ).fetchone()
        assert row is not None
        assert row["experiment_id"] == "T0"
        assert row["feature_set"] == "market_only"
