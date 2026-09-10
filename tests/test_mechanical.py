"""Mechanical equity long/flat runner tests."""

from __future__ import annotations

from dataclasses import replace
from decimal import Decimal
from pathlib import Path

from quantile_ledger.baselines import issue_b0_persistence
from quantile_ledger.db import connection, initialize_database
from quantile_ledger.experiments import M0, T0
from quantile_ledger.forecast_store import (
    insert_forecast,
    insert_outcome,
    upsert_experiment,
)
from quantile_ledger.mamba_quantile import issue_m0_forecast, train_m0_model
from quantile_ledger.mechanical import run_mechanical_underlying
from quantile_ledger.paper import (
    default_underlying_policy,
    freeze_policy,
    insert_policy,
    latest_cash_balance,
    open_share_quantity,
    upsert_paper_account,
)
from quantile_ledger.synthetic import make_synthetic_hourly_closes, one_step_log_returns
from quantile_ledger.tft_quantile import issue_t0_forecast, train_t0_model


def _seed_challenger_forecasts(db: Path) -> None:
    series = make_synthetic_hourly_closes(n=220, seed=9)
    bar_horizon = 6
    issue_idx = 180
    closes = series.closes[: issue_idx + 1]
    rets = one_step_log_returns(closes)
    issued = series.bar_ends[issue_idx]
    target = series.bar_ends[issue_idx + bar_horizon]
    spot = closes[-1]
    outcome = series.closes[issue_idx + bar_horizon]
    actual = __import__("math").log(outcome / spot)

    m0_model, _ = train_m0_model(
        closes, bar_horizon=bar_horizon, lookback=32, epochs=12, seed=9, min_samples=40
    )
    t0_model, _ = train_t0_model(
        closes, bar_horizon=bar_horizon, lookback=32, epochs=12, seed=9, min_samples=40
    )
    initialize_database(db)
    with connection(db) as conn:
        for spec in (M0, T0):
            upsert_experiment(conn, spec)
        for issue_fn, model in (
            (issue_m0_forecast, m0_model),
            (issue_t0_forecast, t0_model),
        ):
            fc = issue_fn(
                model,
                ticker=series.ticker,
                issued_at=issued,
                origin_bar_at=issued,
                target_at=target,
                spot_at_issue=spot,
                horizon_hours=bar_horizon,
                recent_one_step_log_returns=rets,
                training_cutoff=issued,
                data_as_of=issued,
                is_synthetic=True,
            )
            # Force a clear long signal for mechanical entry test on M0 only
            # by leaving model output as-is; policy min is 0.001.
            insert_forecast(conn, fc)
            qmap = dict(zip(fc.quantile_levels, fc.quantile_values, strict=True))
            insert_outcome(
                conn,
                forecast_id=fc.forecast_id,
                outcome_price=outcome,
                actual_return=actual,
                outcome_bar_at=target,
                settled_at=target,
                p10=qmap[0.10],
                p90=qmap[0.90],
            )


def test_mechanical_run_idempotent_and_fills(tmp_path: Path) -> None:
    db = tmp_path / "m.db"
    _seed_challenger_forecasts(db)
    draft = replace(
        default_underlying_policy(),
        # Low bar so trained synthetic p50 usually triggers long.
        min_p50_log_return="-1.0",
        shares_per_entry="5",
        commission_per_share="0.01",
        slippage_bps="0",
        max_notional_per_trade="100000",
    )
    with connection(db) as conn:
        account_id = upsert_paper_account(conn, name="mech", starting_cash="50000")
        insert_policy(conn, draft)
        policy = freeze_policy(conn, draft.policy_id)
        first = run_mechanical_underlying(
            conn,
            account_id=account_id,
            policy_id=policy.policy_id,
            experiment_id="M0",
            is_forward=True,
        )
        assert first.decisions == 1
        assert first.entries == 1
        assert open_share_quantity(conn, account_id, "SYN") == Decimal("5.0000")
        bal1 = latest_cash_balance(conn, account_id)
        second = run_mechanical_underlying(
            conn,
            account_id=account_id,
            policy_id=policy.policy_id,
            experiment_id="M0",
            is_forward=True,
        )
        assert second.decisions == 0
        assert second.entries == 0
        assert latest_cash_balance(conn, account_id) == bal1


def test_mechanical_flat_records_without_fill(tmp_path: Path) -> None:
    db = tmp_path / "m2.db"
    initialize_database(db)
    series_issued = "2024-01-10T15:00:00Z"
    fc = issue_b0_persistence(
        ticker="SYN",
        issued_at=series_issued,
        origin_bar_at=series_issued,
        target_at="2024-01-10T21:00:00Z",
        spot_at_issue=100.0,
        horizon_hours=6,
        training_cutoff=series_issued,
        data_as_of=series_issued,
        is_synthetic=True,
    )
    # Re-tag as M0 for runner eligibility while keeping p50=0 → flat under default min.
    fc = fc.model_copy(update={"experiment_id": "M0", "model_family": "mamba_quantile"})
    draft = replace(
        default_underlying_policy(),
        min_p50_log_return="0.0010",
    )
    with connection(db) as conn:
        upsert_experiment(conn, M0)
        insert_forecast(conn, fc)
        qmap = dict(zip(fc.quantile_levels, fc.quantile_values, strict=True))
        insert_outcome(
            conn,
            forecast_id=fc.forecast_id,
            outcome_price=100.0,
            actual_return=0.0,
            outcome_bar_at="2024-01-10T21:00:00Z",
            settled_at="2024-01-10T21:00:00Z",
            p10=qmap[0.10],
            p90=qmap[0.90],
        )
        account_id = upsert_paper_account(conn, name="flat", starting_cash="10000")
        insert_policy(conn, draft)
        policy = freeze_policy(conn, draft.policy_id)
        result = run_mechanical_underlying(
            conn,
            account_id=account_id,
            policy_id=policy.policy_id,
            experiment_id="M0",
        )
        assert result.decisions == 1
        assert result.entries == 0
        assert result.flats_recorded == 1
        assert open_share_quantity(conn, account_id, "SYN") == Decimal("0")
        row = conn.execute(
            "SELECT action, reason FROM paper_decisions WHERE forecast_id = ?",
            (fc.forecast_id,),
        ).fetchone()
        assert row is not None
        assert row["action"] == "flat"
