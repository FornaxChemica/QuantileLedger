"""Phase G news ablation tests (T1/T2/M1/N0)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from quantile_ledger.ablations import (
    issue_m1_from_m0,
    issue_n0_news_only,
    issue_t1_from_t0,
    issue_t2_from_t0,
    load_context_for_ablation,
)
from quantile_ledger.db import connection, initialize_database
from quantile_ledger.errors import InsufficientDataError
from quantile_ledger.experiments import M0, M1, N0, T0, T1, T2
from quantile_ledger.forecast_store import insert_forecast, upsert_experiment
from quantile_ledger.mamba_quantile import issue_m0_forecast, train_m0_model
from quantile_ledger.news import import_news_file
from quantile_ledger.sentiment import (
    FakeFinBERT,
    build_sentiment_context,
    score_unscored_news,
)
from quantile_ledger.synthetic import make_synthetic_hourly_closes, one_step_log_returns
from quantile_ledger.tft_quantile import issue_t0_forecast, train_t0_model


def _seed_news_and_parents(tmp_path: Path) -> tuple[Path, object, object, object]:
    db = tmp_path / "a.db"
    initialize_database(db)
    series = make_synthetic_hourly_closes(n=220, seed=9)
    bar_horizon = 6
    issue_idx = 180
    closes = series.closes[: issue_idx + 1]
    rets = one_step_log_returns(closes)
    issued = series.bar_ends[issue_idx]
    target = series.bar_ends[issue_idx + bar_horizon]
    spot = closes[-1]
    news_path = tmp_path / "news.json"
    news_path.write_text(
        json.dumps(
            [
                {
                    "ticker": series.ticker,
                    "headline": "Company beats estimates and shares surge",
                    "published_at": series.bar_ends[issue_idx - 4],
                    "ingested_at": series.bar_ends[issue_idx - 3],
                    "is_synthetic": True,
                }
            ]
        ),
        encoding="utf-8",
    )
    m0_model, _ = train_m0_model(
        closes, bar_horizon=bar_horizon, lookback=32, epochs=8, seed=9, min_samples=40
    )
    t0_model, _ = train_t0_model(
        closes, bar_horizon=bar_horizon, lookback=32, epochs=8, seed=9, min_samples=40
    )
    with connection(db) as conn:
        for spec in (M0, T0, T1, T2, M1, N0):
            upsert_experiment(conn, spec)
        import_news_file(conn, news_path, force_synthetic=True)
        score_unscored_news(conn, scorer=FakeFinBERT())
        m0 = issue_m0_forecast(
            m0_model,
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
        t0 = issue_t0_forecast(
            t0_model,
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
        insert_forecast(conn, m0)
        insert_forecast(conn, t0)
        ctx = load_context_for_ablation(
            conn, ticker=series.ticker, issued_at=issued, lookback_hours=72
        )
    return db, m0, t0, ctx


def test_ablations_shift_and_refuse_missing(tmp_path: Path) -> None:
    db, m0, t0, ctx = _seed_news_and_parents(tmp_path)
    assert ctx.status == "scored"
    t1 = issue_t1_from_t0(t0, ctx)
    t2 = issue_t2_from_t0(t0, ctx)
    m1 = issue_m1_from_m0(m0, ctx)
    assert t1.experiment_id == "T1"
    assert t2.experiment_id == "T2"
    assert m1.experiment_id == "M1"
    assert t1.parent_forecast_id == t0.forecast_id
    assert t1.sentiment_context is not None
    assert t1.feature_set == "market_plus_news_volume"
    assert t2.feature_set == "market_plus_news"
    # Volume-only vs volume+sentiment should differ when polarity nonzero.
    assert t1.quantile_values != t2.quantile_values
    n0 = issue_n0_news_only(
        ticker=t0.ticker,
        issued_at=t0.issued_at,
        origin_bar_at=t0.origin_bar_at,
        target_at=t0.target_at,
        spot_at_issue=t0.spot_at_issue,
        horizon_hours=t0.nominal_horizon_hours,
        training_cutoff=t0.training_cutoff,
        data_as_of=t0.data_as_of,
        ctx=ctx,
        is_synthetic=True,
    )
    assert n0.feature_set == "news_only"
    assert n0.sentiment_context is not None

    with connection(db) as conn:
        missing = build_sentiment_context(
            conn, ticker="ZZZ", issued_at=t0.issued_at, lookback_hours=72
        )
    with pytest.raises(InsufficientDataError, match="missing"):
        issue_t1_from_t0(t0, missing)


def test_partial_status_and_registry() -> None:
    assert T1.hyperparameters["parent_experiment_id"] == "T0"
    assert T2.hyperparameters["require_full_scores"] is True
    assert M1.hyperparameters["parent_experiment_id"] == "M0"
    assert N0.feature_set == "news_only"
    assert N0.version == "2"


def test_market_only_parent_unchanged(tmp_path: Path) -> None:
    _, m0, t0, ctx = _seed_news_and_parents(tmp_path)
    _ = issue_t1_from_t0(t0, ctx)
    assert t0.feature_set == "market_only"
    assert m0.feature_set == "market_only"
    # Parent contract object is not mutated.
    assert t0.sentiment_context is None
