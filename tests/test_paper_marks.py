"""Mark-to-market and multi-day forward paper book tests."""

from __future__ import annotations

from dataclasses import replace
from decimal import Decimal
from pathlib import Path

from quantile_ledger.db import SCHEMA_VERSION, connection, initialize_database
from quantile_ledger.mechanical import quote_from_spot
from quantile_ledger.paper import (
    compute_mark_equity,
    default_underlying_policy,
    freeze_policy,
    insert_policy,
    list_marks,
    paper_equity_summary,
    record_account_mark,
    upsert_paper_account,
)
from quantile_ledger.paper_forward import run_synthetic_forward_book


def test_schema_v5_paper_marks(tmp_path: Path) -> None:
    db = tmp_path / "m.db"
    assert initialize_database(db) == SCHEMA_VERSION == 5
    with connection(db) as conn:
        tables = {
            row["name"]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
    assert "paper_marks" in tables


def test_compute_mark_equity_hand_calculated() -> None:
    cash = Decimal("10000.0000")
    positions = {"SYN": Decimal("10.0000")}
    quote = quote_from_spot(
        ticker="SYN",
        spot=100.0,
        half_spread_bps="10",  # 10 bps half → bid 99.9, ask 100.1
        quote_time="2024-01-10T15:00:00Z",
    )
    mark = compute_mark_equity(cash, positions, {"SYN": quote})
    assert mark.equity_mid == Decimal("10000.0000") + Decimal("10") * quote.mid
    assert mark.equity_bid == Decimal("10000.0000") + Decimal("10") * quote.bid
    assert mark.equity_bid < mark.equity_mid
    # Hand numbers: mid=100, bid=99.9
    assert quote.mid == Decimal("100.0") or abs(quote.mid - Decimal("100")) < Decimal(
        "0.0001"
    )
    assert abs(quote.bid - Decimal("99.9")) < Decimal("0.0001")
    assert mark.equity_mid == Decimal("11000.0000")
    assert abs(mark.equity_bid - Decimal("10999.0000")) < Decimal("0.0001")


def test_record_mark_and_summary(tmp_path: Path) -> None:
    db = tmp_path / "m2.db"
    initialize_database(db)
    with connection(db) as conn:
        account_id = upsert_paper_account(conn, name="mark", starting_cash="10000")
        # Cash-only mark (no open position)
        mark = record_account_mark(
            conn,
            account_id=account_id,
            quotes={},
            marked_at="2024-01-10T15:00:00Z",
            quote_source="test",
            is_forward=True,
            is_synthetic=True,
        )
        assert mark.equity_mid == Decimal("10000.0000")
        assert mark.equity_bid == Decimal("10000.0000")
        summary = paper_equity_summary(conn, account_id)
        assert summary.mark_count == 1
        assert summary.equity_mid == Decimal("10000.0000")
        assert summary.return_mid == Decimal("0")


def test_synthetic_forward_book_multi_day(tmp_path: Path) -> None:
    db = tmp_path / "fwd.db"
    initialize_database(db)
    draft = replace(
        default_underlying_policy(),
        min_p50_log_return="-1.0",
        shares_per_entry="5",
        commission_per_share="0.01",
        slippage_bps="0",
        max_notional_per_trade="100000",
    )
    with connection(db) as conn:
        account_id = upsert_paper_account(conn, name="fwd", starting_cash="50000")
        insert_policy(conn, draft)
        policy = freeze_policy(conn, draft.policy_id)
        result = run_synthetic_forward_book(
            conn,
            account_id=account_id,
            policy_id=policy.policy_id,
            experiment_id="M0",
            steps=4,
            train_epochs=4,
            series_n=300,
            seed=13,
        )
        assert result.forecasts == 4
        assert result.decisions == 4
        assert result.distinct_issue_days >= 3
        assert result.marks >= 2
        marks = list_marks(conn, account_id)
        assert len(marks) == result.marks
        assert all(m.is_forward for m in marks)
        assert all(m.is_synthetic for m in marks)
        # Bid liquidation never exceeds mid accounting for the same snapshot.
        for m in marks:
            assert m.equity_bid <= m.equity_mid
        summary = paper_equity_summary(conn, account_id)
        assert summary.forward_decision_count == 4
        assert summary.mark_count == result.marks
        # Mechanical second pass is idempotent (no new decisions).
        from quantile_ledger.mechanical import run_mechanical_underlying

        second = run_mechanical_underlying(
            conn,
            account_id=account_id,
            policy_id=policy.policy_id,
            experiment_id="M0",
            is_forward=True,
        )
        assert second.decisions == 0
