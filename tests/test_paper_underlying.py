"""Underlying long/flat paper policy tests (Decimal costs, freeze gate)."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pytest

from quantile_ledger.db import SCHEMA_VERSION, connection, initialize_database
from quantile_ledger.errors import PaperRiskRejectionError
from quantile_ledger.paper import (
    EquityQuote,
    build_entry_fill,
    build_exit_fill,
    decide_long_flat,
    default_underlying_policy,
    freeze_policy,
    insert_decision,
    insert_policy,
    latest_cash_balance,
    open_share_quantity,
    record_fill,
    upsert_paper_account,
)


def test_schema_v4_paper_tables(tmp_path: Path) -> None:
    db = tmp_path / "p.db"
    assert initialize_database(db) == SCHEMA_VERSION == 6
    with connection(db) as conn:
        tables = {
            row["name"]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
    assert "paper_policies" in tables
    assert "paper_decisions" in tables
    assert "paper_fills" in tables
    assert "paper_cash_ledger" in tables
    assert "paper_marks" in tables
    assert "news_sentiment" in tables


def test_decide_long_flat_and_refuse_unfrozen() -> None:
    policy = default_underlying_policy()
    with pytest.raises(PaperRiskRejectionError):
        decide_long_flat(
            policy=policy,
            ticker="SYN",
            p50_log_return=0.01,
            experiment_id="T0",
        )
    from dataclasses import replace

    frozen = replace(policy, status="frozen", frozen_at="2024-01-01T00:00:00Z")
    long_d = decide_long_flat(
        policy=frozen,
        ticker="SYN",
        p50_log_return=0.01,
        experiment_id="T0",
    )
    assert long_d.action == "long"
    flat_d = decide_long_flat(
        policy=frozen,
        ticker="SYN",
        p50_log_return=0.0001,
        experiment_id="T0",
    )
    assert flat_d.action == "flat"
    baseline = decide_long_flat(
        policy=frozen,
        ticker="SYN",
        p50_log_return=0.05,
        experiment_id="B1",
    )
    assert baseline.action == "flat"
    assert "challenger" in baseline.reason


def test_entry_exit_costs_hand_calculated(tmp_path: Path) -> None:
    db = tmp_path / "p.db"
    initialize_database(db)
    from dataclasses import replace

    draft = default_underlying_policy()
    draft = replace(
        draft,
        shares_per_entry="10",
        commission_per_share="0.01",
        slippage_bps="0",
        half_spread_bps="0",
        max_notional_per_trade="100000",
    )
    with connection(db) as conn:
        account_id = upsert_paper_account(conn, name="t", starting_cash="10000.00")
        insert_policy(conn, draft)
        policy = freeze_policy(conn, draft.policy_id)
        quote = EquityQuote(
            ticker="SYN",
            bid=Decimal("100"),
            ask=Decimal("101"),
            quote_time="2024-01-02T15:00:00Z",
        )
        decision = decide_long_flat(
            policy=policy,
            ticker="SYN",
            p50_log_return=0.02,
            experiment_id="M0",
            is_synthetic=True,
        )
        insert_decision(
            conn, account_id=account_id, policy_id=policy.policy_id, decision=decision
        )
        entry = build_entry_fill(policy=policy, quote=quote, fill_source="test")
        # 10 * 101 + 10*0.01 = 1010.10
        assert entry.fill_price == Decimal("101")
        assert entry.cash_delta == Decimal("-1010.10")
        bal = record_fill(
            conn,
            account_id=account_id,
            policy=policy,
            ticker="SYN",
            decision_id=decision.decision_id,
            fill=entry,
            is_synthetic=True,
        )
        assert bal == Decimal("8989.9000")
        assert open_share_quantity(conn, account_id, "SYN") == Decimal("10.0000")

        exit_fill = build_exit_fill(
            policy=policy,
            quote=quote,
            shares=Decimal("10"),
            fill_source="test",
        )
        # 10 * 100 - 0.10 = 999.90
        assert exit_fill.cash_delta == Decimal("999.90")
        bal2 = record_fill(
            conn,
            account_id=account_id,
            policy=policy,
            ticker="SYN",
            decision_id=decision.decision_id,
            fill=exit_fill,
            is_synthetic=True,
        )
        assert bal2 == Decimal("9989.8000")
        assert open_share_quantity(conn, account_id, "SYN") == Decimal("0.0000")
        assert latest_cash_balance(conn, account_id) == bal2
