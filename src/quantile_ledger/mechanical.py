"""Mechanical underlying long/flat paper runner (no broker, no live orders)."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from quantile_ledger.errors import DatabaseError, PaperRiskRejectionError
from quantile_ledger.paper import (
    EquityQuote,
    build_entry_fill,
    build_exit_fill,
    decide_long_flat,
    get_policy,
    insert_decision,
    open_share_quantity,
    record_fill,
)


@dataclass(frozen=True)
class MechanicalRunResult:
    decisions: int
    entries: int
    exits: int
    flats_recorded: int
    skipped_already_decided: int
    cash_after: str | None


def quote_from_spot(
    *,
    ticker: str,
    spot: float,
    half_spread_bps: str,
    quote_time: str,
    quality: str = "ok",
) -> EquityQuote:
    """Build a conservative bid/ask around spot using half-spread bps."""
    mid = Decimal(str(spot))
    if mid <= 0:
        msg = "spot must be positive for paper quote"
        raise PaperRiskRejectionError(msg)
    half = Decimal(str(half_spread_bps)) / Decimal("10000")
    bid = mid * (Decimal("1") - half)
    ask = mid * (Decimal("1") + half)
    if bid <= 0 or ask < bid:
        msg = "constructed equity quote invalid"
        raise PaperRiskRejectionError(msg)
    return EquityQuote(
        ticker=ticker.upper(),
        bid=bid,
        ask=ask,
        quote_time=quote_time,
        quality=quality,
    )


def _p50_from_quantiles(levels: list[float], values: list[float]) -> float:
    if 0.5 in levels:
        return float(values[levels.index(0.5)])
    # Nearest level to 0.5 if exact median missing.
    idx = min(range(len(levels)), key=lambda i: abs(levels[i] - 0.5))
    return float(values[idx])


def list_undecided_settled_challenger_forecasts(
    conn: Any,
    *,
    account_id: str,
    policy_id: str,
    experiment_id: str,
) -> list[dict[str, Any]]:
    """Settled forecasts for one challenger with no prior decision on this book."""
    rows = conn.execute(
        """
        SELECT f.forecast_id, f.experiment_id, f.ticker, f.issued_at,
               f.spot_at_issue, f.is_synthetic
        FROM forecasts f
        JOIN outcomes o ON o.forecast_id = f.forecast_id
        WHERE f.experiment_id = ?
          AND o.actual_return IS NOT NULL
          AND NOT EXISTS (
            SELECT 1 FROM paper_decisions d
            WHERE d.account_id = ?
              AND d.policy_id = ?
              AND d.forecast_id = f.forecast_id
          )
        ORDER BY f.issued_at, f.forecast_id
        """,
        (experiment_id, account_id, policy_id),
    ).fetchall()
    out: list[dict[str, Any]] = []
    for row in rows:
        qrows = conn.execute(
            """
            SELECT q, return_value FROM forecast_quantiles
            WHERE forecast_id = ? ORDER BY q
            """,
            (row["forecast_id"],),
        ).fetchall()
        levels = [float(q["q"]) for q in qrows]
        values = [float(q["return_value"]) for q in qrows]
        if not levels:
            continue
        out.append(
            {
                "forecast_id": row["forecast_id"],
                "experiment_id": row["experiment_id"],
                "ticker": row["ticker"],
                "issued_at": row["issued_at"],
                "spot_at_issue": float(row["spot_at_issue"]),
                "is_synthetic": bool(row["is_synthetic"]),
                "quantile_levels": levels,
                "quantile_values": values,
                "p50": _p50_from_quantiles(levels, values),
            }
        )
    return out


def run_mechanical_underlying(
    conn: Any,
    *,
    account_id: str,
    policy_id: str,
    experiment_id: str,
    is_forward: bool = True,
) -> MechanicalRunResult:
    """
    For each undecided settled challenger forecast: decide long/flat, then
    enter/exit with ask/bid costs under the frozen policy.
    """
    policy = get_policy(conn, policy_id)
    policy.require_frozen()
    if experiment_id not in policy.challenger_experiment_ids:
        msg = (
            f"experiment {experiment_id} not in policy challenger set "
            f"{policy.challenger_experiment_ids}"
        )
        raise PaperRiskRejectionError(msg)

    acct = conn.execute(
        "SELECT account_id FROM paper_accounts WHERE account_id = ?",
        (account_id,),
    ).fetchone()
    if acct is None:
        msg = f"unknown paper account_id: {account_id}"
        raise DatabaseError(msg)

    pending = list_undecided_settled_challenger_forecasts(
        conn,
        account_id=account_id,
        policy_id=policy_id,
        experiment_id=experiment_id,
    )
    decisions = 0
    entries = 0
    exits = 0
    flats = 0
    cash_after: str | None = None

    for item in pending:
        ticker = str(item["ticker"]).upper()
        decision = decide_long_flat(
            policy=policy,
            ticker=ticker,
            p50_log_return=float(item["p50"]),
            forecast_id=str(item["forecast_id"]),
            experiment_id=experiment_id,
            decided_at=str(item["issued_at"]),
            is_forward=is_forward,
            is_synthetic=bool(item["is_synthetic"]),
        )
        insert_decision(
            conn,
            account_id=account_id,
            policy_id=policy.policy_id,
            decision=decision,
        )
        decisions += 1
        open_qty = open_share_quantity(conn, account_id, ticker)
        quote = quote_from_spot(
            ticker=ticker,
            spot=float(item["spot_at_issue"]),
            half_spread_bps=policy.half_spread_bps,
            quote_time=str(item["issued_at"]),
            quality="synthetic_from_spot" if item["is_synthetic"] else "spot_derived",
        )

        if decision.action == "long" and open_qty == 0:
            fill = build_entry_fill(
                policy=policy,
                quote=quote,
                fill_source="mechanical",
                fill_time=str(item["issued_at"]),
            )
            bal = record_fill(
                conn,
                account_id=account_id,
                policy=policy,
                ticker=ticker,
                decision_id=decision.decision_id,
                fill=fill,
                is_forward=is_forward,
                is_synthetic=bool(item["is_synthetic"]),
            )
            entries += 1
            cash_after = str(bal)
        elif decision.action == "flat" and open_qty > 0:
            fill = build_exit_fill(
                policy=policy,
                quote=quote,
                shares=open_qty,
                fill_source="mechanical",
                fill_time=str(item["issued_at"]),
            )
            bal = record_fill(
                conn,
                account_id=account_id,
                policy=policy,
                ticker=ticker,
                decision_id=decision.decision_id,
                fill=fill,
                is_forward=is_forward,
                is_synthetic=bool(item["is_synthetic"]),
            )
            exits += 1
            cash_after = str(bal)
        else:
            flats += 1

    return MechanicalRunResult(
        decisions=decisions,
        entries=entries,
        exits=exits,
        flats_recorded=flats,
        skipped_already_decided=0,
        cash_after=cash_after,
    )


__all__ = [
    "MechanicalRunResult",
    "list_undecided_settled_challenger_forecasts",
    "quote_from_spot",
    "run_mechanical_underlying",
]
