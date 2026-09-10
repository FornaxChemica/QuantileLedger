"""Synthetic multi-day forward paper book (labeled; no live edge claimed)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from quantile_ledger.errors import DatabaseError, PaperRiskRejectionError
from quantile_ledger.experiments import get_experiment
from quantile_ledger.forecast_store import (
    insert_forecast,
    insert_outcome,
    upsert_experiment,
)
from quantile_ledger.kronos_quantile import (
    FakeKronosSampler,
    closes_to_ohlc_bars,
    issue_k0_forecast,
)
from quantile_ledger.mamba_quantile import issue_m0_forecast, train_m0_model
from quantile_ledger.mechanical import quote_from_spot, run_mechanical_underlying
from quantile_ledger.paper import (
    EquityQuote,
    PaperEquitySummary,
    PaperMark,
    compute_mark_equity,
    get_policy,
    insert_mark,
    latest_cash_balance,
    list_marks,
    open_share_quantity,
    paper_equity_summary,
)
from quantile_ledger.synthetic import make_synthetic_hourly_closes, one_step_log_returns
from quantile_ledger.tft_quantile import issue_t0_forecast, train_t0_model


@dataclass(frozen=True)
class ForwardBookResult:
    experiment_id: str
    forecasts: int
    decisions: int
    entries: int
    exits: int
    flats_recorded: int
    marks: int
    distinct_issue_days: int
    summary: PaperEquitySummary


def _issue_indices(
    *,
    n_bars: int,
    bar_horizon: int,
    lookback: int,
    steps: int,
    min_samples: int = 40,
) -> list[int]:
    """Chronological issue indices that leave room for settlement and lookback."""
    # Need enough completed bars for lookback windows + horizon labels + min_samples.
    first = lookback + bar_horizon + min_samples
    last = n_bars - bar_horizon - 1
    if last <= first:
        msg = "synthetic series too short for forward book"
        raise InsufficientForwardDataError(msg)
    span = last - first
    if steps < 2:
        msg = "forward book requires steps >= 2"
        raise ValueError(msg)
    if steps == 2:
        return [first, last]
    indices = [first + round(i * span / (steps - 1)) for i in range(steps)]
    out: list[int] = []
    for idx in indices:
        idx = int(idx)
        if not out or idx > out[-1]:
            out.append(idx)
    while len(out) < steps and out[-1] + 1 <= last:
        out.append(out[-1] + 1)
    return out[:steps]


class InsufficientForwardDataError(DatabaseError):
    """Raised when the synthetic path cannot support the requested forward steps."""


def _positions_as_of(conn: Any, account_id: str, as_of: str) -> dict[str, Decimal]:
    rows = conn.execute(
        """
        SELECT ticker, side, quantity FROM paper_fills
        WHERE account_id = ? AND fill_time <= ?
        ORDER BY fill_time, rowid
        """,
        (account_id, as_of),
    ).fetchall()
    qty_by: dict[str, Decimal] = {}
    for row in rows:
        ticker = str(row["ticker"]).upper()
        q = Decimal(str(row["quantity"]))
        cur = qty_by.get(ticker, Decimal("0"))
        if row["side"] == "buy":
            cur += q
        else:
            cur -= q
        qty_by[ticker] = cur
    return {t: q for t, q in qty_by.items() if q > 0}


def _cash_as_of(conn: Any, account_id: str, as_of: str) -> Decimal:
    """Reconstruct cash from starting balance + fills with fill_time <= as_of."""
    acct = conn.execute(
        "SELECT starting_cash FROM paper_accounts WHERE account_id = ?",
        (account_id,),
    ).fetchone()
    if acct is None:
        msg = f"unknown paper account_id: {account_id}"
        raise DatabaseError(msg)
    cash = Decimal(str(acct["starting_cash"]))
    rows = conn.execute(
        """
        SELECT cash_delta FROM paper_fills
        WHERE account_id = ? AND fill_time <= ?
        ORDER BY fill_time, rowid
        """,
        (account_id, as_of),
    ).fetchall()
    for row in rows:
        cash += Decimal(str(row["cash_delta"]))
    return cash


def mark_curve_after_mechanical(
    conn: Any,
    *,
    account_id: str,
    policy_half_spread_bps: str,
    forecast_spots: dict[str, tuple[float, str]],
    final_spot: float | None,
    final_marked_at: str | None,
    is_forward: bool = True,
    is_synthetic: bool = True,
) -> list[PaperMark]:
    """Insert mid/bid marks at each decision time (and optional final bar)."""
    decisions = conn.execute(
        """
        SELECT decision_id, forecast_id, ticker, decided_at
        FROM paper_decisions
        WHERE account_id = ?
        ORDER BY decided_at ASC, rowid ASC
        """,
        (account_id,),
    ).fetchall()
    marks: list[PaperMark] = []
    for row in decisions:
        forecast_id = row["forecast_id"]
        if forecast_id is None or forecast_id not in forecast_spots:
            continue
        spot, _issued = forecast_spots[forecast_id]
        marked_at = str(row["decided_at"])
        cash = _cash_as_of(conn, account_id, marked_at)
        positions = _positions_as_of(conn, account_id, marked_at)
        quotes: dict[str, EquityQuote] = {}
        for ticker in positions:
            quotes[ticker] = quote_from_spot(
                ticker=ticker,
                spot=spot,
                half_spread_bps=policy_half_spread_bps,
                quote_time=marked_at,
                quality="synthetic_from_spot",
            )
        # Single-ticker SYN book; multi-ticker needs per-ticker spots.
        mark = compute_mark_equity(cash, positions, quotes)
        marks.append(
            insert_mark(
                conn,
                account_id=account_id,
                marked_at=marked_at,
                mark=mark,
                quote_source="forward_decision_spot",
                data_quality="synthetic_from_spot",
                is_forward=is_forward,
                is_synthetic=is_synthetic,
                note=f"decision:{row['decision_id']}",
            )
        )

    if final_spot is not None and final_marked_at is not None:
        cash = latest_cash_balance(conn, account_id)
        positions = {
            t: open_share_quantity(conn, account_id, t)
            for t in {
                r["ticker"]
                for r in conn.execute(
                    "SELECT DISTINCT ticker FROM paper_fills WHERE account_id = ?",
                    (account_id,),
                ).fetchall()
            }
            if open_share_quantity(conn, account_id, t) > 0
        }
        quotes = {
            ticker: quote_from_spot(
                ticker=ticker,
                spot=final_spot,
                half_spread_bps=policy_half_spread_bps,
                quote_time=final_marked_at,
                quality="synthetic_from_spot",
            )
            for ticker in positions
        }
        mark = compute_mark_equity(cash, positions, quotes)
        marks.append(
            insert_mark(
                conn,
                account_id=account_id,
                marked_at=final_marked_at,
                mark=mark,
                quote_source="forward_final_spot",
                data_quality="synthetic_from_spot",
                is_forward=is_forward,
                is_synthetic=is_synthetic,
                note="final_bar",
            )
        )
    return marks


def run_synthetic_forward_book(
    conn: Any,
    *,
    account_id: str,
    policy_id: str,
    experiment_id: str = "M0",
    steps: int = 8,
    bar_horizon: int = 6,
    lookback: int = 32,
    seed: int = 11,
    series_n: int = 320,
    train_epochs: int = 8,
    is_forward: bool = True,
) -> ForwardBookResult:
    """
    Issue + settle a multi-day synthetic challenger path, run mechanical long/flat,
    then store mid vs bid marks. All rows are labeled is_synthetic=1.
    """
    eid = experiment_id.upper()
    if eid not in {"M0", "K0", "T0"}:
        msg = f"forward book supports M0/K0/T0 only, got {experiment_id}"
        raise PaperRiskRejectionError(msg)

    policy = get_policy(conn, policy_id)
    policy.require_frozen()
    if eid not in policy.challenger_experiment_ids:
        msg = f"experiment {eid} not in policy challenger set"
        raise PaperRiskRejectionError(msg)

    acct = conn.execute(
        "SELECT account_id FROM paper_accounts WHERE account_id = ?",
        (account_id,),
    ).fetchone()
    if acct is None:
        msg = f"unknown paper account_id: {account_id}"
        raise DatabaseError(msg)

    # Idempotency: refuse if this account already has forward marks from a prior demo.
    existing_marks = list_marks(conn, account_id)
    if existing_marks:
        msg = (
            "account already has paper marks; use a fresh account for forward-demo "
            "or clear marks manually"
        )
        raise PaperRiskRejectionError(msg)

    series = make_synthetic_hourly_closes(n=series_n, seed=seed)
    indices = _issue_indices(
        n_bars=len(series.closes),
        bar_horizon=bar_horizon,
        lookback=lookback,
        steps=steps,
    )
    upsert_experiment(conn, get_experiment(eid))

    forecast_spots: dict[str, tuple[float, str]] = {}
    issue_days: set[str] = set()

    for issue_idx in indices:
        closes_known = series.closes[: issue_idx + 1]
        issued_at = series.bar_ends[issue_idx]
        target_at = series.bar_ends[issue_idx + bar_horizon]
        spot = closes_known[-1]
        outcome_price = series.closes[issue_idx + bar_horizon]
        actual_return = math.log(outcome_price / spot)
        issue_days.add(issued_at[:10])
        rets = one_step_log_returns(closes_known)

        if eid == "M0":
            m0_model, _ = train_m0_model(
                closes_known,
                bar_horizon=bar_horizon,
                lookback=lookback,
                epochs=train_epochs,
                seed=seed,
                min_samples=40,
            )
            fc = issue_m0_forecast(
                m0_model,
                ticker=series.ticker,
                issued_at=issued_at,
                origin_bar_at=issued_at,
                target_at=target_at,
                spot_at_issue=spot,
                horizon_hours=bar_horizon,
                recent_one_step_log_returns=rets,
                training_cutoff=issued_at,
                data_as_of=issued_at,
                is_synthetic=True,
            )
        elif eid == "T0":
            t0_model, _ = train_t0_model(
                closes_known,
                bar_horizon=bar_horizon,
                lookback=lookback,
                epochs=train_epochs,
                seed=seed,
                min_samples=40,
            )
            fc = issue_t0_forecast(
                t0_model,
                ticker=series.ticker,
                issued_at=issued_at,
                origin_bar_at=issued_at,
                target_at=target_at,
                spot_at_issue=spot,
                horizon_hours=bar_horizon,
                recent_one_step_log_returns=rets,
                training_cutoff=issued_at,
                data_as_of=issued_at,
                is_synthetic=True,
            )
        else:
            sampler = FakeKronosSampler(lookback=lookback)
            ohlc = closes_to_ohlc_bars(closes_known)
            future_ends = series.bar_ends[issue_idx + 1 : issue_idx + bar_horizon + 1]
            fc = issue_k0_forecast(
                sampler,
                bars=ohlc,
                bar_ends=series.bar_ends[: issue_idx + 1],
                future_bar_ends=future_ends,
                ticker=series.ticker,
                issued_at=issued_at,
                origin_bar_at=issued_at,
                target_at=target_at,
                spot_at_issue=spot,
                horizon_hours=bar_horizon,
                pred_len=bar_horizon,
                training_cutoff=issued_at,
                data_as_of=issued_at,
                seed=seed,
                is_synthetic=True,
            )

        insert_forecast(conn, fc)
        qmap = dict(zip(fc.quantile_levels, fc.quantile_values, strict=True))
        insert_outcome(
            conn,
            forecast_id=fc.forecast_id,
            outcome_price=outcome_price,
            actual_return=actual_return,
            outcome_bar_at=target_at,
            settled_at=target_at,
            p10=qmap[0.10],
            p90=qmap[0.90],
        )
        forecast_spots[fc.forecast_id] = (spot, issued_at)

    mech = run_mechanical_underlying(
        conn,
        account_id=account_id,
        policy_id=policy_id,
        experiment_id=eid,
        is_forward=is_forward,
    )

    last_idx = indices[-1] + bar_horizon
    final_spot = series.closes[last_idx]
    final_at = series.bar_ends[last_idx]
    marks = mark_curve_after_mechanical(
        conn,
        account_id=account_id,
        policy_half_spread_bps=policy.half_spread_bps,
        forecast_spots=forecast_spots,
        final_spot=final_spot,
        final_marked_at=final_at,
        is_forward=is_forward,
        is_synthetic=True,
    )

    summary = paper_equity_summary(conn, account_id)
    return ForwardBookResult(
        experiment_id=eid,
        forecasts=len(forecast_spots),
        decisions=mech.decisions,
        entries=mech.entries,
        exits=mech.exits,
        flats_recorded=mech.flats_recorded,
        marks=len(marks),
        distinct_issue_days=len(issue_days),
        summary=summary,
    )


__all__ = [
    "ForwardBookResult",
    "InsufficientForwardDataError",
    "mark_curve_after_mechanical",
    "run_synthetic_forward_book",
]
