"""Evidence gate report: skill vs B1 and paper P&L delta (Phase WF).

Synthetic forecasts are excluded from the gate cohort. Statuses are
pass / fail / inconclusive — human review still required before Phase I / J2.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Literal

from quantile_ledger.compare import SettledForecast, build_paired_cohort
from quantile_ledger.errors import DatabaseError
from quantile_ledger.paper import latest_cash_balance, list_marks

GateStatus = Literal["pass", "fail", "inconclusive"]


@dataclass(frozen=True)
class GateReport:
    status: GateStatus
    reasons: tuple[str, ...]
    challenger_experiment_id: str
    paired_n: int
    skill_vs_b1: float | None
    coverage: float | None
    mean_width: float | None
    approx_crps: float | None
    pinball: float | None
    min_metric_samples: int
    synthetic_excluded: int
    challenger_equity_mid: str | None
    b1_equity_mid: str | None
    equity_delta_mid: str | None
    challenger_fills: int
    b1_fills: int
    checklist: tuple[str, ...]


def _load_settled_non_synthetic(
    conn: Any,
    *,
    experiment_ids: tuple[str, ...],
    ticker: str | None = None,
) -> tuple[list[SettledForecast], int]:
    """Load settled forecasts with is_synthetic=0 only."""
    params: list[Any] = list(experiment_ids)
    exp_placeholders = ",".join("?" for _ in experiment_ids)
    ticker_clause = ""
    if ticker is not None:
        ticker_clause = " AND f.ticker = ?"
        params.append(ticker.upper())
    try:
        rows = conn.execute(
            f"""
            SELECT f.forecast_id, f.experiment_id, f.ticker, f.issued_at,
                   f.origin_bar_at, f.target_at, f.horizon_hours, f.spot_at_issue,
                   f.price_type, f.variant, f.regime, f.parent_forecast_id,
                   f.is_synthetic, o.actual_return
            FROM forecasts f
            JOIN outcomes o ON o.forecast_id = f.forecast_id
            WHERE f.experiment_id IN ({exp_placeholders})
              AND o.actual_return IS NOT NULL
              {ticker_clause}
            ORDER BY f.issued_at, f.forecast_id
            """,
            params,
        ).fetchall()
    except Exception as exc:
        msg = f"gate forecast load failed: {exc}"
        raise DatabaseError(msg) from exc

    excluded = 0
    out: list[SettledForecast] = []
    for row in rows:
        if bool(row["is_synthetic"]):
            excluded += 1
            continue
        qrows = conn.execute(
            """
            SELECT q, return_value FROM forecast_quantiles
            WHERE forecast_id = ? ORDER BY q
            """,
            (row["forecast_id"],),
        ).fetchall()
        out.append(
            SettledForecast(
                forecast_id=str(row["forecast_id"]),
                experiment_id=str(row["experiment_id"]),
                ticker=str(row["ticker"]),
                issued_at=str(row["issued_at"]),
                origin_bar_at=str(row["origin_bar_at"]),
                target_at=str(row["target_at"]),
                horizon_hours=int(row["horizon_hours"]),
                spot_at_issue=float(row["spot_at_issue"]),
                quantile_levels=tuple(float(q["q"]) for q in qrows),
                quantile_values=tuple(float(q["return_value"]) for q in qrows),
                actual_return=float(row["actual_return"]),
                price_type=str(row["price_type"] or "adjusted_research"),
                variant=str(row["variant"] or "raw"),
                regime=row["regime"],
                parent_forecast_id=row["parent_forecast_id"],
            )
        )
    return out, excluded


def _fill_count(conn: Any, account_id: str) -> int:
    row = conn.execute(
        "SELECT COUNT(*) AS n FROM paper_fills WHERE account_id = ?",
        (account_id,),
    ).fetchone()
    return int(row["n"]) if row else 0


def _equity_mid(conn: Any, account_id: str | None) -> Decimal | None:
    if account_id is None:
        return None
    marks = list_marks(conn, account_id)
    if marks:
        return Decimal(marks[-1].equity_mid)
    try:
        cash = latest_cash_balance(conn, account_id)
        return cash
    except DatabaseError:
        return None


def evaluate_gate(
    conn: Any,
    *,
    challenger_experiment_id: str,
    min_metric_samples: int = 30,
    ticker: str | None = None,
    challenger_account_id: str | None = None,
    b1_account_id: str | None = None,
    require_positive_skill: bool = True,
    require_positive_paper_delta: bool = True,
) -> GateReport:
    """
    Compute gate status from non-synthetic paired cohort + optional paper books.

    inconclusive: N < min_metric_samples or missing paper books when required.
    pass: skill > 0 (if required) AND paper equity_mid delta > 0 (if required).
    fail: adequate N but skill or paper delta fails.
    """
    eid = challenger_experiment_id.upper()
    forecasts, synthetic_excluded = _load_settled_non_synthetic(
        conn,
        experiment_ids=("B1", eid),
        ticker=ticker,
    )
    paired = build_paired_cohort(forecasts, experiment_ids=["B1", eid])
    metrics_by_exp = paired.metrics_by_experiment

    n = len(paired.cohort)
    ch_m = metrics_by_exp.get(eid, {})
    skill_raw = ch_m.get("skill_vs_B1")
    skill = float(skill_raw) if isinstance(skill_raw, (int, float)) else None
    coverage_raw = ch_m.get("coverage_p10_p90")
    coverage = float(coverage_raw) if isinstance(coverage_raw, (int, float)) else None
    width_raw = ch_m.get("mean_width_return")
    width = float(width_raw) if isinstance(width_raw, (int, float)) else None
    crps_raw = ch_m.get("approx_crps")
    crps = float(crps_raw) if isinstance(crps_raw, (int, float)) else None
    pinball_raw = ch_m.get("mean_pinball")
    pinball = float(pinball_raw) if isinstance(pinball_raw, (int, float)) else None

    ch_eq = _equity_mid(conn, challenger_account_id)
    b1_eq = _equity_mid(conn, b1_account_id)
    delta: Decimal | None = None
    if ch_eq is not None and b1_eq is not None:
        delta = ch_eq - b1_eq

    ch_fills = _fill_count(conn, challenger_account_id) if challenger_account_id else 0
    b1_fills = _fill_count(conn, b1_account_id) if b1_account_id else 0

    reasons: list[str] = []
    status: GateStatus

    if n < min_metric_samples:
        status = "inconclusive"
        reasons.append(f"paired_n={n} < min_metric_samples={min_metric_samples}")
    else:
        skill_ok = True
        if require_positive_skill:
            if skill is None:
                skill_ok = False
                reasons.append("skill_vs_B1 missing")
            elif skill <= 0:
                skill_ok = False
                reasons.append(f"skill_vs_B1={skill:.6f} <= 0")
            else:
                reasons.append(f"skill_vs_B1={skill:.6f} > 0")

        paper_ok = True
        if require_positive_paper_delta:
            if delta is None:
                paper_ok = False
                reasons.append(
                    "paper equity delta unavailable "
                    "(provide --challenger-account and --b1-account)"
                )
            elif delta <= 0:
                paper_ok = False
                reasons.append(f"equity_delta_mid={delta} <= 0")
            else:
                reasons.append(f"equity_delta_mid={delta} > 0")

        status = "pass" if skill_ok and paper_ok else "fail"

    checklist = (
        "Phase I (E0): only after non-inconclusive gate with positive skill vs B1 "
        "and adequate N (human review).",
        "Phase J2 (options): only after underlying paper equity_delta_mid > 0 "
        "after costs vs B1 control with adequate N (human review).",
        "Negative or inconclusive results are valid research outcomes — do not "
        "force unlock.",
        "Synthetic demo rows are excluded from this cohort "
        f"(excluded={synthetic_excluded}).",
    )

    return GateReport(
        status=status,
        reasons=tuple(reasons),
        challenger_experiment_id=eid,
        paired_n=n,
        skill_vs_b1=skill,
        coverage=coverage,
        mean_width=width,
        approx_crps=crps,
        pinball=pinball,
        min_metric_samples=min_metric_samples,
        synthetic_excluded=synthetic_excluded,
        challenger_equity_mid=str(ch_eq) if ch_eq is not None else None,
        b1_equity_mid=str(b1_eq) if b1_eq is not None else None,
        equity_delta_mid=str(delta) if delta is not None else None,
        challenger_fills=ch_fills,
        b1_fills=b1_fills,
        checklist=checklist,
    )


__all__ = [
    "GateReport",
    "GateStatus",
    "evaluate_gate",
]
