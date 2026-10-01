"""Non-synthetic bars walk-forward issue/settle (Phase WF).

Daily (1d) bars support horizon_hours in {24, 72} only. Synthetic forward-demo
in paper_forward.py is unchanged.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from quantile_ledger.artifacts import register_artifact, write_json_artifact
from quantile_ledger.bars import (
    DAILY_GATE_HORIZONS,
    CloseSeries,
    horizon_hours_to_bar_horizon,
    list_closes_as_of,
    normalize_interval,
)
from quantile_ledger.baselines import issue_b1_empirical
from quantile_ledger.errors import (
    InsufficientDataError,
    MalformedInputError,
    PaperRiskRejectionError,
)
from quantile_ledger.experiments import B1, get_experiment
from quantile_ledger.forecast_store import (
    insert_forecast,
    insert_outcome,
    upsert_experiment,
)
from quantile_ledger.kronos_quantile import (
    KRONOS_BACKEND_FAKE,
    KronosLocalPaths,
    KronosSampler,
    build_kronos_sampler,
    closes_to_ohlc_bars,
    issue_k0_forecast,
)
from quantile_ledger.mamba_quantile import issue_m0_forecast, train_m0_model
from quantile_ledger.mechanical import run_mechanical_underlying
from quantile_ledger.paper import get_policy
from quantile_ledger.runs import close_run, open_run
from quantile_ledger.synthetic import one_step_log_returns
from quantile_ledger.tft_quantile import issue_t0_forecast, train_t0_model
from quantile_ledger.timeutil import to_iso_utc, utc_now

CHALLENGER_IDS = frozenset({"M0", "K0", "T0"})


@dataclass(frozen=True)
class WalkForwardResult:
    ticker: str
    interval: str
    horizon_hours: int
    challenger_experiment_id: str
    run_id: str
    issued: int
    settled: int
    skipped_insufficient: int
    is_synthetic_cohort: bool
    issue_bar_ends: tuple[str, ...]


def _issue_indices(
    *,
    n_bars: int,
    bar_horizon: int,
    lookback: int,
    min_history: int,
    max_issues: int | None,
) -> list[int]:
    """Indices where we can train/issue and still settle within series."""
    # Need lookback one-step returns ⇒ lookback+1 closes; plus bar_horizon for label;
    # plus min_history for B1/M0 sample floors.
    first = max(lookback + bar_horizon, min_history + bar_horizon)
    last = n_bars - bar_horizon - 1
    if last < first:
        msg = (
            f"not enough bars for walk-forward: n={n_bars}, "
            f"need first={first} last>={first}"
        )
        raise InsufficientDataError(msg)
    indices = list(range(first, last + 1))
    if max_issues is not None and max_issues > 0 and len(indices) > max_issues:
        # Evenly subsample chronologically (keep order).
        step = len(indices) / max_issues
        picked = [indices[int(i * step)] for i in range(max_issues)]
        # Deduplicate while preserving order.
        out: list[int] = []
        seen: set[int] = set()
        for i in picked:
            if i not in seen:
                out.append(i)
                seen.add(i)
        return out
    return indices


def _settle_return(
    series: CloseSeries,
    *,
    issue_idx: int,
    bar_horizon: int,
) -> tuple[float, float, str]:
    spot = series.closes[issue_idx]
    outcome_idx = issue_idx + bar_horizon
    outcome_price = series.closes[outcome_idx]
    outcome_bar_at = series.bar_ends[outcome_idx]
    actual_return = math.log(outcome_price / spot)
    return outcome_price, actual_return, outcome_bar_at


def _fit_once_challenger(
    eid: str,
    *,
    series: CloseSeries,
    indices: list[int],
    bar_horizon: int,
    lookback: int,
    train_epochs: int,
    seed: int,
    min_model_samples: int,
) -> tuple[Any | None, str | None]:
    """Train M0/T0 once on the first issuance window that has enough samples.

    Returns ``(model, training_cutoff)`` where ``training_cutoff`` is the bar-end
    timestamp of that first eligible issuance. Returns ``(None, None)`` if no
    window had enough samples (caller raises). PIT holds because the cutoff is
    the first issuance's own timestamp, which precedes every later issuance.
    """
    for issue_idx in indices:
        closes_known = series.closes[: issue_idx + 1]
        cutoff = series.bar_ends[issue_idx]
        model: Any
        try:
            if eid == "M0":
                model, _ = train_m0_model(
                    closes_known,
                    bar_horizon=bar_horizon,
                    lookback=lookback,
                    epochs=train_epochs,
                    seed=seed,
                    min_samples=min_model_samples,
                )
            else:  # T0
                model, _ = train_t0_model(
                    closes_known,
                    bar_horizon=bar_horizon,
                    lookback=lookback,
                    epochs=train_epochs,
                    seed=seed,
                    min_samples=min_model_samples,
                )
        except InsufficientDataError:
            continue
        return model, cutoff
    return None, None


def _persist_fit_once_artifact(
    conn: Any,
    *,
    model: Any,
    experiment_id: str,
    model_dir: Path,
) -> None:
    """Write the fit_once model weights locally and register the artifact.

    Uses relative paths only (no private absolute paths in the DB), mirroring
    the demo's weight-persistence approach.
    """
    kind = f"{experiment_id.lower()}_weights"
    artifact_id, digest, rel = write_json_artifact(
        artifacts_dir=model_dir,
        kind=kind,
        payload=model.to_dict(),
        filename_stem=f"{experiment_id.lower()}-fit-once",
    )
    register_artifact(
        conn,
        artifact_id=artifact_id,
        digest=digest,
        kind=kind,
        relative_path=str(rel),
        experiment_id=experiment_id,
        model_version_id=None,
        metadata={"training_mode": "fit_once"},
    )


def run_bars_walk_forward(
    conn: Any,
    *,
    ticker: str,
    interval: str = "1d",
    horizon_hours: int = 24,
    challenger_experiment_id: str = "M0",
    provider: str | None = None,
    lookback: int = 32,
    train_epochs: int = 12,
    seed: int = 7,
    max_issues: int | None = None,
    min_b1_samples: int = 30,
    min_model_samples: int = 40,
    exclude_synthetic: bool = True,
    kronos_backend: str = KRONOS_BACKEND_FAKE,
    kronos_paths: KronosLocalPaths | None = None,
    training_mode: Literal["per_issuance", "fit_once"] = "per_issuance",
    model_dir: Path | None = None,
) -> WalkForwardResult:
    """
    Issue B1 + one challenger on completed bars and settle from later bars.

    When exclude_synthetic=True (gate path), refuse if the PIT series mixes or
    is entirely synthetic.

    ``kronos_backend`` selects the K0 sampler ("fake" default, offline; "real"
    requires the optional [ml] extra and a fetched local bundle). It is only
    consulted when the challenger is K0. ``kronos_paths`` supplies the local
    weight-cache location for the real backend.

    ``training_mode`` controls how M0/T0 are fit (K0 does not train):

    - ``per_issuance`` (default): retrain from scratch on the expanding window
      before every issuance. Cheap for the local stand-in models; preserves the
      original behavior.
    - ``fit_once``: fit a single model on the window up to the first eligible
      issuance, persist it (when ``model_dir`` is given), and reuse it for every
      later issuance. This is the seam real neural models (e.g. cluster-trained
      Mamba/TFT) plug into. It is point-in-time safe because the single
      ``training_cutoff`` equals the first issuance timestamp and therefore
      precedes every later ``issued_at``.

    ``model_dir`` is the local directory for persisted fit_once weight
    artifacts; when omitted, the fitted model is reused in-memory without being
    written to disk.
    """
    if training_mode not in ("per_issuance", "fit_once"):
        msg = (
            f"training_mode must be 'per_issuance' or 'fit_once', got {training_mode!r}"
        )
        raise MalformedInputError(msg)
    iv = normalize_interval(interval)
    if iv == "1d" and horizon_hours not in DAILY_GATE_HORIZONS:
        msg = (
            f"daily walk-forward requires horizon_hours in "
            f"{sorted(DAILY_GATE_HORIZONS)}, got {horizon_hours}"
        )
        raise MalformedInputError(msg)

    eid = challenger_experiment_id.upper()
    if eid not in CHALLENGER_IDS:
        msg = f"challenger must be one of {sorted(CHALLENGER_IDS)}, got {eid}"
        raise MalformedInputError(msg)

    bar_horizon = horizon_hours_to_bar_horizon(horizon_hours, iv)
    # Load full history (PIT as of far future), then truncate per issuance.
    series = list_closes_as_of(
        conn,
        ticker=ticker,
        interval=iv,
        as_of="9999-12-31T23:59:59Z",
        provider=provider,
    )
    if len(series.closes) < min_b1_samples + bar_horizon + 2:
        msg = (
            f"insufficient bars for {ticker}/{iv}: have {len(series.closes)}, "
            f"need >={min_b1_samples + bar_horizon + 2}"
        )
        raise InsufficientDataError(msg)

    if exclude_synthetic and series.is_synthetic:
        msg = (
            "walk-forward gate refuses synthetic bar cohorts; "
            "re-import with is_synthetic=0 or fetch from Stooq"
        )
        raise MalformedInputError(msg)

    indices = _issue_indices(
        n_bars=len(series.closes),
        bar_horizon=bar_horizon,
        lookback=lookback,
        min_history=max(min_b1_samples, min_model_samples) + lookback,
        max_issues=max_issues,
    )

    # Build the K0 sampler once (real backend loads weights lazily/at most once).
    kronos_sampler: KronosSampler | None = None
    if eid == "K0":
        backend = kronos_backend.strip().lower()
        if backend != KRONOS_BACKEND_FAKE and kronos_paths is None:
            msg = (
                f"K0 backend {kronos_backend!r} requires kronos_paths "
                "(local weight-cache location); none was provided"
            )
            raise MalformedInputError(msg)
        # Fake sampler ignores paths; real needs a cache location from the caller.
        resolved_paths = kronos_paths or KronosLocalPaths(
            weights_dir=Path(), source_dir=Path()
        )
        kronos_sampler = build_kronos_sampler(
            backend,
            paths=resolved_paths,
            lookback=lookback,
        )

    run_id = open_run(
        conn,
        run_type="walk_forward",
        provider=provider or "bars",
    )
    upsert_experiment(conn, B1)
    upsert_experiment(conn, get_experiment(eid))

    # fit_once: pre-train M0/T0 a single time on the window up to the first
    # eligible issuance. The resulting (model, training_cutoff) is reused for
    # every issuance; training_cutoff <= every later issued_at keeps it PIT-safe.
    fitted_model: Any | None = None
    fit_once_cutoff: str | None = None
    if training_mode == "fit_once" and eid in ("M0", "T0"):
        fitted_model, fit_once_cutoff = _fit_once_challenger(
            eid,
            series=series,
            indices=indices,
            bar_horizon=bar_horizon,
            lookback=lookback,
            train_epochs=train_epochs,
            seed=seed,
            min_model_samples=min_model_samples,
        )
        if fitted_model is None or fit_once_cutoff is None:
            msg = (
                f"fit_once could not train {eid}: no issuance window had "
                f">= {min_model_samples} samples"
            )
            raise InsufficientDataError(msg)
        if model_dir is not None:
            _persist_fit_once_artifact(
                conn,
                model=fitted_model,
                experiment_id=eid,
                model_dir=model_dir,
            )

    issued = 0
    settled = 0
    skipped = 0
    issue_ends: list[str] = []

    try:
        for issue_idx in indices:
            closes_known = series.closes[: issue_idx + 1]
            bar_ends_known = series.bar_ends[: issue_idx + 1]
            issued_at = bar_ends_known[-1]
            origin_bar_at = issued_at
            target_at = series.bar_ends[issue_idx + bar_horizon]
            spot = closes_known[-1]
            data_as_of = issued_at
            training_cutoff = issued_at
            rets = one_step_log_returns(closes_known)

            try:
                b1 = issue_b1_empirical(
                    ticker=series.ticker,
                    issued_at=issued_at,
                    origin_bar_at=origin_bar_at,
                    target_at=target_at,
                    spot_at_issue=spot,
                    horizon_hours=horizon_hours,
                    closes_known_by_issue=closes_known,
                    bar_horizon=bar_horizon,
                    training_cutoff=training_cutoff,
                    data_as_of=data_as_of,
                    min_samples=min_b1_samples,
                    is_synthetic=series.is_synthetic,
                    run_id=run_id,
                    random_seed=seed,
                )
                if eid == "M0":
                    m0_model: Any
                    if training_mode == "fit_once":
                        m0_model = fitted_model
                        m0_cutoff = fit_once_cutoff
                    else:
                        m0_model, _ = train_m0_model(
                            closes_known,
                            bar_horizon=bar_horizon,
                            lookback=lookback,
                            epochs=train_epochs,
                            seed=seed,
                            min_samples=min_model_samples,
                        )
                        m0_cutoff = training_cutoff
                    assert m0_cutoff is not None and m0_cutoff <= issued_at
                    ch = issue_m0_forecast(
                        m0_model,
                        ticker=series.ticker,
                        issued_at=issued_at,
                        origin_bar_at=origin_bar_at,
                        target_at=target_at,
                        spot_at_issue=spot,
                        horizon_hours=horizon_hours,
                        recent_one_step_log_returns=rets,
                        training_cutoff=m0_cutoff,
                        data_as_of=data_as_of,
                        is_synthetic=series.is_synthetic,
                        run_id=run_id,
                    )
                elif eid == "T0":
                    t0_model: Any
                    if training_mode == "fit_once":
                        t0_model = fitted_model
                        t0_cutoff = fit_once_cutoff
                    else:
                        t0_model, _ = train_t0_model(
                            closes_known,
                            bar_horizon=bar_horizon,
                            lookback=lookback,
                            epochs=train_epochs,
                            seed=seed,
                            min_samples=min_model_samples,
                        )
                        t0_cutoff = training_cutoff
                    assert t0_cutoff is not None and t0_cutoff <= issued_at
                    ch = issue_t0_forecast(
                        t0_model,
                        ticker=series.ticker,
                        issued_at=issued_at,
                        origin_bar_at=origin_bar_at,
                        target_at=target_at,
                        spot_at_issue=spot,
                        horizon_hours=horizon_hours,
                        recent_one_step_log_returns=rets,
                        training_cutoff=t0_cutoff,
                        data_as_of=data_as_of,
                        is_synthetic=series.is_synthetic,
                        run_id=run_id,
                    )
                else:
                    assert kronos_sampler is not None  # built above for K0
                    ohlc = closes_to_ohlc_bars(closes_known)
                    future_bar_ends = series.bar_ends[
                        issue_idx + 1 : issue_idx + bar_horizon + 1
                    ]
                    ch = issue_k0_forecast(
                        kronos_sampler,
                        bars=ohlc,
                        bar_ends=bar_ends_known,
                        future_bar_ends=future_bar_ends,
                        ticker=series.ticker,
                        issued_at=issued_at,
                        origin_bar_at=origin_bar_at,
                        target_at=target_at,
                        spot_at_issue=spot,
                        horizon_hours=horizon_hours,
                        pred_len=bar_horizon,
                        training_cutoff=training_cutoff,
                        data_as_of=data_as_of,
                        seed=seed,
                        is_synthetic=series.is_synthetic,
                        run_id=run_id,
                    )
            except InsufficientDataError:
                skipped += 1
                continue

            insert_forecast(conn, b1)
            insert_forecast(conn, ch)
            issued += 2
            issue_ends.append(issued_at)

            outcome_price, actual_return, outcome_bar_at = _settle_return(
                series, issue_idx=issue_idx, bar_horizon=bar_horizon
            )
            settled_at = to_iso_utc(utc_now())
            for fc in (b1, ch):
                levels = fc.quantile_levels
                values = fc.quantile_values
                p10 = values[levels.index(0.10)] if 0.10 in levels else None
                p90 = values[levels.index(0.90)] if 0.90 in levels else None
                insert_outcome(
                    conn,
                    forecast_id=fc.forecast_id,
                    outcome_price=outcome_price,
                    actual_return=actual_return,
                    outcome_bar_at=outcome_bar_at,
                    settled_at=settled_at,
                    p10=p10,
                    p90=p90,
                )
                settled += 1
        close_run(
            conn,
            run_id,
            status="succeeded",
            row_counts={
                "issued": issued,
                "settled": settled,
                "skipped_insufficient": skipped,
            },
        )
    except Exception:
        close_run(conn, run_id, status="failed", error_sanitized="walk_forward_failed")
        raise

    return WalkForwardResult(
        ticker=series.ticker,
        interval=iv,
        horizon_hours=horizon_hours,
        challenger_experiment_id=eid,
        run_id=run_id,
        issued=issued,
        settled=settled,
        skipped_insufficient=skipped,
        is_synthetic_cohort=series.is_synthetic,
        issue_bar_ends=tuple(issue_ends),
    )


def run_paper_gate_books(
    conn: Any,
    *,
    challenger_account_id: str,
    challenger_policy_id: str,
    b1_account_id: str,
    b1_policy_id: str,
    challenger_experiment_id: str,
    is_forward: bool = True,
) -> dict[str, Any]:
    """Run mechanical paper for challenger and B1 control on settled forecasts."""
    ch_policy = get_policy(conn, challenger_policy_id)
    b1_policy = get_policy(conn, b1_policy_id)
    ch_policy.require_frozen()
    b1_policy.require_frozen()
    eid = challenger_experiment_id.upper()
    if eid not in ch_policy.challenger_experiment_ids:
        msg = f"{eid} not in challenger policy set"
        raise PaperRiskRejectionError(msg)
    if "B1" not in b1_policy.challenger_experiment_ids:
        msg = "B1 control policy must include B1 in challenger_experiment_ids"
        raise PaperRiskRejectionError(msg)

    ch = run_mechanical_underlying(
        conn,
        account_id=challenger_account_id,
        policy_id=challenger_policy_id,
        experiment_id=eid,
        is_forward=is_forward,
    )
    b1 = run_mechanical_underlying(
        conn,
        account_id=b1_account_id,
        policy_id=b1_policy_id,
        experiment_id="B1",
        is_forward=is_forward,
    )
    return {"challenger": ch, "b1": b1}


__all__ = [
    "CHALLENGER_IDS",
    "WalkForwardResult",
    "run_bars_walk_forward",
    "run_paper_gate_books",
]
