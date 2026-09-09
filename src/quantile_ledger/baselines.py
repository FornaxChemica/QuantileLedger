"""B0 persistence and B1 rolling empirical probabilistic baselines."""

from __future__ import annotations

import math
import uuid
from collections.abc import Sequence
from dataclasses import dataclass

from quantile_ledger.contract import (
    DEFAULT_QUANTILES,
    FEATURE_SET_MARKET_ONLY,
    FEATURE_VERSION_V1,
    FORECAST_SPACE_RETURN,
    TARGET_LOG_RETURN,
    ForecastContract,
    validate_forecast_contract,
)
from quantile_ledger.errors import InsufficientDataError
from quantile_ledger.experiments import B0, B1
from quantile_ledger.timeutil import to_iso_utc, utc_now


@dataclass(frozen=True)
class Bar:
    """Completed OHLCV bar (close used for returns)."""

    bar_end: str  # ISO UTC
    close: float


def log_returns_for_horizon(closes: Sequence[float], horizon_bars: int) -> list[float]:
    if horizon_bars < 1:
        msg = "horizon_bars must be >= 1"
        raise ValueError(msg)
    out: list[float] = []
    for i in range(len(closes) - horizon_bars):
        left = closes[i]
        right = closes[i + horizon_bars]
        if left <= 0 or right <= 0:
            continue
        out.append(math.log(right / left))
    return out


def empirical_quantiles(
    samples: Sequence[float], levels: Sequence[float]
) -> list[float]:
    if not samples:
        msg = "no samples for empirical quantiles"
        raise InsufficientDataError(msg)
    ordered = sorted(samples)
    n = len(ordered)
    values: list[float] = []
    for q in levels:
        if q <= 0 or q >= 1:
            msg = f"invalid quantile {q}"
            raise ValueError(msg)
        # Hyndman-Fan type 7-ish position.
        pos = 1 + (n - 1) * q
        lo = math.floor(pos) - 1
        hi = math.ceil(pos) - 1
        lo = max(0, min(n - 1, lo))
        hi = max(0, min(n - 1, hi))
        if lo == hi:
            values.append(ordered[lo])
        else:
            w = pos - math.floor(pos)
            values.append((1 - w) * ordered[lo] + w * ordered[hi])
    return values


def issue_b0_persistence(
    *,
    ticker: str,
    issued_at: str,
    origin_bar_at: str,
    target_at: str,
    spot_at_issue: float,
    horizon_hours: int,
    training_cutoff: str,
    data_as_of: str,
    quantiles: Sequence[float] = DEFAULT_QUANTILES,
    residual_samples: Sequence[float] | None = None,
    is_synthetic: bool = False,
    run_id: str | None = None,
    random_seed: int | None = None,
) -> ForecastContract:
    """
    B0: median log-return is 0 (price stays at spot).

    If residual_samples are provided (known by issuance), attach that residual
    distribution around zero; otherwise only P50 is scientifically supported and
    other quantiles are set equal to 0 with metadata flag point_only.
    """
    levels = list(quantiles)
    point_only = residual_samples is None or len(residual_samples) < 2
    if point_only:
        values = [0.0 for _ in levels]
        meta = {"distribution": "none", "point_only_median": True}
    else:
        assert residual_samples is not None
        samples = list(residual_samples)
        resid = empirical_quantiles(samples, levels)
        values = list(resid)
        meta = {
            "distribution": "historical_residual",
            "point_only_median": False,
            "residual_n": len(samples),
        }
    created = to_iso_utc(utc_now())
    contract = ForecastContract(
        forecast_id=str(uuid.uuid4()),
        run_id=run_id,
        experiment_id=B0.experiment_id,
        ticker=ticker,
        model_family=B0.model_family,
        model_version=f"{B0.experiment_id}-v{B0.version}",
        artifact_digest=B0.config_hash(),
        feature_set=FEATURE_SET_MARKET_ONLY,
        feature_version=FEATURE_VERSION_V1,
        training_cutoff=training_cutoff,
        data_as_of=data_as_of,
        maximum_feature_timestamp=data_as_of,
        issued_at=issued_at,
        origin_bar_at=origin_bar_at,
        target_at=target_at,
        nominal_horizon_hours=horizon_hours,
        spot_at_issue=spot_at_issue,
        target_definition=TARGET_LOG_RETURN,
        quantile_levels=levels,
        quantile_values=values,
        forecast_space=FORECAST_SPACE_RETURN,
        random_seed=random_seed,
        created_at=created,
        is_synthetic=is_synthetic,
        generation_metadata={"ticker": ticker, **meta},
    )
    return validate_forecast_contract(contract)


def issue_b1_empirical(
    *,
    ticker: str,
    issued_at: str,
    origin_bar_at: str,
    target_at: str,
    spot_at_issue: float,
    horizon_hours: int,
    closes_known_by_issue: Sequence[float],
    bar_horizon: int,
    training_cutoff: str,
    data_as_of: str,
    window: int = 200,
    min_samples: int = 30,
    quantiles: Sequence[float] = DEFAULT_QUANTILES,
    is_synthetic: bool = False,
    run_id: str | None = None,
    random_seed: int | None = None,
) -> ForecastContract:
    """B1: empirical quantiles of trailing log returns known by issuance."""
    returns = log_returns_for_horizon(closes_known_by_issue, bar_horizon)
    # Only returns whose outcome bar is within the known close series.
    if len(returns) > window:
        returns = returns[-window:]
    if len(returns) < min_samples:
        msg = (
            f"B1 insufficient history for {ticker}: have {len(returns)}, "
            f"need {min_samples}"
        )
        raise InsufficientDataError(msg)
    levels = list(quantiles)
    values = empirical_quantiles(returns, levels)
    created = to_iso_utc(utc_now())
    contract = ForecastContract(
        forecast_id=str(uuid.uuid4()),
        run_id=run_id,
        experiment_id=B1.experiment_id,
        ticker=ticker,
        model_family=B1.model_family,
        model_version=f"{B1.experiment_id}-v{B1.version}",
        artifact_digest=B1.config_hash(),
        feature_set=FEATURE_SET_MARKET_ONLY,
        feature_version=FEATURE_VERSION_V1,
        training_cutoff=training_cutoff,
        data_as_of=data_as_of,
        maximum_feature_timestamp=data_as_of,
        issued_at=issued_at,
        origin_bar_at=origin_bar_at,
        target_at=target_at,
        nominal_horizon_hours=horizon_hours,
        spot_at_issue=spot_at_issue,
        target_definition=TARGET_LOG_RETURN,
        quantile_levels=levels,
        quantile_values=values,
        forecast_space=FORECAST_SPACE_RETURN,
        random_seed=random_seed,
        created_at=created,
        is_synthetic=is_synthetic,
        generation_metadata={
            "ticker": ticker,
            "window": window,
            "min_samples": min_samples,
            "effective_n": len(returns),
            "bar_horizon": bar_horizon,
            "history_start_index": max(
                0, len(closes_known_by_issue) - window - bar_horizon
            ),
            "history_end_index": len(closes_known_by_issue) - 1,
        },
    )
    return validate_forecast_contract(contract)
