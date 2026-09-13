"""Phase G controlled news ablations (T1/T2/M1/N0).

Market-only parents (T0/M0) stay untouched. News-aware experiments are separate
IDs that refuse issuance when PIT news context is missing/partial (never impute
neutral). Adjustments are documented frozen additive return shifts + isotonic.
"""

from __future__ import annotations

import math
import uuid
from collections.abc import Sequence
from typing import Any

from quantile_ledger.contract import (
    DEFAULT_QUANTILES,
    FEATURE_SET_NEWS_ONLY,
    FORECAST_SPACE_RETURN,
    TARGET_LOG_RETURN,
    ForecastContract,
    validate_forecast_contract,
)
from quantile_ledger.errors import ForecastValidationError, InsufficientDataError
from quantile_ledger.experiments import M1, N0, T1, T2, ExperimentSpec
from quantile_ledger.sentiment import (
    SCORER_FAKE,
    SentimentContext,
    build_sentiment_context,
)
from quantile_ledger.timeutil import to_iso_utc, utc_now


def _isotonic_increasing(values: Sequence[float]) -> list[float]:
    """Pool-adjacent-violators for non-decreasing projection."""
    n = len(values)
    out = list(values)
    i = 0
    while i < n - 1:
        if out[i] <= out[i + 1] + 1e-15:
            i += 1
            continue
        j = i
        while j >= 0 and out[j] > out[j + 1]:
            avg = 0.5 * (out[j] + out[j + 1])
            out[j] = avg
            out[j + 1] = avg
            j -= 1
        i = max(i - 1, 0)
    return out


def news_volume_feature(n_items: int) -> float:
    """log1p volume — stable, always defined when news exists."""
    return math.log1p(max(0, n_items))


def news_polarity(ctx: SentimentContext) -> float:
    """Signed polarity in [-1, 1] from mean class probs."""
    if ctx.mean_positive is None or ctx.mean_negative is None:
        msg = "polarity requires scored sentiment means"
        raise ForecastValidationError(msg)
    return float(ctx.mean_positive) - float(ctx.mean_negative)


def require_news_context(
    ctx: SentimentContext,
    *,
    require_full_scores: bool,
) -> None:
    if ctx.status == "missing":
        msg = "news ablation refused: sentiment context missing (not neutral)"
        raise InsufficientDataError(msg)
    if ctx.status == "unscored":
        msg = "news ablation refused: news present but unscored"
        raise InsufficientDataError(msg)
    if require_full_scores and ctx.status == "partial":
        msg = (
            f"news ablation refused: partial scores "
            f"({ctx.n_scored}/{ctx.n_items}); not imputed"
        )
        raise InsufficientDataError(msg)
    if ctx.status not in {"scored", "partial"}:
        msg = f"news ablation refused: unexpected status {ctx.status}"
        raise InsufficientDataError(msg)


def compute_news_delta(
    ctx: SentimentContext,
    *,
    volume_weight: float,
    sentiment_weight: float,
) -> float:
    vol = news_volume_feature(ctx.n_items)
    delta = volume_weight * vol
    if sentiment_weight != 0.0:
        delta += sentiment_weight * news_polarity(ctx)
    return delta


def issue_news_adjusted_from_parent(
    parent: ForecastContract,
    *,
    spec: ExperimentSpec,
    ctx: SentimentContext,
) -> ForecastContract:
    """Shift parent return quantiles by frozen news delta; keep parent intact."""
    require_full = bool(spec.hyperparameters.get("require_full_scores", False))
    require_news_context(ctx, require_full_scores=require_full)
    volume_weight = float(spec.hyperparameters["volume_weight"])
    sentiment_weight = float(spec.hyperparameters["sentiment_weight"])
    delta = compute_news_delta(
        ctx, volume_weight=volume_weight, sentiment_weight=sentiment_weight
    )
    shifted = [float(v) + delta for v in parent.quantile_values]
    values = _isotonic_increasing(shifted)
    if any(values[i] > values[i + 1] + 1e-9 for i in range(len(values) - 1)):
        msg = f"{spec.experiment_id} produced crossing quantiles after isotonic"
        raise ForecastValidationError(msg)

    max_feat = parent.maximum_feature_timestamp
    if ctx.max_published_at and ctx.max_published_at > max_feat:
        max_feat = ctx.max_published_at
    if ctx.max_ingested_at and ctx.max_ingested_at > max_feat:
        max_feat = ctx.max_ingested_at
    if max_feat > parent.issued_at:
        msg = "news feature timestamp exceeds issued_at"
        raise ForecastValidationError(msg)

    created = to_iso_utc(utc_now())
    contract = ForecastContract(
        forecast_id=str(uuid.uuid4()),
        run_id=parent.run_id,
        experiment_id=spec.experiment_id,
        ticker=parent.ticker,
        model_family=spec.model_family,
        model_version=f"{spec.experiment_id}-v{spec.version}",
        artifact_digest=spec.config_hash(),
        feature_set=spec.feature_set,
        feature_version=spec.feature_version,
        training_cutoff=parent.training_cutoff,
        data_as_of=parent.data_as_of,
        maximum_feature_timestamp=max_feat,
        issued_at=parent.issued_at,
        origin_bar_at=parent.origin_bar_at,
        target_at=parent.target_at,
        nominal_horizon_hours=parent.nominal_horizon_hours,
        spot_at_issue=parent.spot_at_issue,
        target_definition=TARGET_LOG_RETURN,
        quantile_levels=list(parent.quantile_levels),
        quantile_values=values,
        forecast_space=FORECAST_SPACE_RETURN,
        parent_forecast_id=parent.forecast_id,
        random_seed=parent.random_seed,
        created_at=created,
        is_synthetic=parent.is_synthetic or ctx.is_synthetic,
        sentiment_context=ctx.as_dict(),
        generation_metadata={
            "parent_experiment_id": parent.experiment_id,
            "parent_forecast_id": parent.forecast_id,
            "news_delta": delta,
            "volume_feature": news_volume_feature(ctx.n_items),
            "polarity": (news_polarity(ctx) if sentiment_weight != 0.0 else None),
            "volume_weight": volume_weight,
            "sentiment_weight": sentiment_weight,
            "adjustment": spec.hyperparameters.get("adjustment"),
            "experiment_config_hash": spec.config_hash(),
            "missing_is_not_neutral": True,
        },
    )
    return validate_forecast_contract(contract)


def issue_n0_news_only(
    *,
    ticker: str,
    issued_at: str,
    origin_bar_at: str,
    target_at: str,
    spot_at_issue: float,
    horizon_hours: int,
    training_cutoff: str,
    data_as_of: str,
    ctx: SentimentContext,
    is_synthetic: bool = False,
    run_id: str | None = None,
    quantiles: Sequence[float] = DEFAULT_QUANTILES,
) -> ForecastContract:
    """News-only baseline: P50 from polarity; width from volume (no market returns)."""
    require_news_context(ctx, require_full_scores=True)
    p50_w = float(N0.hyperparameters["sentiment_to_p50"])
    width_w = float(N0.hyperparameters["volume_to_width"])
    base_w = float(N0.hyperparameters["base_width"])
    polarity = news_polarity(ctx)
    vol = news_volume_feature(ctx.n_items)
    p50 = p50_w * polarity
    half_width = base_w + width_w * vol
    levels = list(quantiles)
    # Symmetric tent around p50 scaled by quantile distance from 0.5.
    values = [p50 + half_width * (2.0 * q - 1.0) for q in levels]
    values = _isotonic_increasing(values)

    max_feat = data_as_of
    if ctx.max_published_at and ctx.max_published_at > max_feat:
        max_feat = ctx.max_published_at
    if ctx.max_ingested_at and ctx.max_ingested_at > max_feat:
        max_feat = ctx.max_ingested_at
    if max_feat > issued_at:
        msg = "N0 news feature timestamp exceeds issued_at"
        raise ForecastValidationError(msg)

    created = to_iso_utc(utc_now())
    contract = ForecastContract(
        forecast_id=str(uuid.uuid4()),
        run_id=run_id,
        experiment_id=N0.experiment_id,
        ticker=ticker,
        model_family=N0.model_family,
        model_version=f"{N0.experiment_id}-v{N0.version}",
        artifact_digest=N0.config_hash(),
        feature_set=FEATURE_SET_NEWS_ONLY,
        feature_version=N0.feature_version,
        training_cutoff=training_cutoff,
        data_as_of=data_as_of,
        maximum_feature_timestamp=max_feat,
        issued_at=issued_at,
        origin_bar_at=origin_bar_at,
        target_at=target_at,
        nominal_horizon_hours=horizon_hours,
        spot_at_issue=spot_at_issue,
        target_definition=TARGET_LOG_RETURN,
        quantile_levels=levels,
        quantile_values=values,
        forecast_space=FORECAST_SPACE_RETURN,
        created_at=created,
        is_synthetic=is_synthetic or ctx.is_synthetic,
        sentiment_context=ctx.as_dict(),
        generation_metadata={
            "p50": p50,
            "half_width": half_width,
            "polarity": polarity,
            "volume_feature": vol,
            "experiment_config_hash": N0.config_hash(),
            "missing_is_not_neutral": True,
            "note": "News-only baseline; no market return features.",
        },
    )
    return validate_forecast_contract(contract)


def load_context_for_ablation(
    conn: Any,
    *,
    ticker: str,
    issued_at: str,
    lookback_hours: int,
    scorer_id: str = SCORER_FAKE,
) -> SentimentContext:
    return build_sentiment_context(
        conn,
        ticker=ticker,
        issued_at=issued_at,
        scorer_id=scorer_id,
        lookback_hours=lookback_hours,
    )


def issue_t1_from_t0(
    parent: ForecastContract, ctx: SentimentContext
) -> ForecastContract:
    return issue_news_adjusted_from_parent(parent, spec=T1, ctx=ctx)


def issue_t2_from_t0(
    parent: ForecastContract, ctx: SentimentContext
) -> ForecastContract:
    return issue_news_adjusted_from_parent(parent, spec=T2, ctx=ctx)


def issue_m1_from_m0(
    parent: ForecastContract, ctx: SentimentContext
) -> ForecastContract:
    return issue_news_adjusted_from_parent(parent, spec=M1, ctx=ctx)


__all__ = [
    "compute_news_delta",
    "issue_m1_from_m0",
    "issue_n0_news_only",
    "issue_news_adjusted_from_parent",
    "issue_t1_from_t0",
    "issue_t2_from_t0",
    "load_context_for_ablation",
    "news_polarity",
    "news_volume_feature",
    "require_news_context",
]
