"""Strict paired cohort comparison across experiments."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

from quantile_ledger.metrics import (
    aggregate_quantile_loss,
    coverage_error,
    interval_coverage,
    mean_approx_crps,
    mean_relative_width,
    mean_width,
    median_absolute_error,
    skill_score,
)


@dataclass(frozen=True)
class SettledForecast:
    """Minimal settled forecast view for pairing."""

    forecast_id: str
    experiment_id: str
    ticker: str
    issued_at: str
    origin_bar_at: str
    target_at: str
    horizon_hours: int
    spot_at_issue: float
    quantile_levels: tuple[float, ...]
    quantile_values: tuple[float, ...]  # return space
    actual_return: float
    price_type: str = "adjusted_research"


@dataclass
class Exclusion:
    reason: str
    forecast_id: str | None = None
    detail: str = ""


@dataclass
class PairedComparison:
    key_fields: tuple[str, ...]
    cohort: list[tuple[SettledForecast, ...]]
    exclusions: list[Exclusion] = field(default_factory=list)
    metrics_by_experiment: dict[str, dict[str, float | int | None]] = field(
        default_factory=dict
    )


def comparison_key(f: SettledForecast) -> tuple[str, str, str, str, int, str]:
    return (
        f.ticker,
        f.issued_at,
        f.origin_bar_at,
        f.target_at,
        f.horizon_hours,
        f.price_type,
    )


def build_paired_cohort(
    forecasts: Sequence[SettledForecast],
    *,
    experiment_ids: Sequence[str],
) -> PairedComparison:
    """Keep only keys where every requested experiment has exactly one forecast."""
    wanted = list(experiment_ids)
    by_key: dict[tuple[str, str, str, str, int, str], dict[str, SettledForecast]] = {}
    exclusions: list[Exclusion] = []
    for f in forecasts:
        if f.experiment_id not in wanted:
            exclusions.append(
                Exclusion(
                    reason="experiment_not_requested",
                    forecast_id=f.forecast_id,
                    detail=f.experiment_id,
                )
            )
            continue
        key = comparison_key(f)
        bucket = by_key.setdefault(key, {})
        if f.experiment_id in bucket:
            exclusions.append(
                Exclusion(
                    reason="duplicate_experiment_on_key",
                    forecast_id=f.forecast_id,
                    detail=f.experiment_id,
                )
            )
            continue
        bucket[f.experiment_id] = f

    cohort: list[tuple[SettledForecast, ...]] = []
    for key, bucket in sorted(by_key.items(), key=lambda kv: kv[0]):
        missing = [eid for eid in wanted if eid not in bucket]
        if missing:
            exclusions.append(
                Exclusion(
                    reason="incomplete_key",
                    detail=f"key={key}; missing={missing}",
                )
            )
            continue
        cohort.append(tuple(bucket[eid] for eid in wanted))

    result = PairedComparison(
        key_fields=(
            "ticker",
            "issued_at",
            "origin_bar_at",
            "target_at",
            "horizon_hours",
            "price_type",
        ),
        cohort=cohort,
        exclusions=exclusions,
    )
    result.metrics_by_experiment = score_paired_cohort(result, wanted)
    return result


def score_paired_cohort(
    paired: PairedComparison,
    experiment_ids: Sequence[str],
    *,
    baseline_id: str = "B1",
    p10: float = 0.10,
    p90: float = 0.90,
    p50: float = 0.50,
) -> dict[str, dict[str, float | int | None]]:
    out: dict[str, dict[str, float | int | None]] = {}
    if not paired.cohort:
        for eid in experiment_ids:
            out[eid] = {"sample_count": 0}
        return out

    # Assume shared quantile grid from first forecast of each experiment.
    for idx, eid in enumerate(experiment_ids):
        actuals = [row[idx].actual_return for row in paired.cohort]
        levels = list(paired.cohort[0][idx].quantile_levels)
        by_q: dict[float, Sequence[float]] = {
            q: [
                dict(
                    zip(row[idx].quantile_levels, row[idx].quantile_values, strict=True)
                )[q]
                for row in paired.cohort
            ]
            for q in levels
        }
        lowers: list[float] = []
        uppers: list[float] = []
        medians: list[float] = []
        spots: list[float] = []
        for row in paired.cohort:
            f = row[idx]
            qmap = dict(zip(f.quantile_levels, f.quantile_values, strict=True))
            lowers.append(qmap[p10])
            uppers.append(qmap[p90])
            medians.append(qmap[p50])
            spots.append(f.spot_at_issue)
        cov = interval_coverage(actuals, lowers, uppers)
        loss = aggregate_quantile_loss(actuals, levels, by_q)
        out[eid] = {
            "sample_count": len(actuals),
            "mean_pinball": loss,
            "approx_crps": mean_approx_crps(actuals, levels, by_q),
            "coverage_p10_p90": cov,
            "coverage_error_vs_0_80": coverage_error(cov, 0.80),
            "mean_width_return": mean_width(lowers, uppers),
            "mean_relative_width_price": mean_relative_width(
                [math_exp_price(s, lo) for s, lo in zip(spots, lowers, strict=True)],
                [math_exp_price(s, hi) for s, hi in zip(spots, uppers, strict=True)],
                spots,
            ),
            "median_ae": median_absolute_error(actuals, medians),
        }

    if baseline_id in out and out[baseline_id].get("mean_pinball") is not None:
        base_loss = out[baseline_id]["mean_pinball"]
        assert isinstance(base_loss, float)
        for eid in experiment_ids:
            model_loss = out[eid].get("mean_pinball")
            if isinstance(model_loss, float):
                out[eid]["skill_vs_B1"] = skill_score(model_loss, base_loss)
            else:
                out[eid]["skill_vs_B1"] = None
    return out


def math_exp_price(spot: float, log_return: float) -> float:
    return spot * math.exp(log_return)
