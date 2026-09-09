"""Pure scoring functions: pinball, coverage, width, skill, approx CRPS."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence


def pinball_loss(actual: float, prediction: float, quantile: float) -> float:
    """Standard pinball / quantile loss for one observation."""
    error = actual - prediction
    if error >= 0:
        return quantile * error
    return (quantile - 1.0) * error


def mean_pinball(
    actuals: Sequence[float],
    predictions: Sequence[float],
    quantile: float,
) -> float:
    if len(actuals) != len(predictions):
        msg = "actuals and predictions length mismatch"
        raise ValueError(msg)
    if not actuals:
        msg = "empty sample"
        raise ValueError(msg)
    total = sum(
        pinball_loss(a, p, quantile) for a, p in zip(actuals, predictions, strict=True)
    )
    return total / len(actuals)


def aggregate_quantile_loss(
    actuals: Sequence[float],
    quantile_grid: Sequence[float],
    predicted_by_q: Mapping[float, Sequence[float]],
) -> float:
    """Mean pinball averaged equally across quantiles."""
    if not quantile_grid:
        msg = "empty quantile grid"
        raise ValueError(msg)
    losses = [mean_pinball(actuals, predicted_by_q[q], q) for q in quantile_grid]
    return sum(losses) / len(losses)


def approx_crps_from_quantiles(
    actual: float,
    quantile_levels: Sequence[float],
    quantile_values: Sequence[float],
) -> float:
    """Quantile-grid CRPS approximation (not full-distribution CRPS)."""
    if len(quantile_levels) != len(quantile_values):
        msg = "quantile level/value mismatch"
        raise ValueError(msg)
    if len(quantile_levels) < 2:
        msg = "need at least two quantiles for CRPS approximation"
        raise ValueError(msg)
    # Trapezoidal weights on adjacent quantile gaps.
    total = 0.0
    weight_sum = 0.0
    for i, (q, v) in enumerate(zip(quantile_levels, quantile_values, strict=True)):
        if i == 0:
            w = (quantile_levels[1] - q) / 2.0
        elif i == len(quantile_levels) - 1:
            w = (q - quantile_levels[i - 1]) / 2.0
        else:
            w = (quantile_levels[i + 1] - quantile_levels[i - 1]) / 2.0
        total += w * pinball_loss(actual, v, q)
        weight_sum += w
    if weight_sum <= 0:
        msg = "invalid quantile spacing"
        raise ValueError(msg)
    return 2.0 * total  # standard quantile CRPS scaling approximation


def mean_approx_crps(
    actuals: Sequence[float],
    quantile_levels: Sequence[float],
    predicted_by_q: Mapping[float, Sequence[float]],
) -> float:
    if not actuals:
        msg = "empty sample"
        raise ValueError(msg)
    n = len(actuals)
    acc = 0.0
    for i in range(n):
        values = [predicted_by_q[q][i] for q in quantile_levels]
        acc += approx_crps_from_quantiles(actuals[i], quantile_levels, values)
    return acc / n


def interval_coverage(
    actuals: Sequence[float],
    lower: Sequence[float],
    upper: Sequence[float],
) -> float:
    if not (len(actuals) == len(lower) == len(upper)):
        msg = "coverage input length mismatch"
        raise ValueError(msg)
    if not actuals:
        msg = "empty sample"
        raise ValueError(msg)
    hits = sum(
        1 for a, lo, hi in zip(actuals, lower, upper, strict=True) if lo <= a <= hi
    )
    return hits / len(actuals)


def coverage_error(observed_coverage: float, nominal: float = 0.80) -> float:
    return observed_coverage - nominal


def mean_width(lower: Sequence[float], upper: Sequence[float]) -> float:
    if len(lower) != len(upper) or not lower:
        msg = "width input invalid"
        raise ValueError(msg)
    return sum(hi - lo for lo, hi in zip(lower, upper, strict=True)) / len(lower)


def mean_relative_width(
    lower: Sequence[float],
    upper: Sequence[float],
    spots: Sequence[float],
) -> float:
    if not (len(lower) == len(upper) == len(spots)) or not lower:
        msg = "relative width input invalid"
        raise ValueError(msg)
    widths = []
    for lo, hi, spot in zip(lower, upper, spots, strict=True):
        if spot <= 0:
            msg = "spot must be positive"
            raise ValueError(msg)
        widths.append((hi - lo) / spot)
    return sum(widths) / len(widths)


def skill_score(model_loss: float, baseline_loss: float) -> float | None:
    """Positive means model beats baseline on lower-is-better loss."""
    if not math.isfinite(model_loss) or not math.isfinite(baseline_loss):
        return None
    if baseline_loss == 0.0:
        if model_loss == 0.0:
            return 0.0
        return None
    return 1.0 - (model_loss / baseline_loss)


def median_absolute_error(
    actuals: Sequence[float], predictions: Sequence[float]
) -> float:
    if len(actuals) != len(predictions) or not actuals:
        msg = "MAE input invalid"
        raise ValueError(msg)
    errors = sorted(abs(a - p) for a, p in zip(actuals, predictions, strict=True))
    mid = len(errors) // 2
    if len(errors) % 2 == 1:
        return errors[mid]
    return 0.5 * (errors[mid - 1] + errors[mid])
