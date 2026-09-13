"""Phase H isotonic PIT recalibration (separate child forecast rows).

Fit only on validation-window settled parents. Apply only to eval-window
parents. Never rewrite raw forecasts.
"""

from __future__ import annotations

import math
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from quantile_ledger.contract import (
    FORECAST_SPACE_RETURN,
    TARGET_LOG_RETURN,
    ForecastContract,
    validate_forecast_contract,
)
from quantile_ledger.errors import ForecastValidationError, InsufficientDataError
from quantile_ledger.experiments import (
    PARENT_TO_CALIBRATED,
    ExperimentSpec,
    get_experiment,
)
from quantile_ledger.timeutil import to_iso_utc, utc_now

CALIBRATION_METHOD = "isotonic_pit"
CALIBRATION_VERSION = "v1"
CALIBRATION_VARIANT = "isotonic_v1"
PIT_EPS = 1e-6


@dataclass(frozen=True)
class PeriodSplit:
    """Chronological validation / eval split by issued_at."""

    validation_issued_ats: frozenset[str]
    eval_issued_ats: frozenset[str]
    validation_start: str
    validation_end: str
    eval_start: str
    eval_end: str

    def assert_fit_issued_ats(self, issued_ats: Sequence[str]) -> None:
        bad = [t for t in issued_ats if t not in self.validation_issued_ats]
        if bad:
            msg = (
                "refusing recalibration fit on non-validation issued_at values: "
                f"{sorted(set(bad))[:5]}"
            )
            raise ForecastValidationError(msg)
        overlap = self.validation_issued_ats & self.eval_issued_ats
        if overlap:
            msg = f"validation/eval issued_at overlap: {sorted(overlap)[:5]}"
            raise ForecastValidationError(msg)


@dataclass(frozen=True)
class IsotonicPitMap:
    """Monotone map G: predicted CDF level -> empirical frequency."""

    x: tuple[float, ...]  # sorted support (predicted PIT / CDF levels)
    y: tuple[float, ...]  # isotonic targets in [0, 1]
    n_fit: int
    validation_start: str
    validation_end: str
    parent_experiment_id: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "x": list(self.x),
            "y": list(self.y),
            "n_fit": self.n_fit,
            "validation_start": self.validation_start,
            "validation_end": self.validation_end,
            "parent_experiment_id": self.parent_experiment_id,
            "calibration_method": CALIBRATION_METHOD,
            "calibration_version": CALIBRATION_VERSION,
        }

    def g(self, u: float) -> float:
        """Forward map G(u)."""
        return _interp_clamped(u, self.x, self.y)

    def g_inv(self, q: float) -> float:
        """Inverse G^{-1}(q) for quantile remapping."""
        return _interp_clamped(q, self.y, self.x)


def split_issuance_periods(
    issued_ats: Sequence[str],
    *,
    val_fraction: float = 0.6,
    min_validation: int = 3,
    min_eval: int = 2,
) -> PeriodSplit:
    """Split unique issued_at timestamps into validation then eval."""
    unique = sorted(set(issued_ats))
    n = len(unique)
    if n < min_validation + min_eval:
        msg = (
            f"need at least {min_validation + min_eval} distinct issued_at "
            f"for val/eval split; got {n}"
        )
        raise InsufficientDataError(msg)
    n_val = max(min_validation, math.floor(n * val_fraction))
    if n - n_val < min_eval:
        n_val = n - min_eval
    if n_val < min_validation:
        msg = f"validation slice too small after split: {n_val}"
        raise InsufficientDataError(msg)
    val = unique[:n_val]
    ev = unique[n_val:]
    return PeriodSplit(
        validation_issued_ats=frozenset(val),
        eval_issued_ats=frozenset(ev),
        validation_start=val[0],
        validation_end=val[-1],
        eval_start=ev[0],
        eval_end=ev[-1],
    )


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


def _interp_clamped(query: float, xs: Sequence[float], ys: Sequence[float]) -> float:
    if not xs or len(xs) != len(ys):
        msg = "interpolation grid empty or mismatched"
        raise ForecastValidationError(msg)
    if query <= xs[0]:
        return float(ys[0])
    if query >= xs[-1]:
        return float(ys[-1])
    for i in range(len(xs) - 1):
        x0, x1 = xs[i], xs[i + 1]
        if x0 <= query <= x1:
            if abs(x1 - x0) < 1e-18:
                return float(ys[i])
            t = (query - x0) / (x1 - x0)
            return float(ys[i] + t * (ys[i + 1] - ys[i]))
    return float(ys[-1])


def predictive_cdf(
    y: float,
    levels: Sequence[float],
    values: Sequence[float],
) -> float:
    """Invert piecewise-linear quantile function to get F̂(y)."""
    if len(levels) != len(values) or not levels:
        msg = "quantile ladder required for PIT"
        raise ForecastValidationError(msg)
    if y <= values[0]:
        return max(PIT_EPS, float(levels[0]))
    if y >= values[-1]:
        return min(1.0 - PIT_EPS, float(levels[-1]))
    for i in range(len(values) - 1):
        v0, v1 = values[i], values[i + 1]
        if v0 <= y <= v1:
            if abs(v1 - v0) < 1e-18:
                return float(levels[i])
            t = (y - v0) / (v1 - v0)
            u = levels[i] + t * (levels[i + 1] - levels[i])
            return min(1.0 - PIT_EPS, max(PIT_EPS, float(u)))
    return min(1.0 - PIT_EPS, float(levels[-1]))


def quantile_at_level(
    q: float,
    levels: Sequence[float],
    values: Sequence[float],
) -> float:
    """Evaluate quantile function at probability q (piecewise linear)."""
    return _interp_clamped(q, levels, values)


def fit_isotonic_pit_map(
    *,
    parent_experiment_id: str,
    levels_list: Sequence[Sequence[float]],
    values_list: Sequence[Sequence[float]],
    actuals: Sequence[float],
    issued_ats: Sequence[str],
    period: PeriodSplit,
) -> IsotonicPitMap:
    """Fit G on validation PITs only; refuse any eval issued_at in the fit set."""
    if not (len(levels_list) == len(values_list) == len(actuals) == len(issued_ats)):
        msg = "fit inputs length mismatch"
        raise ForecastValidationError(msg)
    period.assert_fit_issued_ats(issued_ats)
    if any(t in period.eval_issued_ats for t in issued_ats):
        msg = "eval issued_at leaked into recalibration fit set"
        raise ForecastValidationError(msg)
    pits: list[float] = []
    for levels, values, y in zip(levels_list, values_list, actuals, strict=True):
        pits.append(predictive_cdf(float(y), levels, values))
    n = len(pits)
    if n < 2:
        msg = f"need at least 2 validation PITs to fit; got {n}"
        raise InsufficientDataError(msg)
    order = sorted(range(n), key=lambda i: pits[i])
    xs = [pits[i] for i in order]
    # (i+1)/(n+1) plotting positions — avoid 0/1 endpoints.
    ys_raw = [(i + 1) / (n + 1) for i in range(n)]
    ys = _isotonic_increasing(ys_raw)
    # Enforce endpoints slightly expanded for stable inverse.
    xs_out = [PIT_EPS, *xs, 1.0 - PIT_EPS]
    ys_out = [PIT_EPS, *ys, 1.0 - PIT_EPS]
    ys_out = _isotonic_increasing(ys_out)
    return IsotonicPitMap(
        x=tuple(xs_out),
        y=tuple(ys_out),
        n_fit=n,
        validation_start=period.validation_start,
        validation_end=period.validation_end,
        parent_experiment_id=parent_experiment_id,
    )


def apply_isotonic_pit(
    levels: Sequence[float],
    values: Sequence[float],
    pit_map: IsotonicPitMap,
) -> list[float]:
    """Child quantile at q = parent quantile at G^{-1}(q); then PAV."""
    remapped = [
        quantile_at_level(pit_map.g_inv(float(q)), levels, values) for q in levels
    ]
    return _isotonic_increasing(remapped)


def issue_recalibrated_from_parent(
    parent: ForecastContract,
    *,
    pit_map: IsotonicPitMap,
    period: PeriodSplit,
    spec: ExperimentSpec | None = None,
) -> ForecastContract:
    """Emit calibrated child for an eval-window parent only."""
    if parent.issued_at not in period.eval_issued_ats:
        msg = f"refusing recalibrated child for non-eval issued_at {parent.issued_at!r}"
        raise ForecastValidationError(msg)
    if parent.issued_at in period.validation_issued_ats:
        msg = "refusing recalibrated child for validation-window parent"
        raise ForecastValidationError(msg)
    if parent.experiment_id != pit_map.parent_experiment_id:
        msg = (
            f"parent experiment {parent.experiment_id} != map parent "
            f"{pit_map.parent_experiment_id}"
        )
        raise ForecastValidationError(msg)
    child_id = PARENT_TO_CALIBRATED.get(parent.experiment_id)
    if child_id is None:
        msg = f"no calibrated experiment registered for {parent.experiment_id}"
        raise ForecastValidationError(msg)
    child_spec = spec or get_experiment(child_id)
    values = apply_isotonic_pit(parent.quantile_levels, parent.quantile_values, pit_map)
    if any(values[i] > values[i + 1] + 1e-9 for i in range(len(values) - 1)):
        msg = f"{child_spec.experiment_id} produced crossing quantiles after PAV"
        raise ForecastValidationError(msg)
    created = to_iso_utc(utc_now())
    contract = ForecastContract(
        forecast_id=str(uuid.uuid4()),
        run_id=parent.run_id,
        experiment_id=child_spec.experiment_id,
        ticker=parent.ticker,
        model_family=child_spec.model_family,
        model_version=f"{child_spec.experiment_id}-v{child_spec.version}",
        artifact_digest=child_spec.config_hash(),
        feature_set=child_spec.feature_set,
        feature_version=child_spec.feature_version,
        training_cutoff=parent.training_cutoff,
        data_as_of=parent.data_as_of,
        maximum_feature_timestamp=parent.maximum_feature_timestamp,
        issued_at=parent.issued_at,
        origin_bar_at=parent.origin_bar_at,
        target_at=parent.target_at,
        nominal_horizon_hours=parent.nominal_horizon_hours,
        spot_at_issue=parent.spot_at_issue,
        target_definition=TARGET_LOG_RETURN,
        quantile_levels=list(parent.quantile_levels),
        quantile_values=values,
        forecast_space=FORECAST_SPACE_RETURN,
        calibration_method=CALIBRATION_METHOD,
        calibration_version=CALIBRATION_VERSION,
        parent_forecast_id=parent.forecast_id,
        random_seed=parent.random_seed,
        created_at=created,
        variant=CALIBRATION_VARIANT,
        regime=parent.regime,
        regime_reasons=parent.regime_reasons,
        is_synthetic=parent.is_synthetic,
        sentiment_context=parent.sentiment_context,
        generation_metadata={
            "parent_experiment_id": parent.experiment_id,
            "parent_forecast_id": parent.forecast_id,
            "calibration_method": CALIBRATION_METHOD,
            "calibration_version": CALIBRATION_VERSION,
            "variant": CALIBRATION_VARIANT,
            "pit_map": pit_map.to_dict(),
            "period": {
                "validation_start": period.validation_start,
                "validation_end": period.validation_end,
                "eval_start": period.eval_start,
                "eval_end": period.eval_end,
            },
            "experiment_config_hash": child_spec.config_hash(),
        },
    )
    return validate_forecast_contract(contract)
