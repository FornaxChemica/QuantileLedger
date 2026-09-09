"""Common probabilistic forecast contract and validation."""

from __future__ import annotations

import math
from typing import Any, Literal, Protocol

from pydantic import BaseModel, Field, field_validator, model_validator

from quantile_ledger.errors import ForecastValidationError

TARGET_LOG_RETURN = "log_return"
FORECAST_SPACE_RETURN: Literal["return"] = "return"
FEATURE_SET_MARKET_ONLY = "market_only"
FEATURE_VERSION_V1 = "v1"

DEFAULT_QUANTILES: tuple[float, ...] = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)


class QuantilePoint(BaseModel):
    """One quantile level with return- and price-space values."""

    q: float = Field(gt=0.0, lt=1.0)
    return_value: float
    price_value: float = Field(gt=0.0)

    @field_validator("return_value", "price_value")
    @classmethod
    def _finite(cls, value: float) -> float:
        if not math.isfinite(value):
            msg = "quantile values must be finite"
            raise ValueError(msg)
        return value


class ForecastContract(BaseModel):
    """Semantically identical contract for every probabilistic candidate."""

    forecast_id: str
    run_id: str | None = None
    experiment_id: str
    ticker: str
    model_family: str
    model_version: str
    artifact_digest: str | None = None
    feature_set: str
    feature_version: str
    training_cutoff: str
    data_as_of: str
    maximum_feature_timestamp: str
    issued_at: str
    origin_bar_at: str
    target_at: str
    nominal_horizon_hours: int = Field(gt=0)
    spot_at_issue: float = Field(gt=0.0)
    target_definition: str = TARGET_LOG_RETURN
    target_transform: str = "identity"
    quantile_levels: list[float]
    quantile_values: list[float]
    forecast_space: Literal["return", "price"] = FORECAST_SPACE_RETURN
    calibration_method: str = "none"
    calibration_version: str | None = None
    parent_forecast_id: str | None = None
    random_seed: int | None = None
    status: str = "issued"
    created_at: str
    price_type: str = "adjusted_research"
    variant: str = "raw"
    is_synthetic: bool = False
    generation_metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_invariants(self) -> ForecastContract:
        if len(self.quantile_levels) != len(self.quantile_values):
            msg = "quantile_levels and quantile_values length mismatch"
            raise ValueError(msg)
        if len(self.quantile_levels) < 1:
            msg = "at least one quantile is required"
            raise ValueError(msg)
        levels = self.quantile_levels
        if any(levels[i] >= levels[i + 1] for i in range(len(levels) - 1)):
            msg = "quantile_levels must be strictly increasing"
            raise ValueError(msg)
        values = self.quantile_values
        if any(not math.isfinite(v) for v in values):
            msg = "quantile_values must be finite"
            raise ValueError(msg)
        # Crossing in return space is a validation failure (not silently repaired).
        if any(values[i] > values[i + 1] + 1e-12 for i in range(len(values) - 1)):
            msg = "quantile_values cross (non-decreasing required in return space)"
            raise ValueError(msg)
        for left, right in (
            (self.training_cutoff, self.issued_at),
            (self.data_as_of, self.issued_at),
            (self.maximum_feature_timestamp, self.issued_at),
            (self.origin_bar_at, self.issued_at),
        ):
            if left > right:
                msg = (
                    f"time invariant violated: {left!r} must be <= issued_at "
                    f"{self.issued_at!r}"
                )
                raise ValueError(msg)
        if self.target_at <= self.issued_at:
            msg = "target_at must be strictly after issued_at"
            raise ValueError(msg)
        return self

    def price_quantiles(self) -> list[float]:
        """Map return-space quantiles to prices via P * exp(r)."""
        if self.forecast_space != FORECAST_SPACE_RETURN:
            msg = "price_quantiles requires forecast_space='return'"
            raise ForecastValidationError(msg)
        return [self.spot_at_issue * math.exp(r) for r in self.quantile_values]

    def as_quantile_points(self) -> list[QuantilePoint]:
        prices = self.price_quantiles()
        return [
            QuantilePoint(q=q, return_value=r, price_value=p)
            for q, r, p in zip(
                self.quantile_levels, self.quantile_values, prices, strict=True
            )
        ]


def validate_forecast_contract(
    payload: ForecastContract | dict[str, Any],
) -> ForecastContract:
    """Validate and return a ForecastContract.

    Raises ForecastValidationError on failure.
    """
    try:
        if isinstance(payload, ForecastContract):
            # Re-validate in case of model_construct bypass.
            return ForecastContract.model_validate(payload.model_dump())
        return ForecastContract.model_validate(payload)
    except Exception as exc:  # pydantic ValidationError or ValueError
        msg = f"forecast contract invalid: {exc}"
        raise ForecastValidationError(msg) from exc


class ForecastIssuer(Protocol):
    """Thin adapter protocol for probabilistic candidates."""

    experiment_id: str
    model_family: str

    def issue(self, **kwargs: Any) -> ForecastContract:
        """Produce one validated forecast contract instance."""
        ...
