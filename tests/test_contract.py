"""Forecast contract invariant tests."""

from __future__ import annotations

import pytest

from quantile_ledger.contract import ForecastContract, validate_forecast_contract
from quantile_ledger.errors import ForecastValidationError


def _valid_kwargs() -> dict[str, object]:
    return {
        "forecast_id": "f1",
        "experiment_id": "M0",
        "ticker": "SYN",
        "model_family": "mamba_quantile",
        "model_version": "M0-v1",
        "feature_set": "market_only",
        "feature_version": "v1",
        "training_cutoff": "2024-01-10T15:00:00Z",
        "data_as_of": "2024-01-10T15:00:00Z",
        "maximum_feature_timestamp": "2024-01-10T15:00:00Z",
        "issued_at": "2024-01-10T15:00:00Z",
        "origin_bar_at": "2024-01-10T15:00:00Z",
        "target_at": "2024-01-10T21:00:00Z",
        "nominal_horizon_hours": 6,
        "spot_at_issue": 100.0,
        "quantile_levels": [0.1, 0.5, 0.9],
        "quantile_values": [-0.01, 0.0, 0.01],
        "created_at": "2024-01-10T15:00:01Z",
    }


def test_contract_accepts_valid_payload() -> None:
    fc = validate_forecast_contract(_valid_kwargs())
    assert fc.experiment_id == "M0"
    prices = fc.price_quantiles()
    assert prices[1] == pytest.approx(100.0)


def test_rejects_leaky_feature_timestamp() -> None:
    payload = _valid_kwargs()
    payload["maximum_feature_timestamp"] = "2024-01-10T16:00:00Z"
    with pytest.raises(ForecastValidationError):
        validate_forecast_contract(payload)


def test_rejects_quantile_crossing() -> None:
    payload = _valid_kwargs()
    payload["quantile_values"] = [0.02, 0.0, -0.01]
    with pytest.raises(ForecastValidationError):
        validate_forecast_contract(payload)


def test_rejects_target_before_issue() -> None:
    payload = _valid_kwargs()
    payload["target_at"] = "2024-01-10T14:00:00Z"
    with pytest.raises(ForecastValidationError):
        validate_forecast_contract(payload)


def test_model_validate_roundtrip() -> None:
    fc = ForecastContract.model_validate(_valid_kwargs())
    again = validate_forecast_contract(fc)
    assert again.forecast_id == "f1"
