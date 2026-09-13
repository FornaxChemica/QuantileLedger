"""Phase H isotonic PIT recalibration tests."""

from __future__ import annotations

import pytest

from quantile_ledger.contract import ForecastContract, validate_forecast_contract
from quantile_ledger.errors import ForecastValidationError
from quantile_ledger.recalibration import (
    CALIBRATION_VARIANT,
    apply_isotonic_pit,
    fit_isotonic_pit_map,
    issue_recalibrated_from_parent,
    predictive_cdf,
    split_issuance_periods,
)


def _parent(
    *,
    forecast_id: str = "parent-1",
    experiment_id: str = "M0",
    issued_at: str = "2024-01-20T15:00:00Z",
    values: list[float] | None = None,
) -> ForecastContract:
    levels = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
    if values is None:
        values = [-0.04, -0.02, -0.01, 0.0, 0.01, 0.02, 0.04]
    return validate_forecast_contract(
        {
            "forecast_id": forecast_id,
            "experiment_id": experiment_id,
            "ticker": "SYN",
            "model_family": "mamba_quantile",
            "model_version": "M0-v1",
            "feature_set": "market_only",
            "feature_version": "v1",
            "training_cutoff": "2024-01-10T15:00:00Z",
            "data_as_of": issued_at,
            "maximum_feature_timestamp": issued_at,
            "issued_at": issued_at,
            "origin_bar_at": issued_at,
            "target_at": issued_at.replace("T15:00:00Z", "T21:00:00Z"),
            "nominal_horizon_hours": 6,
            "spot_at_issue": 100.0,
            "quantile_levels": levels,
            "quantile_values": values,
            "created_at": issued_at,
            "is_synthetic": True,
        }
    )


def test_predictive_cdf_at_median() -> None:
    levels = [0.1, 0.5, 0.9]
    values = [-0.02, 0.0, 0.02]
    assert predictive_cdf(0.0, levels, values) == pytest.approx(0.5)


def test_split_and_refuse_eval_in_fit() -> None:
    ats = [f"2024-01-{d:02d}T15:00:00Z" for d in range(1, 6)]
    period = split_issuance_periods(ats, val_fraction=0.6, min_validation=3, min_eval=2)
    assert len(period.validation_issued_ats) == 3
    assert len(period.eval_issued_ats) == 2
    assert not (period.validation_issued_ats & period.eval_issued_ats)
    with pytest.raises(ForecastValidationError, match="non-validation"):
        period.assert_fit_issued_ats([ats[-1]])


def test_fit_refuses_eval_leak() -> None:
    ats = [f"2024-01-{d:02d}T15:00:00Z" for d in range(1, 6)]
    period = split_issuance_periods(ats)
    levels = [[0.1, 0.5, 0.9]] * 3
    values = [[-0.01, 0.0, 0.01]] * 3
    with pytest.raises(ForecastValidationError):
        fit_isotonic_pit_map(
            parent_experiment_id="M0",
            levels_list=levels,
            values_list=values,
            actuals=[0.0, 0.01, -0.01],
            issued_ats=[ats[0], ats[1], ats[-1]],
            period=period,
        )


def test_issue_child_eval_only_and_immutable_fields() -> None:
    ats = [f"2024-01-{d:02d}T15:00:00Z" for d in range(10, 15)]
    period = split_issuance_periods(ats)
    val_ats = sorted(period.validation_issued_ats)
    levels = [[0.1, 0.5, 0.9] for _ in val_ats]
    values = [[-0.02, 0.0, 0.02] for _ in val_ats]
    actuals = [0.0] * len(val_ats)
    pit_map = fit_isotonic_pit_map(
        parent_experiment_id="M0",
        levels_list=levels,
        values_list=values,
        actuals=actuals,
        issued_ats=val_ats,
        period=period,
    )
    eval_at = sorted(period.eval_issued_ats)[0]
    parent = _parent(issued_at=eval_at, forecast_id="p-eval")
    child = issue_recalibrated_from_parent(parent, pit_map=pit_map, period=period)
    assert child.experiment_id == "M0c"
    assert child.variant == CALIBRATION_VARIANT
    assert child.parent_forecast_id == parent.forecast_id
    assert child.calibration_method == "isotonic_pit"
    assert child.forecast_id != parent.forecast_id
    assert parent.variant == "raw"
    assert parent.quantile_values == [-0.04, -0.02, -0.01, 0.0, 0.01, 0.02, 0.04]

    val_parent = _parent(issued_at=val_ats[0], forecast_id="p-val")
    with pytest.raises(ForecastValidationError, match="non-eval"):
        issue_recalibrated_from_parent(val_parent, pit_map=pit_map, period=period)


def test_apply_preserves_monotonicity() -> None:
    levels = [0.1, 0.5, 0.9]
    values = [-0.02, 0.0, 0.03]
    ats = [f"2024-02-{d:02d}T15:00:00Z" for d in range(1, 6)]
    period = split_issuance_periods(ats)
    val = sorted(period.validation_issued_ats)
    pit_map = fit_isotonic_pit_map(
        parent_experiment_id="T0",
        levels_list=[levels] * len(val),
        values_list=[values] * len(val),
        actuals=[0.0] * len(val),
        issued_ats=val,
        period=period,
    )
    out = apply_isotonic_pit(levels, values, pit_map)
    assert all(out[i] <= out[i + 1] + 1e-12 for i in range(len(out) - 1))
