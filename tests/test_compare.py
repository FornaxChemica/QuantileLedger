"""Paired cohort comparison tests."""

from __future__ import annotations

from quantile_ledger.compare import SettledForecast, build_paired_cohort


def _sf(
    eid: str,
    fid: str,
    actual: float,
    values: tuple[float, float, float],
) -> SettledForecast:
    return SettledForecast(
        forecast_id=fid,
        experiment_id=eid,
        ticker="SYN",
        issued_at="2024-01-10T15:00:00Z",
        origin_bar_at="2024-01-10T15:00:00Z",
        target_at="2024-01-10T21:00:00Z",
        horizon_hours=6,
        spot_at_issue=100.0,
        quantile_levels=(0.1, 0.5, 0.9),
        quantile_values=values,
        actual_return=actual,
    )


def test_paired_intersection_and_exclusions() -> None:
    forecasts = [
        _sf("B1", "a", 0.0, (-0.02, 0.0, 0.02)),
        _sf("M0", "b", 0.0, (-0.01, 0.0, 0.01)),
        SettledForecast(
            forecast_id="c",
            experiment_id="B1",
            ticker="SYN",
            issued_at="2024-01-11T15:00:00Z",
            origin_bar_at="2024-01-11T15:00:00Z",
            target_at="2024-01-11T21:00:00Z",
            horizon_hours=6,
            spot_at_issue=100.0,
            quantile_levels=(0.1, 0.5, 0.9),
            quantile_values=(-0.02, 0.0, 0.02),
            actual_return=0.1,
        ),
    ]
    paired = build_paired_cohort(forecasts, experiment_ids=["B1", "M0"])
    assert len(paired.cohort) == 1
    assert paired.metrics_by_experiment["B1"]["sample_count"] == 1
    assert any(e.reason == "incomplete_key" for e in paired.exclusions)
