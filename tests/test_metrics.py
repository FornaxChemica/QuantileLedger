"""Hand-calculated metric tests."""

from __future__ import annotations

import pytest

from quantile_ledger.metrics import (
    approx_crps_from_quantiles,
    coverage_error,
    interval_coverage,
    mean_pinball,
    pinball_loss,
    skill_score,
)


def test_pinball_hand_calc() -> None:
    # actual=1, pred=0, q=0.9 -> 0.9 * 1 = 0.9
    assert pinball_loss(1.0, 0.0, 0.9) == pytest.approx(0.9)
    # actual=0, pred=1, q=0.1 -> (0.1-1)*(-1) = 0.9
    assert pinball_loss(0.0, 1.0, 0.1) == pytest.approx(0.9)


def test_mean_pinball() -> None:
    assert mean_pinball([1.0, 0.0], [0.0, 1.0], 0.5) == pytest.approx(0.5)


def test_coverage_and_error() -> None:
    cov = interval_coverage([0.0, 0.5, 2.0], [-1.0, -1.0, -1.0], [1.0, 1.0, 1.0])
    assert cov == pytest.approx(2 / 3)
    assert coverage_error(cov, 0.80) == pytest.approx(2 / 3 - 0.80)


def test_skill_score() -> None:
    assert skill_score(0.5, 1.0) == pytest.approx(0.5)
    assert skill_score(1.0, 1.0) == pytest.approx(0.0)
    assert skill_score(2.0, 1.0) == pytest.approx(-1.0)
    assert skill_score(1.0, 0.0) is None


def test_approx_crps_positive() -> None:
    value = approx_crps_from_quantiles(0.0, [0.1, 0.5, 0.9], [-1.0, 0.0, 1.0])
    assert value >= 0.0
