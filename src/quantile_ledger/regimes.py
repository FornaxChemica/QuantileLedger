"""Point-in-time realized-vol regime labels (Phase H)."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from quantile_ledger.errors import ForecastValidationError, InsufficientDataError

REGIME_LOW = "vol_low"
REGIME_MID = "vol_mid"
REGIME_HIGH = "vol_high"
REGIME_LABELS: tuple[str, ...] = (REGIME_LOW, REGIME_MID, REGIME_HIGH)


@dataclass(frozen=True)
class VolTercileThresholds:
    """Tercile cutpoints fit on validation-era realized vols only."""

    low_max: float
    mid_max: float
    lookback: int
    n_fit: int
    fit_as_of_max: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "low_max": self.low_max,
            "mid_max": self.mid_max,
            "lookback": self.lookback,
            "n_fit": self.n_fit,
            "fit_as_of_max": self.fit_as_of_max,
            "method": "realized_vol_tercile_v1",
        }


def log_returns_from_closes(closes: Sequence[float]) -> list[float]:
    if len(closes) < 2:
        return []
    out: list[float] = []
    for i in range(1, len(closes)):
        if closes[i - 1] <= 0 or closes[i] <= 0:
            msg = "closes must be positive for log returns"
            raise ForecastValidationError(msg)
        out.append(math.log(closes[i] / closes[i - 1]))
    return out


def realized_vol(
    closes: Sequence[float],
    *,
    lookback: int = 20,
) -> float:
    """Stdev of the last `lookback` one-step log returns."""
    if lookback < 2:
        msg = "lookback must be >= 2"
        raise ForecastValidationError(msg)
    rets = log_returns_from_closes(closes)
    if len(rets) < lookback:
        msg = f"need {lookback} returns for realized vol; got {len(rets)}"
        raise InsufficientDataError(msg)
    window = rets[-lookback:]
    mean = sum(window) / len(window)
    var = sum((r - mean) ** 2 for r in window) / len(window)
    return math.sqrt(var)


def fit_vol_tercile_thresholds(
    vols: Sequence[float],
    *,
    fit_as_of_max: str,
    lookback: int = 20,
) -> VolTercileThresholds:
    """Fit tercile cutpoints from validation-era vols (never eval outcomes)."""
    clean = [float(v) for v in vols if math.isfinite(v)]
    if len(clean) < 3:
        msg = f"need at least 3 vols to fit terciles; got {len(clean)}"
        raise InsufficientDataError(msg)
    ordered = sorted(clean)
    n = len(ordered)
    i1 = max(0, min(n - 1, n // 3 - 1 if n >= 3 else 0))
    i2 = max(i1 + 1, min(n - 1, (2 * n) // 3 - 1 if n >= 3 else n - 1))
    # Use order-statistic boundaries so labels are exclusive on the right.
    low_max = ordered[max(0, (n // 3) - 1)]
    mid_max = ordered[max(0, (2 * n // 3) - 1)]
    if mid_max < low_max:
        mid_max = low_max
    _ = i1, i2  # kept for readability of index intent
    return VolTercileThresholds(
        low_max=float(low_max),
        mid_max=float(mid_max),
        lookback=lookback,
        n_fit=n,
        fit_as_of_max=fit_as_of_max,
    )


def assign_vol_regime(
    vol: float,
    thresholds: VolTercileThresholds,
) -> tuple[str, dict[str, Any]]:
    if not math.isfinite(vol):
        msg = "realized vol must be finite"
        raise ForecastValidationError(msg)
    if vol <= thresholds.low_max:
        label = REGIME_LOW
    elif vol <= thresholds.mid_max:
        label = REGIME_MID
    else:
        label = REGIME_HIGH
    reasons = {
        "method": "realized_vol_tercile_v1",
        "realized_vol": vol,
        "lookback": thresholds.lookback,
        "low_max": thresholds.low_max,
        "mid_max": thresholds.mid_max,
        "fit_as_of_max": thresholds.fit_as_of_max,
        "n_fit": thresholds.n_fit,
    }
    return label, reasons


def regime_for_closes(
    closes: Sequence[float],
    *,
    thresholds: VolTercileThresholds,
    bar_ends: Sequence[str] | None = None,
    issued_at: str | None = None,
) -> tuple[str, dict[str, Any]]:
    """Assign regime using only completes bars (caller must PIT-truncate closes)."""
    if bar_ends is not None and issued_at is not None:
        if len(bar_ends) != len(closes):
            msg = "bar_ends and closes length mismatch"
            raise ForecastValidationError(msg)
        if any(b > issued_at for b in bar_ends):
            msg = "regime feature bar_end exceeds issued_at (PIT violation)"
            raise ForecastValidationError(msg)
    vol = realized_vol(closes, lookback=thresholds.lookback)
    return assign_vol_regime(vol, thresholds)
