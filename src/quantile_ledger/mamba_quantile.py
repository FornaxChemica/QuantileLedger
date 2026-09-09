"""M0: market-only MambaQuantile — diagonal selective SSM + pinball loss.

This is a local, dependency-light selective state-space quantile model
(Mamba-inspired discrete diagonal S6-style scan). It does not require the
CUDA ``mamba-ssm`` package. Experiment ID: M0.
"""

from __future__ import annotations

import hashlib
import math
import uuid
from collections.abc import Sequence
from dataclasses import dataclass

from quantile_ledger.contract import (
    DEFAULT_QUANTILES,
    FEATURE_SET_MARKET_ONLY,
    FEATURE_VERSION_V1,
    FORECAST_SPACE_RETURN,
    TARGET_LOG_RETURN,
    ForecastContract,
    validate_forecast_contract,
)
from quantile_ledger.errors import ForecastValidationError, InsufficientDataError
from quantile_ledger.experiments import M0
from quantile_ledger.metrics import pinball_loss
from quantile_ledger.timeutil import to_iso_utc, utc_now


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


@dataclass
class MambaQuantileModel:
    """Multi-quantile head on a shared diagonal selective SSM encoding."""

    lookback: int
    state_dim: int
    quantile_levels: tuple[float, ...]
    seed: int
    # Parameters
    log_delta: list[float]
    b_gate: list[float]
    c_out: list[float]
    # quantile heads: bias + scale on final state mean
    q_bias: list[float]
    q_scale: list[float]

    @classmethod
    def create(
        cls,
        *,
        lookback: int = 32,
        state_dim: int = 8,
        quantile_levels: Sequence[float] = DEFAULT_QUANTILES,
        seed: int = 0,
    ) -> MambaQuantileModel:
        rng = _RNG(seed)
        levels = tuple(quantile_levels)
        return cls(
            lookback=lookback,
            state_dim=state_dim,
            quantile_levels=levels,
            seed=seed,
            log_delta=[rng.uniform(-2.0, 0.0) for _ in range(state_dim)],
            b_gate=[rng.uniform(-0.5, 0.5) for _ in range(state_dim)],
            c_out=[rng.uniform(-0.5, 0.5) for _ in range(state_dim)],
            q_bias=[0.0 for _ in levels],
            q_scale=[0.05 + 0.02 * i for i in range(len(levels))],
        )

    def encode(self, window: Sequence[float]) -> float:
        """Return scalar readout from selective diagonal SSM scan."""
        if len(window) != self.lookback:
            msg = f"expected lookback {self.lookback}, got {len(window)}"
            raise ValueError(msg)
        state = [0.0] * self.state_dim
        for x in window:
            for j in range(self.state_dim):
                delta = math.exp(self.log_delta[j])
                # Input-dependent gate (selective).
                g = _sigmoid(self.b_gate[j] * x)
                a = math.exp(-delta)
                state[j] = a * state[j] + (1.0 - a) * g * x
        return sum(c * s for c, s in zip(self.c_out, state, strict=True))

    def predict_quantiles(self, window: Sequence[float]) -> list[float]:
        z = self.encode(window)
        raw = [b + s * z for b, s in zip(self.q_bias, self.q_scale, strict=True)]
        # Enforce non-decreasing via isotonic projection (explicit, versioned).
        return _isotonic_increasing(raw)

    def train(
        self,
        windows: Sequence[Sequence[float]],
        targets: Sequence[float],
        *,
        epochs: int = 40,
        learning_rate: float = 0.05,
    ) -> dict[str, float]:
        if len(windows) != len(targets) or not windows:
            msg = "train requires non-empty paired windows/targets"
            raise InsufficientDataError(msg)
        last_loss = 0.0
        for _ in range(epochs):
            total = 0.0
            # Finite-difference gradient on biases/scales (stable, dependency-free).
            for window, y in zip(windows, targets, strict=True):
                pred = self.predict_quantiles(window)
                for qi, q in enumerate(self.quantile_levels):
                    total += pinball_loss(y, pred[qi], q)
            last_loss = total / (len(windows) * len(self.quantile_levels))
            # Coordinate steps on q_bias / q_scale using pinball subgradient.
            for window, y in zip(windows, targets, strict=True):
                z = self.encode(window)
                pred = self.predict_quantiles(window)
                for qi, q in enumerate(self.quantile_levels):
                    err = y - pred[qi]
                    # subgradient of pinball w.r.t. prediction
                    dpred = -q if err >= 0 else -(q - 1.0)
                    self.q_bias[qi] -= learning_rate * dpred
                    self.q_scale[qi] -= learning_rate * dpred * z
            # Keep scales positive-ish for ordering stability.
            self.q_scale = [max(1e-4, s) for s in self.q_scale]
        return {"mean_pinball": last_loss, "n": float(len(windows))}

    def artifact_digest(self) -> str:
        return hashlib.sha256(
            json_dumps_stable(self.to_dict()).encode("utf-8")
        ).hexdigest()

    def to_dict(self) -> dict[str, object]:
        return {
            "lookback": self.lookback,
            "state_dim": self.state_dim,
            "quantile_levels": list(self.quantile_levels),
            "seed": self.seed,
            "log_delta": list(self.log_delta),
            "b_gate": list(self.b_gate),
            "c_out": list(self.c_out),
            "q_bias": list(self.q_bias),
            "q_scale": list(self.q_scale),
            "backbone": "diagonal_selective_ssm_numpy",
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> MambaQuantileModel:
        levels_raw = payload["quantile_levels"]
        assert isinstance(levels_raw, list)
        levels = tuple(float(x) for x in levels_raw)

        def _floats(key: str) -> list[float]:
            raw = payload[key]
            assert isinstance(raw, list)
            return [float(x) for x in raw]

        return cls(
            lookback=int(str(payload["lookback"])),
            state_dim=int(str(payload["state_dim"])),
            quantile_levels=levels,
            seed=int(str(payload["seed"])),
            log_delta=_floats("log_delta"),
            b_gate=_floats("b_gate"),
            c_out=_floats("c_out"),
            q_bias=_floats("q_bias"),
            q_scale=_floats("q_scale"),
        )


def json_dumps_stable(payload: dict[str, object]) -> str:
    import json

    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


class _RNG:
    def __init__(self, seed: int) -> None:
        self._state = seed % (2**31 - 1)
        if self._state <= 0:
            self._state = 1

    def random(self) -> float:
        # Park-Miller minimal standard
        self._state = (self._state * 48271) % 2147483647
        return self._state / 2147483647

    def uniform(self, a: float, b: float) -> float:
        return a + (b - a) * self.random()


def _isotonic_increasing(values: Sequence[float]) -> list[float]:
    """Pool-adjacent-violators for non-decreasing projection."""
    n = len(values)
    out = list(values)
    i = 0
    while i < n - 1:
        if out[i] <= out[i + 1] + 1e-15:
            i += 1
            continue
        # Pool back.
        j = i
        while j >= 0 and out[j] > out[j + 1]:
            avg = 0.5 * (out[j] + out[j + 1])
            out[j] = avg
            out[j + 1] = avg
            j -= 1
        i = max(i - 1, 0)
    return out


def build_supervised_windows(
    closes: Sequence[float],
    *,
    lookback: int,
    bar_horizon: int,
) -> tuple[list[list[float]], list[float]]:
    """Build PIT-safe windows: features end at t; target is log return t→t+h."""
    rets = []
    for i in range(len(closes) - 1):
        if closes[i] > 0 and closes[i + 1] > 0:
            rets.append(math.log(closes[i + 1] / closes[i]))
        else:
            rets.append(0.0)
    windows: list[list[float]] = []
    targets: list[float] = []
    # Need lookback one-step returns ending at t; target uses closes[t] to t+h.
    for t in range(lookback, len(closes) - bar_horizon):
        window = rets[t - lookback : t]
        left = closes[t]
        right = closes[t + bar_horizon]
        if left <= 0 or right <= 0:
            continue
        windows.append(list(window))
        targets.append(math.log(right / left))
    return windows, targets


def train_m0_model(
    closes: Sequence[float],
    *,
    bar_horizon: int,
    lookback: int = 32,
    state_dim: int = 8,
    epochs: int = 40,
    learning_rate: float = 0.05,
    seed: int = 0,
    min_samples: int = 40,
    quantile_levels: Sequence[float] = DEFAULT_QUANTILES,
) -> tuple[MambaQuantileModel, dict[str, float]]:
    windows, targets = build_supervised_windows(
        closes, lookback=lookback, bar_horizon=bar_horizon
    )
    if len(windows) < min_samples:
        msg = f"M0 insufficient training samples: {len(windows)} < {min_samples}"
        raise InsufficientDataError(msg)
    model = MambaQuantileModel.create(
        lookback=lookback,
        state_dim=state_dim,
        quantile_levels=quantile_levels,
        seed=seed,
    )
    metrics = model.train(windows, targets, epochs=epochs, learning_rate=learning_rate)
    return model, metrics


def issue_m0_forecast(
    model: MambaQuantileModel,
    *,
    ticker: str,
    issued_at: str,
    origin_bar_at: str,
    target_at: str,
    spot_at_issue: float,
    horizon_hours: int,
    recent_one_step_log_returns: Sequence[float],
    training_cutoff: str,
    data_as_of: str,
    is_synthetic: bool = False,
    run_id: str | None = None,
) -> ForecastContract:
    if len(recent_one_step_log_returns) < model.lookback:
        msg = (
            f"M0 needs {model.lookback} one-step returns, "
            f"got {len(recent_one_step_log_returns)}"
        )
        raise InsufficientDataError(msg)
    window = list(recent_one_step_log_returns[-model.lookback :])
    values = model.predict_quantiles(window)
    if any(values[i] > values[i + 1] + 1e-9 for i in range(len(values) - 1)):
        msg = "M0 produced crossing quantiles after isotonic projection"
        raise ForecastValidationError(msg)
    created = to_iso_utc(utc_now())
    contract = ForecastContract(
        forecast_id=str(uuid.uuid4()),
        run_id=run_id,
        experiment_id=M0.experiment_id,
        ticker=ticker,
        model_family=M0.model_family,
        model_version=f"{M0.experiment_id}-v{M0.version}",
        artifact_digest=model.artifact_digest(),
        feature_set=FEATURE_SET_MARKET_ONLY,
        feature_version=FEATURE_VERSION_V1,
        training_cutoff=training_cutoff,
        data_as_of=data_as_of,
        maximum_feature_timestamp=data_as_of,
        issued_at=issued_at,
        origin_bar_at=origin_bar_at,
        target_at=target_at,
        nominal_horizon_hours=horizon_hours,
        spot_at_issue=spot_at_issue,
        target_definition=TARGET_LOG_RETURN,
        quantile_levels=list(model.quantile_levels),
        quantile_values=values,
        forecast_space=FORECAST_SPACE_RETURN,
        random_seed=model.seed,
        created_at=created,
        is_synthetic=is_synthetic,
        generation_metadata={
            "ticker": ticker,
            "backbone": "diagonal_selective_ssm_numpy",
            "lookback": model.lookback,
            "state_dim": model.state_dim,
            "monotonicity_method": "isotonic_pav_v1",
            "experiment_config_hash": M0.config_hash(),
        },
    )
    return validate_forecast_contract(contract)
