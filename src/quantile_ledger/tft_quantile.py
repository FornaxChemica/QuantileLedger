"""T0: market-only Temporal Fusion-style quantile model (local, dependency-light).

This is a compact TFT-inspired encoder: gated lookback selection + single-head
attention over lagged log returns, then multi-quantile heads trained with
pinball loss and isotonic projection. It does **not** require
``pytorch-forecasting`` / Lightning. Experiment ID: T0.
"""

from __future__ import annotations

import hashlib
import json
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
from quantile_ledger.experiments import T0
from quantile_ledger.mamba_quantile import build_supervised_windows
from quantile_ledger.metrics import pinball_loss
from quantile_ledger.timeutil import to_iso_utc, utc_now

BACKBONE = "local_tft_attention_numpy_v1"


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _softmax(xs: Sequence[float]) -> list[float]:
    if not xs:
        return []
    m = max(xs)
    exps = [math.exp(x - m) for x in xs]
    s = sum(exps) or 1.0
    return [e / s for e in exps]


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


@dataclass
class TemporalFusionQuantileModel:
    """Multi-quantile head on a gated attention encoding of lagged returns."""

    lookback: int
    hidden_dim: int
    quantile_levels: tuple[float, ...]
    seed: int
    # Variable-selection / gate over lookback positions.
    gate_w: list[float]
    gate_b: list[float]
    # Attention query/key/value projections (scalar→hidden, then score).
    attn_q: list[float]
    attn_k: list[float]
    attn_v: list[float]
    # Quantile heads on context mean.
    q_bias: list[float]
    q_scale: list[float]

    @classmethod
    def create(
        cls,
        *,
        lookback: int = 32,
        hidden_dim: int = 8,
        quantile_levels: Sequence[float] = DEFAULT_QUANTILES,
        seed: int = 0,
    ) -> TemporalFusionQuantileModel:
        rng = _RNG(seed)
        levels = tuple(quantile_levels)
        return cls(
            lookback=lookback,
            hidden_dim=hidden_dim,
            quantile_levels=levels,
            seed=seed,
            gate_w=[rng.uniform(-0.3, 0.3) for _ in range(lookback)],
            gate_b=[rng.uniform(-0.1, 0.1) for _ in range(lookback)],
            attn_q=[rng.uniform(-0.4, 0.4) for _ in range(hidden_dim)],
            attn_k=[rng.uniform(-0.4, 0.4) for _ in range(hidden_dim)],
            attn_v=[rng.uniform(-0.4, 0.4) for _ in range(hidden_dim)],
            q_bias=[0.0 for _ in levels],
            q_scale=[0.04 + 0.015 * i for i in range(len(levels))],
        )

    def encode(self, window: Sequence[float]) -> float:
        if len(window) != self.lookback:
            msg = f"expected lookback {self.lookback}, got {len(window)}"
            raise ValueError(msg)
        # Gated selection of each lagged return (TFT-style variable selection lite).
        gated = [
            x * _sigmoid(self.gate_w[i] * x + self.gate_b[i])
            for i, x in enumerate(window)
        ]
        # Project to hidden and attend (single query from mean gated value).
        mean_g = sum(gated) / len(gated)
        query = [self.attn_q[j] * mean_g for j in range(self.hidden_dim)]
        keys: list[list[float]] = []
        values: list[list[float]] = []
        for x in gated:
            keys.append([self.attn_k[j] * x for j in range(self.hidden_dim)])
            values.append([self.attn_v[j] * x for j in range(self.hidden_dim)])
        scores = [
            sum(query[j] * keys[t][j] for j in range(self.hidden_dim))
            / math.sqrt(self.hidden_dim)
            for t in range(self.lookback)
        ]
        weights = _softmax(scores)
        context = [
            sum(weights[t] * values[t][j] for t in range(self.lookback))
            for j in range(self.hidden_dim)
        ]
        return sum(context) / self.hidden_dim

    def predict_quantiles(self, window: Sequence[float]) -> list[float]:
        z = self.encode(window)
        raw = [b + s * z for b, s in zip(self.q_bias, self.q_scale, strict=True)]
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
            for window, y in zip(windows, targets, strict=True):
                pred = self.predict_quantiles(window)
                for qi, q in enumerate(self.quantile_levels):
                    total += pinball_loss(y, pred[qi], q)
            last_loss = total / (len(windows) * len(self.quantile_levels))
            for window, y in zip(windows, targets, strict=True):
                z = self.encode(window)
                pred = self.predict_quantiles(window)
                for qi, q in enumerate(self.quantile_levels):
                    err = y - pred[qi]
                    dpred = -q if err >= 0 else -(q - 1.0)
                    self.q_bias[qi] -= learning_rate * dpred
                    self.q_scale[qi] -= learning_rate * dpred * z
            self.q_scale = [max(1e-4, s) for s in self.q_scale]
            # Light gate updates toward recent residual signal.
            for window, y in zip(windows, targets, strict=True):
                mid = self.predict_quantiles(window)[len(self.quantile_levels) // 2]
                resid = y - mid
                for i, x in enumerate(window):
                    self.gate_w[i] -= learning_rate * 0.01 * resid * x
                    self.gate_b[i] -= learning_rate * 0.01 * resid
        return {"mean_pinball": last_loss, "n": float(len(windows))}

    def artifact_digest(self) -> str:
        return hashlib.sha256(
            json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        ).hexdigest()

    def to_dict(self) -> dict[str, object]:
        return {
            "lookback": self.lookback,
            "hidden_dim": self.hidden_dim,
            "quantile_levels": list(self.quantile_levels),
            "seed": self.seed,
            "gate_w": list(self.gate_w),
            "gate_b": list(self.gate_b),
            "attn_q": list(self.attn_q),
            "attn_k": list(self.attn_k),
            "attn_v": list(self.attn_v),
            "q_bias": list(self.q_bias),
            "q_scale": list(self.q_scale),
            "backbone": BACKBONE,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> TemporalFusionQuantileModel:
        levels_raw = payload["quantile_levels"]
        assert isinstance(levels_raw, list)
        levels = tuple(float(x) for x in levels_raw)

        def _floats(key: str) -> list[float]:
            raw = payload[key]
            assert isinstance(raw, list)
            return [float(x) for x in raw]

        return cls(
            lookback=int(str(payload["lookback"])),
            hidden_dim=int(str(payload["hidden_dim"])),
            quantile_levels=levels,
            seed=int(str(payload["seed"])),
            gate_w=_floats("gate_w"),
            gate_b=_floats("gate_b"),
            attn_q=_floats("attn_q"),
            attn_k=_floats("attn_k"),
            attn_v=_floats("attn_v"),
            q_bias=_floats("q_bias"),
            q_scale=_floats("q_scale"),
        )


class _RNG:
    def __init__(self, seed: int) -> None:
        self._state = seed % (2**31 - 1)
        if self._state <= 0:
            self._state = 1

    def random(self) -> float:
        self._state = (self._state * 48271) % 2147483647
        return self._state / 2147483647

    def uniform(self, a: float, b: float) -> float:
        return a + (b - a) * self.random()


def train_t0_model(
    closes: Sequence[float],
    *,
    bar_horizon: int,
    lookback: int = 32,
    hidden_dim: int = 8,
    epochs: int = 40,
    learning_rate: float = 0.05,
    seed: int = 0,
    min_samples: int = 40,
    quantile_levels: Sequence[float] = DEFAULT_QUANTILES,
) -> tuple[TemporalFusionQuantileModel, dict[str, float]]:
    windows, targets = build_supervised_windows(
        closes, lookback=lookback, bar_horizon=bar_horizon
    )
    if len(windows) < min_samples:
        msg = f"T0 insufficient training samples: {len(windows)} < {min_samples}"
        raise InsufficientDataError(msg)
    model = TemporalFusionQuantileModel.create(
        lookback=lookback,
        hidden_dim=hidden_dim,
        quantile_levels=quantile_levels,
        seed=seed,
    )
    metrics = model.train(windows, targets, epochs=epochs, learning_rate=learning_rate)
    return model, metrics


def issue_t0_forecast(
    model: TemporalFusionQuantileModel,
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
            f"T0 needs {model.lookback} one-step returns, "
            f"got {len(recent_one_step_log_returns)}"
        )
        raise InsufficientDataError(msg)
    window = list(recent_one_step_log_returns[-model.lookback :])
    values = model.predict_quantiles(window)
    if any(values[i] > values[i + 1] + 1e-9 for i in range(len(values) - 1)):
        msg = "T0 produced crossing quantiles after isotonic projection"
        raise ForecastValidationError(msg)
    created = to_iso_utc(utc_now())
    contract = ForecastContract(
        forecast_id=str(uuid.uuid4()),
        run_id=run_id,
        experiment_id=T0.experiment_id,
        ticker=ticker,
        model_family=T0.model_family,
        model_version=f"{T0.experiment_id}-v{T0.version}",
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
            "backbone": BACKBONE,
            "lookback": model.lookback,
            "hidden_dim": model.hidden_dim,
            "monotonicity_method": "isotonic_pav_v1",
            "experiment_config_hash": T0.config_hash(),
            "note": (
                "Local TFT-style attention quantile model; not "
                "pytorch-forecasting TemporalFusionTransformer."
            ),
        },
    )
    return validate_forecast_contract(contract)
