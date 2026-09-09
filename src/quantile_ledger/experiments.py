"""Experiment registry identities (research matrix)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Literal

from quantile_ledger.contract import (
    DEFAULT_QUANTILES,
    FEATURE_SET_MARKET_ONLY,
    FEATURE_VERSION_V1,
    TARGET_LOG_RETURN,
)

Lifecycle = Literal["draft", "candidate", "frozen", "retired", "invalidated"]


@dataclass(frozen=True)
class ExperimentSpec:
    experiment_id: str
    name: str
    version: str
    model_family: str
    feature_set: str
    feature_version: str
    target_definition: str
    status: Lifecycle
    purpose: str
    quantiles: tuple[float, ...]
    horizons_hours: tuple[int, ...]
    hyperparameters: dict[str, Any]
    label: str = "exploratory"

    def config_hash(self) -> str:
        payload = {
            "experiment_id": self.experiment_id,
            "version": self.version,
            "model_family": self.model_family,
            "feature_set": self.feature_set,
            "feature_version": self.feature_version,
            "target_definition": self.target_definition,
            "quantiles": list(self.quantiles),
            "horizons_hours": list(self.horizons_hours),
            "hyperparameters": self.hyperparameters,
        }
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def to_record(self) -> dict[str, Any]:
        data = asdict(self)
        data["config_hash"] = self.config_hash()
        data["quantiles"] = list(self.quantiles)
        data["horizons_hours"] = list(self.horizons_hours)
        return data


DEFAULT_HORIZONS: tuple[int, ...] = (1, 6, 24, 72)

B0 = ExperimentSpec(
    experiment_id="B0",
    name="persistence_baseline",
    version="1",
    model_family="persistence",
    feature_set=FEATURE_SET_MARKET_ONLY,
    feature_version=FEATURE_VERSION_V1,
    target_definition=TARGET_LOG_RETURN,
    status="candidate",
    purpose="Point sanity check: P50 log-return is zero at issuance spot.",
    quantiles=DEFAULT_QUANTILES,
    horizons_hours=DEFAULT_HORIZONS,
    hyperparameters={"distribution": "none", "point_only_median": True},
    label="confirmatory",
)

B1 = ExperimentSpec(
    experiment_id="B1",
    name="rolling_empirical_return",
    version="1",
    model_family="empirical_return",
    feature_set=FEATURE_SET_MARKET_ONLY,
    feature_version=FEATURE_VERSION_V1,
    target_definition=TARGET_LOG_RETURN,
    status="candidate",
    purpose="Primary probabilistic baseline from trailing realized log returns.",
    quantiles=DEFAULT_QUANTILES,
    horizons_hours=DEFAULT_HORIZONS,
    hyperparameters={
        "window": 200,
        "min_samples": 30,
        "pooling": "ticker_then_refuse",
    },
    label="confirmatory",
)

M0 = ExperimentSpec(
    experiment_id="M0",
    name="mamba_quantile_market_only",
    version="1",
    model_family="mamba_quantile",
    feature_set=FEATURE_SET_MARKET_ONLY,
    feature_version=FEATURE_VERSION_V1,
    target_definition=TARGET_LOG_RETURN,
    status="candidate",
    purpose=(
        "Local selective state-space (diagonal S6-style) multi-quantile model "
        "on market-only lagged log returns; optional CUDA mamba-ssm not required."
    ),
    quantiles=DEFAULT_QUANTILES,
    horizons_hours=DEFAULT_HORIZONS,
    hyperparameters={
        "lookback": 32,
        "state_dim": 8,
        "epochs": 40,
        "learning_rate": 0.05,
        "backbone": "diagonal_selective_ssm_numpy",
    },
    label="exploratory",
)

K0 = ExperimentSpec(
    experiment_id="K0",
    name="kronos_sample_quantile_market_only",
    version="1",
    model_family="kronos",
    feature_set=FEATURE_SET_MARKET_ONLY,
    feature_version=FEATURE_VERSION_V1,
    target_definition=TARGET_LOG_RETURN,
    status="candidate",
    purpose=(
        "Kronos foundation-model sample paths mapped to empirical return "
        "quantiles (market-only OHLC). Fake sampler for offline demo; real "
        "weights optional and gated. Sample-quantile approx, not a retained "
        "full predictive density."
    ),
    quantiles=DEFAULT_QUANTILES,
    horizons_hours=DEFAULT_HORIZONS,
    hyperparameters={
        "lookback": 64,
        "sample_count": 32,
        "temperature": 1.0,
        "top_p": 0.9,
        "model_id": "NeoQuasar/Kronos-small",
        "tokenizer_id": "NeoQuasar/Kronos-Tokenizer-base",
        "backbone": "kronos_or_fake_sampler",
        "quantile_method": "empirical_samples_v1",
        "distribution_claim": "sample_quantile_approx",
        "upstream_license": "MIT",
    },
    label="exploratory",
)

T0 = ExperimentSpec(
    experiment_id="T0",
    name="tft_quantile_market_only",
    version="1",
    model_family="tft_quantile",
    feature_set=FEATURE_SET_MARKET_ONLY,
    feature_version=FEATURE_VERSION_V1,
    target_definition=TARGET_LOG_RETURN,
    status="candidate",
    purpose=(
        "Local TFT-style gated attention multi-quantile model on market-only "
        "lagged log returns; pytorch-forecasting not required."
    ),
    quantiles=DEFAULT_QUANTILES,
    horizons_hours=DEFAULT_HORIZONS,
    hyperparameters={
        "lookback": 32,
        "hidden_dim": 8,
        "epochs": 40,
        "learning_rate": 0.05,
        "backbone": "local_tft_attention_numpy_v1",
    },
    label="exploratory",
)

REGISTRY: dict[str, ExperimentSpec] = {
    B0.experiment_id: B0,
    B1.experiment_id: B1,
    M0.experiment_id: M0,
    K0.experiment_id: K0,
    T0.experiment_id: T0,
}


def get_experiment(experiment_id: str) -> ExperimentSpec:
    try:
        return REGISTRY[experiment_id]
    except KeyError as exc:
        msg = f"unknown experiment_id: {experiment_id}"
        raise KeyError(msg) from exc


def list_experiments() -> list[ExperimentSpec]:
    return [REGISTRY[k] for k in sorted(REGISTRY)]
