"""Experiment registry identities (research matrix)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Literal

from quantile_ledger.contract import (
    DEFAULT_QUANTILES,
    FEATURE_SET_MARKET_ONLY,
    FEATURE_SET_MARKET_PLUS_NEWS,
    FEATURE_SET_MARKET_PLUS_NEWS_VOLUME,
    FEATURE_SET_NEWS_ONLY,
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

# Phase G controlled feature ladder (frozen linear news shifts on parent models).
_DEFAULT_NEWS_LOOKBACK_HOURS = 72
_DEFAULT_VOLUME_WEIGHT = 0.0005
_DEFAULT_SENTIMENT_WEIGHT = 0.0020

T1 = ExperimentSpec(
    experiment_id="T1",
    name="tft_plus_news_volume",
    version="1",
    model_family="tft_quantile",
    feature_set=FEATURE_SET_MARKET_PLUS_NEWS_VOLUME,
    feature_version=FEATURE_VERSION_V1,
    target_definition=TARGET_LOG_RETURN,
    status="candidate",
    purpose=(
        "T0 parent TFT quantiles plus frozen news-volume shift only "
        "(log1p count in PIT lookback). Missing news refuses issuance."
    ),
    quantiles=DEFAULT_QUANTILES,
    horizons_hours=DEFAULT_HORIZONS,
    hyperparameters={
        "parent_experiment_id": "T0",
        "news_lookback_hours": _DEFAULT_NEWS_LOOKBACK_HOURS,
        "volume_weight": _DEFAULT_VOLUME_WEIGHT,
        "sentiment_weight": 0.0,
        "missing_policy": "refuse",
        "adjustment": "additive_return_shift_isotonic_v1",
    },
    label="exploratory",
)

T2 = ExperimentSpec(
    experiment_id="T2",
    name="tft_plus_news_volume_finbert",
    version="1",
    model_family="tft_quantile",
    feature_set=FEATURE_SET_MARKET_PLUS_NEWS,
    feature_version=FEATURE_VERSION_V1,
    target_definition=TARGET_LOG_RETURN,
    status="candidate",
    purpose=(
        "T0 parent TFT quantiles plus news volume and FinBERT polarity "
        "(pos-neg). Requires fully scored context; missing/partial refuses."
    ),
    quantiles=DEFAULT_QUANTILES,
    horizons_hours=DEFAULT_HORIZONS,
    hyperparameters={
        "parent_experiment_id": "T0",
        "news_lookback_hours": _DEFAULT_NEWS_LOOKBACK_HOURS,
        "volume_weight": _DEFAULT_VOLUME_WEIGHT,
        "sentiment_weight": _DEFAULT_SENTIMENT_WEIGHT,
        "missing_policy": "refuse",
        "require_full_scores": True,
        "adjustment": "additive_return_shift_isotonic_v1",
    },
    label="exploratory",
)

M1 = ExperimentSpec(
    experiment_id="M1",
    name="mamba_plus_news_volume_finbert",
    version="1",
    model_family="mamba_quantile",
    feature_set=FEATURE_SET_MARKET_PLUS_NEWS,
    feature_version=FEATURE_VERSION_V1,
    target_definition=TARGET_LOG_RETURN,
    status="candidate",
    purpose=(
        "M0 parent MambaQuantile plus news volume and FinBERT polarity. "
        "Missing/partial news refuses issuance (never imputed as neutral)."
    ),
    quantiles=DEFAULT_QUANTILES,
    horizons_hours=DEFAULT_HORIZONS,
    hyperparameters={
        "parent_experiment_id": "M0",
        "news_lookback_hours": _DEFAULT_NEWS_LOOKBACK_HOURS,
        "volume_weight": _DEFAULT_VOLUME_WEIGHT,
        "sentiment_weight": _DEFAULT_SENTIMENT_WEIGHT,
        "missing_policy": "refuse",
        "require_full_scores": True,
        "adjustment": "additive_return_shift_isotonic_v1",
    },
    label="exploratory",
)

N0 = ExperimentSpec(
    experiment_id="N0",
    name="news_only_finbert_baseline",
    version="2",
    model_family="news_only_baseline",
    feature_set=FEATURE_SET_NEWS_ONLY,
    feature_version=FEATURE_VERSION_V1,
    target_definition=TARGET_LOG_RETURN,
    status="candidate",
    purpose=(
        "News-only probabilistic baseline: median shift from FinBERT polarity "
        "plus residual spread from volume; no market returns. Missing news "
        "refuses issuance. Compares to B1 under the same outcome pairing."
    ),
    quantiles=DEFAULT_QUANTILES,
    horizons_hours=DEFAULT_HORIZONS,
    hyperparameters={
        "news_lookback_hours": _DEFAULT_NEWS_LOOKBACK_HOURS,
        "sentiment_to_p50": _DEFAULT_SENTIMENT_WEIGHT,
        "volume_to_width": 0.0010,
        "base_width": 0.0100,
        "missing_policy": "refuse",
        "require_full_scores": True,
        "scorer_default": "fake_finbert_v1",
    },
    label="exploratory",
)

# Phase H: isotonic PIT recalibration of market challengers (separate rows).
_CALIBRATION_VARIANT = "isotonic_v1"
_CALIBRATION_METHOD = "isotonic_pit"
_CALIBRATION_VERSION = "v1"

M0c = ExperimentSpec(
    experiment_id="M0c",
    name="mamba_quantile_isotonic_pit",
    version="1",
    model_family="mamba_quantile",
    feature_set=FEATURE_SET_MARKET_ONLY,
    feature_version=FEATURE_VERSION_V1,
    target_definition=TARGET_LOG_RETURN,
    status="candidate",
    purpose=(
        "M0 parent with isotonic PIT recalibration fit on validation issuances "
        "only; eval-window children only. Raw M0 rows stay immutable."
    ),
    quantiles=DEFAULT_QUANTILES,
    horizons_hours=DEFAULT_HORIZONS,
    hyperparameters={
        "parent_experiment_id": "M0",
        "calibration_method": _CALIBRATION_METHOD,
        "calibration_version": _CALIBRATION_VERSION,
        "variant": _CALIBRATION_VARIANT,
        "fit_period": "validation_only",
    },
    label="exploratory",
)

T0c = ExperimentSpec(
    experiment_id="T0c",
    name="tft_quantile_isotonic_pit",
    version="1",
    model_family="tft_quantile",
    feature_set=FEATURE_SET_MARKET_ONLY,
    feature_version=FEATURE_VERSION_V1,
    target_definition=TARGET_LOG_RETURN,
    status="candidate",
    purpose=(
        "T0 parent with isotonic PIT recalibration fit on validation issuances "
        "only; eval-window children only. Raw T0 rows stay immutable."
    ),
    quantiles=DEFAULT_QUANTILES,
    horizons_hours=DEFAULT_HORIZONS,
    hyperparameters={
        "parent_experiment_id": "T0",
        "calibration_method": _CALIBRATION_METHOD,
        "calibration_version": _CALIBRATION_VERSION,
        "variant": _CALIBRATION_VARIANT,
        "fit_period": "validation_only",
    },
    label="exploratory",
)

K0c = ExperimentSpec(
    experiment_id="K0c",
    name="kronos_isotonic_pit",
    version="1",
    model_family="kronos",
    feature_set=FEATURE_SET_MARKET_ONLY,
    feature_version=FEATURE_VERSION_V1,
    target_definition=TARGET_LOG_RETURN,
    status="candidate",
    purpose=(
        "K0 parent with isotonic PIT recalibration fit on validation issuances "
        "only; eval-window children only. Raw K0 rows stay immutable."
    ),
    quantiles=DEFAULT_QUANTILES,
    horizons_hours=DEFAULT_HORIZONS,
    hyperparameters={
        "parent_experiment_id": "K0",
        "calibration_method": _CALIBRATION_METHOD,
        "calibration_version": _CALIBRATION_VERSION,
        "variant": _CALIBRATION_VARIANT,
        "fit_period": "validation_only",
        "distribution_claim": "sample_quantile_approx",
    },
    label="exploratory",
)

PARENT_TO_CALIBRATED: dict[str, str] = {
    "M0": "M0c",
    "T0": "T0c",
    "K0": "K0c",
}

REGISTRY: dict[str, ExperimentSpec] = {
    B0.experiment_id: B0,
    B1.experiment_id: B1,
    M0.experiment_id: M0,
    M0c.experiment_id: M0c,
    M1.experiment_id: M1,
    K0.experiment_id: K0,
    K0c.experiment_id: K0c,
    T0.experiment_id: T0,
    T0c.experiment_id: T0c,
    T1.experiment_id: T1,
    T2.experiment_id: T2,
    N0.experiment_id: N0,
}


def get_experiment(experiment_id: str) -> ExperimentSpec:
    try:
        return REGISTRY[experiment_id]
    except KeyError as exc:
        msg = f"unknown experiment_id: {experiment_id}"
        raise KeyError(msg) from exc


def list_experiments() -> list[ExperimentSpec]:
    return [REGISTRY[k] for k in sorted(REGISTRY)]
