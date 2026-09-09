"""K0: Kronos market-only sample → quantile forecasts.

Probabilistic path: retain terminal close *samples*, then map to empirical
return-space quantiles. Do not claim a full predictive distribution unless
samples are retained and declared as such.

The official ``KronosPredictor.predict`` averages ``sample_count`` paths
internally. The real backend therefore issues ``sample_count`` single-path
calls so QuantileLedger can keep the sample set.

Default ``ql demo load`` still uses FakeKronosSampler (fast, offline).
Real Kronos: ``uv sync --extra ml`` then ``ql experiment fetch-k0`` (downloads
once into local ``.ql/``; MIT upstream; no API key).

Experiment ID: K0.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import math
import sys
import urllib.error
import urllib.request
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from quantile_ledger.contract import (
    DEFAULT_QUANTILES,
    FEATURE_SET_MARKET_ONLY,
    FEATURE_VERSION_V1,
    FORECAST_SPACE_RETURN,
    TARGET_LOG_RETURN,
    ForecastContract,
    validate_forecast_contract,
)
from quantile_ledger.errors import (
    ForecastValidationError,
    InsufficientDataError,
    ModelUnavailableError,
)
from quantile_ledger.experiments import K0
from quantile_ledger.timeutil import to_iso_utc, utc_now

QUANTILE_METHOD = "empirical_samples_v1"
DISTRIBUTION_CLAIM = "sample_quantile_approx"
FAKE_BACKBONE = "fake_kronos_sampler_v1"
REAL_BACKBONE = "kronos_pretrained_single_path_samples_v1"

# Upstream model IDs (public Hugging Face; no API key).
DEFAULT_TOKENIZER_ID = "NeoQuasar/Kronos-Tokenizer-base"
DEFAULT_MODEL_ID = "NeoQuasar/Kronos-small"

# MIT-licensed architecture sources (cached under local data_dir, not committed).
_KRONOS_SRC_BASE = "https://raw.githubusercontent.com/shiyu-coder/Kronos/master"
_KRONOS_SRC_FILES: dict[str, str] = {
    "model/__init__.py": f"{_KRONOS_SRC_BASE}/model/__init__.py",
    "model/kronos.py": f"{_KRONOS_SRC_BASE}/model/kronos.py",
    "model/module.py": f"{_KRONOS_SRC_BASE}/model/module.py",
    "LICENSE": f"{_KRONOS_SRC_BASE}/LICENSE",
}


@dataclass(frozen=True)
class OhlcBar:
    """One completed OHLC bar (volume optional; missing volume → 0 for Kronos)."""

    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0


@dataclass(frozen=True)
class KronosSampleResult:
    """Retained sample terminal closes plus provenance for the forecast contract."""

    terminal_closes: tuple[float, ...]
    backbone: str
    sample_count: int
    temperature: float
    top_p: float
    lookback: int
    pred_len: int
    model_ref: str
    tokenizer_ref: str
    weights_digest: str | None
    metadata: dict[str, Any]


@dataclass(frozen=True)
class KronosLocalPaths:
    """Local-only cache locations under the user's data_dir."""

    weights_dir: Path
    source_dir: Path

    @classmethod
    def under(cls, data_dir: Path) -> KronosLocalPaths:
        root = data_dir.expanduser().resolve()
        return cls(
            weights_dir=root / "models" / "kronos",
            source_dir=root / "cache" / "kronos_src",
        )


class KronosSampler(Protocol):
    def sample_terminal_closes(
        self,
        bars: Sequence[OhlcBar],
        *,
        bar_ends: Sequence[str],
        future_bar_ends: Sequence[str],
        spot_at_issue: float,
        pred_len: int,
        sample_count: int,
        temperature: float,
        top_p: float,
        seed: int,
    ) -> KronosSampleResult: ...


def closes_to_ohlc_bars(closes: Sequence[float]) -> list[OhlcBar]:
    """Build minimal OHLC from closes when true candles are unavailable.

    Open = previous close (first open = first close). High/low = max/min of
    open and close. Volume = 0 (Kronos accepts missing volume as zero).
    """
    if not closes:
        return []
    bars: list[OhlcBar] = []
    prev = float(closes[0])
    for close in closes:
        c = float(close)
        o = prev
        bars.append(
            OhlcBar(
                open=o,
                high=max(o, c),
                low=min(o, c),
                close=c,
                volume=0.0,
            )
        )
        prev = c
    return bars


def empirical_quantile(sorted_values: Sequence[float], q: float) -> float:
    """Linear-interpolated empirical quantile on a pre-sorted sample."""
    if not sorted_values:
        msg = "empirical quantile requires at least one sample"
        raise InsufficientDataError(msg)
    if not 0.0 < q < 1.0:
        msg = f"quantile level must be in (0,1), got {q}"
        raise ValueError(msg)
    n = len(sorted_values)
    if n == 1:
        return float(sorted_values[0])
    pos = q * (n - 1)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return float(sorted_values[lo])
    weight = pos - lo
    return float(sorted_values[lo]) * (1.0 - weight) + float(sorted_values[hi]) * weight


def terminal_closes_to_return_quantiles(
    terminal_closes: Sequence[float],
    *,
    spot_at_issue: float,
    quantile_levels: Sequence[float],
) -> list[float]:
    """Map retained terminal prices to return-space quantiles via log(P/spot)."""
    if spot_at_issue <= 0:
        msg = "spot_at_issue must be positive"
        raise ForecastValidationError(msg)
    if not terminal_closes:
        msg = "K0 requires at least one retained terminal close sample"
        raise InsufficientDataError(msg)
    if any(p <= 0 for p in terminal_closes):
        msg = "terminal close samples must be positive"
        raise ForecastValidationError(msg)
    returns = sorted(math.log(p / spot_at_issue) for p in terminal_closes)
    return [empirical_quantile(returns, q) for q in quantile_levels]


def kronos_optional_status() -> dict[str, str]:
    """Report K0 stack availability without revealing install paths."""
    return {
        "kronos_fake_adapter": "available",
        "torch": _spec_state("torch"),
        "pandas": _spec_state("pandas"),
        "numpy": _spec_state("numpy"),
        "einops": _spec_state("einops"),
        "huggingface_hub": _spec_state("huggingface_hub"),
        "safetensors": _spec_state("safetensors"),
        "tqdm": _spec_state("tqdm"),
        "kronos_upstream_module": (
            "available"
            if importlib.util.find_spec("model") is not None
            and _module_has_attr("model", "Kronos")
            else "fetch_with_ql_experiment_fetch_k0"
        ),
    }


def _spec_state(name: str) -> str:
    return (
        "available" if importlib.util.find_spec(name) is not None else "not_installed"
    )


def _module_has_attr(module_name: str, attr: str) -> bool:
    try:
        mod = importlib.import_module(module_name)
    except ImportError:
        return False
    return hasattr(mod, attr)


def require_kronos_runtime_deps() -> None:
    """Raise if optional runtime deps for real Kronos inference are missing."""
    required = (
        "torch",
        "pandas",
        "numpy",
        "einops",
        "huggingface_hub",
        "safetensors",
        "tqdm",
    )
    missing = [name for name in required if importlib.util.find_spec(name) is None]
    if missing:
        msg = (
            "K0 real Kronos backend requires optional [ml] dependencies "
            f"(missing: {', '.join(missing)}). "
            "Install with: uv sync --extra ml. "
            "Demo/tests can keep using FakeKronosSampler offline."
        )
        raise ModelUnavailableError(msg)


def kronos_bundle_ready(paths: KronosLocalPaths) -> bool:
    """True when local source + weight snapshots look present."""
    model_init = paths.source_dir / "model" / "__init__.py"
    model_w = paths.weights_dir / "model"
    tok_w = paths.weights_dir / "tokenizer"
    return (
        model_init.is_file()
        and model_w.is_dir()
        and any(model_w.iterdir())
        and tok_w.is_dir()
        and any(tok_w.iterdir())
    )


def _download_url(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(url, timeout=120) as resp:
            data = resp.read()
    except urllib.error.URLError as exc:
        msg = f"failed to download {url}: {exc}"
        raise ModelUnavailableError(msg) from exc
    if not data:
        msg = f"empty download from {url}"
        raise ModelUnavailableError(msg)
    dest.write_bytes(data)


def ensure_kronos_source(source_dir: Path, *, force: bool = False) -> Path:
    """Fetch MIT Kronos architecture sources into local cache (idempotent)."""
    for rel, url in _KRONOS_SRC_FILES.items():
        dest = source_dir / rel
        if force or not dest.is_file() or dest.stat().st_size == 0:
            _download_url(url, dest)
    return source_dir


def ensure_kronos_weights(
    weights_dir: Path,
    *,
    model_id: str = DEFAULT_MODEL_ID,
    tokenizer_id: str = DEFAULT_TOKENIZER_ID,
    force: bool = False,
) -> tuple[Path, Path]:
    """Download public HF weight snapshots into local cache (idempotent)."""
    require_kronos_runtime_deps()
    hub = importlib.import_module("huggingface_hub")
    model_dir = weights_dir / "model"
    tok_dir = weights_dir / "tokenizer"
    if force or not model_dir.is_dir() or not any(model_dir.iterdir()):
        model_dir.mkdir(parents=True, exist_ok=True)
        hub.snapshot_download(repo_id=model_id, local_dir=str(model_dir))
    if force or not tok_dir.is_dir() or not any(tok_dir.iterdir()):
        tok_dir.mkdir(parents=True, exist_ok=True)
        hub.snapshot_download(repo_id=tokenizer_id, local_dir=str(tok_dir))
    return model_dir, tok_dir


def fetch_kronos_assets(
    paths: KronosLocalPaths,
    *,
    model_id: str = DEFAULT_MODEL_ID,
    tokenizer_id: str = DEFAULT_TOKENIZER_ID,
    force: bool = False,
) -> dict[str, str]:
    """Download Kronos source + weights once into local ``.ql`` paths."""
    ensure_kronos_source(paths.source_dir, force=force)
    model_dir, tok_dir = ensure_kronos_weights(
        paths.weights_dir,
        model_id=model_id,
        tokenizer_id=tokenizer_id,
        force=force,
    )
    return {
        "source": (
            "ready"
            if (paths.source_dir / "model" / "__init__.py").is_file()
            else "missing"
        ),
        "model_weights": "ready" if model_dir.is_dir() else "missing",
        "tokenizer_weights": "ready" if tok_dir.is_dir() else "missing",
        "model_id": model_id,
        "tokenizer_id": tokenizer_id,
    }


def _import_kronos_classes(source_dir: Path) -> tuple[Any, Any, Any]:
    """Import upstream Kronos classes from a local source cache."""
    root = str(source_dir.resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    existing = sys.modules.get("model")
    if existing is not None:
        mod_file = getattr(existing, "__file__", "") or ""
        if "kronos_src" not in mod_file.replace("\\", "/"):
            for key in list(sys.modules):
                if key == "model" or key.startswith("model."):
                    del sys.modules[key]
    try:
        kronos_mod = importlib.import_module("model")
        return (
            kronos_mod.Kronos,
            kronos_mod.KronosTokenizer,
            kronos_mod.KronosPredictor,
        )
    except ImportError as exc:
        msg = (
            "Kronos upstream module not importable from local cache. "
            "Run: ql experiment fetch-k0"
        )
        raise ModelUnavailableError(msg) from exc
    except AttributeError as exc:
        msg = "Kronos upstream module missing Kronos/KronosTokenizer/KronosPredictor"
        raise ModelUnavailableError(msg) from exc


class FakeKronosSampler:
    """Deterministic offline sampler for tests and synthetic demo.

    Samples terminal closes from a seeded Gaussian in log-return space using
    recent realized volatility. Not Kronos weights; labeled fake in metadata.
    """

    def __init__(self, *, lookback: int = 64) -> None:
        self.lookback = lookback

    def sample_terminal_closes(
        self,
        bars: Sequence[OhlcBar],
        *,
        bar_ends: Sequence[str],
        future_bar_ends: Sequence[str],
        spot_at_issue: float,
        pred_len: int,
        sample_count: int,
        temperature: float,
        top_p: float,
        seed: int,
    ) -> KronosSampleResult:
        _ = (bar_ends, future_bar_ends, top_p)
        if sample_count < 1:
            msg = "sample_count must be >= 1"
            raise ValueError(msg)
        if pred_len < 1:
            msg = "pred_len must be >= 1"
            raise ValueError(msg)
        if len(bars) < self.lookback:
            msg = f"K0 fake sampler needs lookback {self.lookback}, got {len(bars)}"
            raise InsufficientDataError(msg)
        window = bars[-self.lookback :]
        closes = [b.close for b in window]
        rets: list[float] = []
        for i in range(len(closes) - 1):
            if closes[i] > 0 and closes[i + 1] > 0:
                rets.append(math.log(closes[i + 1] / closes[i]))
        if len(rets) < 2:
            msg = "K0 fake sampler needs at least two positive returns in lookback"
            raise InsufficientDataError(msg)
        mean = sum(rets) / len(rets)
        var = sum((r - mean) ** 2 for r in rets) / (len(rets) - 1)
        vol = math.sqrt(max(var, 1e-12))
        # Scale one-step vol by sqrt(horizon); temperature widens the sample cloud.
        horizon_vol = vol * math.sqrt(pred_len) * max(temperature, 1e-6)
        rng = _RNG(seed)
        terminals: list[float] = []
        for _ in range(sample_count):
            z = rng.gauss()
            r = mean * pred_len + horizon_vol * z
            terminals.append(spot_at_issue * math.exp(r))
        digest = hashlib.sha256(
            json.dumps(
                {
                    "backbone": FAKE_BACKBONE,
                    "lookback": self.lookback,
                    "seed": seed,
                    "sample_count": sample_count,
                    "pred_len": pred_len,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        return KronosSampleResult(
            terminal_closes=tuple(terminals),
            backbone=FAKE_BACKBONE,
            sample_count=sample_count,
            temperature=temperature,
            top_p=top_p,
            lookback=self.lookback,
            pred_len=pred_len,
            model_ref="fake",
            tokenizer_ref="fake",
            weights_digest=digest,
            metadata={
                "is_fake": True,
                "mean_one_step": mean,
                "vol_one_step": vol,
                "horizon_vol": horizon_vol,
            },
        )


class RealKronosSampler:
    """Official Kronos weights via local cache (optional one-time Hub download)."""

    def __init__(
        self,
        *,
        paths: KronosLocalPaths,
        model_id: str = DEFAULT_MODEL_ID,
        tokenizer_id: str = DEFAULT_TOKENIZER_ID,
        max_context: int = 512,
        lookback: int = 400,
        allow_hub_download: bool = True,
        device: str | None = None,
    ) -> None:
        self.paths = paths
        self.model_id = model_id
        self.tokenizer_id = tokenizer_id
        self.max_context = max_context
        self.lookback = lookback
        self.allow_hub_download = allow_hub_download
        self.device = device
        self._predictor: Any | None = None

    def sample_terminal_closes(
        self,
        bars: Sequence[OhlcBar],
        *,
        bar_ends: Sequence[str],
        future_bar_ends: Sequence[str],
        spot_at_issue: float,
        pred_len: int,
        sample_count: int,
        temperature: float,
        top_p: float,
        seed: int,
    ) -> KronosSampleResult:
        _ = spot_at_issue
        require_kronos_runtime_deps()
        if sample_count < 1:
            msg = "sample_count must be >= 1"
            raise ValueError(msg)
        if pred_len < 1:
            msg = "pred_len must be >= 1"
            raise ValueError(msg)
        if len(future_bar_ends) < pred_len:
            msg = "future_bar_ends shorter than pred_len"
            raise InsufficientDataError(msg)
        if len(bars) != len(bar_ends):
            msg = "bars and bar_ends length mismatch"
            raise ValueError(msg)
        if len(bars) < min(self.lookback, 8):
            msg = f"K0 needs sufficient OHLC history, got {len(bars)}"
            raise InsufficientDataError(msg)

        predictor = self._load_predictor()
        pd = importlib.import_module("pandas")
        torch = importlib.import_module("torch")

        torch.manual_seed(seed)
        ctx_len = min(self.lookback, len(bars))
        ctx = bars[-ctx_len:]
        ctx_ts = bar_ends[-ctx_len:]
        df = pd.DataFrame(
            {
                "open": [b.open for b in ctx],
                "high": [b.high for b in ctx],
                "low": [b.low for b in ctx],
                "close": [b.close for b in ctx],
                "volume": [b.volume for b in ctx],
            }
        )
        x_timestamp = pd.to_datetime(pd.Series(list(ctx_ts)), utc=True)
        y_timestamp = pd.to_datetime(
            pd.Series(list(future_bar_ends[:pred_len])), utc=True
        )

        terminals: list[float] = []
        # Official predict() averages sample_count; call once per path to retain.
        for i in range(sample_count):
            torch.manual_seed(seed + i)
            pred_df = predictor.predict(
                df=df,
                x_timestamp=x_timestamp,
                y_timestamp=y_timestamp,
                pred_len=pred_len,
                T=temperature,
                top_p=top_p,
                sample_count=1,
                verbose=False,
            )
            terminals.append(float(pred_df["close"].iloc[-1]))

        return KronosSampleResult(
            terminal_closes=tuple(terminals),
            backbone=REAL_BACKBONE,
            sample_count=sample_count,
            temperature=temperature,
            top_p=top_p,
            lookback=ctx_len,
            pred_len=pred_len,
            model_ref=self.model_id,
            tokenizer_ref=self.tokenizer_id,
            weights_digest=None,
            metadata={
                "is_fake": False,
                "allow_hub_download": self.allow_hub_download,
                "note": (
                    "Quantiles from retained single-path samples; "
                    "not a claimed full predictive density."
                ),
            },
        )

    def _load_predictor(self) -> Any:
        if self._predictor is not None:
            return self._predictor
        if self.allow_hub_download:
            fetch_kronos_assets(
                self.paths,
                model_id=self.model_id,
                tokenizer_id=self.tokenizer_id,
                force=False,
            )
        elif not kronos_bundle_ready(self.paths):
            msg = (
                "Kronos local bundle missing. Run: ql experiment fetch-k0 "
                "(or enable allow_hub_download)."
            )
            raise ModelUnavailableError(msg)

        Kronos, KronosTokenizer, KronosPredictor = _import_kronos_classes(
            self.paths.source_dir
        )
        model_path = self.paths.weights_dir / "model"
        tok_path = self.paths.weights_dir / "tokenizer"
        tokenizer = KronosTokenizer.from_pretrained(str(tok_path))
        model = Kronos.from_pretrained(str(model_path))
        self._predictor = KronosPredictor(
            model, tokenizer, device=self.device, max_context=self.max_context
        )
        return self._predictor


class _RNG:
    def __init__(self, seed: int) -> None:
        self._state = seed % (2**31 - 1)
        if self._state <= 0:
            self._state = 1

    def random(self) -> float:
        self._state = (self._state * 48271) % 2147483647
        return self._state / 2147483647

    def gauss(self) -> float:
        # Box-Muller
        u1 = max(self.random(), 1e-12)
        u2 = self.random()
        return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)


def issue_k0_forecast(
    sampler: KronosSampler,
    *,
    bars: Sequence[OhlcBar],
    bar_ends: Sequence[str],
    future_bar_ends: Sequence[str],
    ticker: str,
    issued_at: str,
    origin_bar_at: str,
    target_at: str,
    spot_at_issue: float,
    horizon_hours: int,
    pred_len: int,
    training_cutoff: str,
    data_as_of: str,
    sample_count: int | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    seed: int = 0,
    is_synthetic: bool = False,
    run_id: str | None = None,
    quantile_levels: Sequence[float] = DEFAULT_QUANTILES,
) -> ForecastContract:
    """Issue a K0 contract from retained Kronos (or fake) samples."""
    hp = K0.hyperparameters
    n_samples = int(sample_count if sample_count is not None else hp["sample_count"])
    temp = float(temperature if temperature is not None else hp["temperature"])
    nucleus = float(top_p if top_p is not None else hp["top_p"])

    result = sampler.sample_terminal_closes(
        bars,
        bar_ends=bar_ends,
        future_bar_ends=future_bar_ends,
        spot_at_issue=spot_at_issue,
        pred_len=pred_len,
        sample_count=n_samples,
        temperature=temp,
        top_p=nucleus,
        seed=seed,
    )
    levels = tuple(quantile_levels)
    values = terminal_closes_to_return_quantiles(
        result.terminal_closes,
        spot_at_issue=spot_at_issue,
        quantile_levels=levels,
    )
    if any(values[i] > values[i + 1] + 1e-9 for i in range(len(values) - 1)):
        msg = "K0 produced crossing quantiles from sample estimates"
        raise ForecastValidationError(msg)

    created = to_iso_utc(utc_now())
    contract = ForecastContract(
        forecast_id=str(uuid.uuid4()),
        run_id=run_id,
        experiment_id=K0.experiment_id,
        ticker=ticker,
        model_family=K0.model_family,
        model_version=f"{K0.experiment_id}-v{K0.version}",
        artifact_digest=result.weights_digest,
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
        quantile_levels=list(levels),
        quantile_values=values,
        forecast_space=FORECAST_SPACE_RETURN,
        random_seed=seed,
        created_at=created,
        is_synthetic=is_synthetic,
        generation_metadata={
            "ticker": ticker,
            "backbone": result.backbone,
            "quantile_method": QUANTILE_METHOD,
            "distribution_claim": DISTRIBUTION_CLAIM,
            "sample_count": result.sample_count,
            "temperature": result.temperature,
            "top_p": result.top_p,
            "lookback": result.lookback,
            "pred_len": result.pred_len,
            "model_ref": result.model_ref,
            "tokenizer_ref": result.tokenizer_ref,
            "experiment_config_hash": K0.config_hash(),
            "sampler_metadata": result.metadata,
            "license": "MIT (upstream Kronos)",
        },
    )
    return validate_forecast_contract(contract)
