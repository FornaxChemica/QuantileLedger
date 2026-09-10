"""Typed non-secret local configuration for QuantileLedger."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import Field, ValidationError, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from quantile_ledger.errors import ConfigurationError

DEFAULT_HORIZONS_HOURS: tuple[int, ...] = (1, 6, 24, 72)
DEFAULT_QUANTILES: tuple[float, ...] = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)
DEFAULT_WATCHLIST: tuple[str, ...] = ("SPY", "QQQ", "IWM", "AAPL", "MSFT")

SETTINGS_FILE_NAME = "settings.json"


class Settings(BaseSettings):
    """Local application settings. No API keys or secrets."""

    model_config = SettingsConfigDict(
        env_prefix="QL_",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    data_dir: Path = Field(default_factory=lambda: Path.cwd() / ".ql")
    database_path: Path | None = None
    cache_dir: Path | None = None
    model_dir: Path | None = None
    import_dir: Path | None = None
    report_dir: Path | None = None
    log_dir: Path | None = None

    watchlist: list[str] = Field(default_factory=lambda: list(DEFAULT_WATCHLIST))
    horizons_hours: list[int] = Field(
        default_factory=lambda: list(DEFAULT_HORIZONS_HOURS)
    )
    quantiles: list[float] = Field(default_factory=lambda: list(DEFAULT_QUANTILES))

    bar_freshness_hours: int = 36
    quote_max_age_seconds: int = 300
    max_option_spread_pct: float = 0.25
    paper_starting_cash: str = "100000.00"
    paper_per_trade_premium_limit: str = "500.00"
    paper_total_premium_limit: str = "5000.00"
    paper_commission_per_contract: str = "0.65"
    paper_slippage_per_contract: str = "0.05"
    # Equity long/flat paper costs (Phase J1). Option keys unused until J2.
    paper_equity_half_spread_bps: str = "5"
    paper_equity_slippage_bps: str = "2"
    paper_equity_commission_per_share: str = "0.005"
    paper_equity_shares_per_entry: str = "10"
    paper_equity_max_notional_per_trade: str = "5000.00"
    paper_min_p50_log_return: str = "0.0010"
    mechanical_dte_min: int = 7
    mechanical_dte_max: int = 21
    min_metric_samples: int = 30
    log_level: str = "INFO"

    @field_validator("watchlist", mode="before")
    @classmethod
    def _normalize_watchlist(cls, value: Any) -> list[str]:
        if isinstance(value, str):
            parts = [part.strip().upper() for part in value.split(",") if part.strip()]
            return parts
        if isinstance(value, list):
            return [str(item).strip().upper() for item in value if str(item).strip()]
        msg = "watchlist must be a list or comma-separated string"
        raise ValueError(msg)

    @field_validator("horizons_hours", mode="before")
    @classmethod
    def _normalize_horizons(cls, value: Any) -> list[int]:
        if isinstance(value, str):
            return [int(part.strip()) for part in value.split(",") if part.strip()]
        if isinstance(value, list):
            return [int(item) for item in value]
        msg = "horizons_hours must be a list or comma-separated string"
        raise ValueError(msg)

    @field_validator("quantiles", mode="before")
    @classmethod
    def _normalize_quantiles(cls, value: Any) -> list[float]:
        if isinstance(value, str):
            return [float(part.strip()) for part in value.split(",") if part.strip()]
        if isinstance(value, list):
            return [float(item) for item in value]
        msg = "quantiles must be a list or comma-separated string"
        raise ValueError(msg)

    def resolve_paths(self) -> Settings:
        """Fill derived paths from data_dir when unset."""
        data_dir = self.data_dir.expanduser().resolve()
        updates: dict[str, Path] = {"data_dir": data_dir}
        if self.database_path is None:
            updates["database_path"] = data_dir / "quantile_ledger.db"
        else:
            updates["database_path"] = self.database_path.expanduser().resolve()
        if self.cache_dir is None:
            updates["cache_dir"] = data_dir / "cache"
        else:
            updates["cache_dir"] = self.cache_dir.expanduser().resolve()
        if self.model_dir is None:
            updates["model_dir"] = data_dir / "models"
        else:
            updates["model_dir"] = self.model_dir.expanduser().resolve()
        if self.import_dir is None:
            updates["import_dir"] = data_dir / "imports"
        else:
            updates["import_dir"] = self.import_dir.expanduser().resolve()
        if self.report_dir is None:
            updates["report_dir"] = data_dir / "reports"
        else:
            updates["report_dir"] = self.report_dir.expanduser().resolve()
        if self.log_dir is None:
            updates["log_dir"] = data_dir / "logs"
        else:
            updates["log_dir"] = self.log_dir.expanduser().resolve()
        return self.model_copy(update=updates)

    def settings_file(self) -> Path:
        return self.data_dir.expanduser().resolve() / SETTINGS_FILE_NAME

    def ensure_directories(self) -> None:
        resolved = self.resolve_paths()
        assert resolved.database_path is not None
        for path in (
            resolved.data_dir,
            resolved.cache_dir,
            resolved.model_dir,
            resolved.import_dir,
            resolved.report_dir,
            resolved.log_dir,
            resolved.database_path.parent,
        ):
            assert path is not None
            path.mkdir(parents=True, exist_ok=True)


def _settings_from_mapping(data: dict[str, Any], *, data_dir: Path | None) -> Settings:
    payload = dict(data)
    if data_dir is not None:
        payload["data_dir"] = data_dir
    return Settings.model_validate(payload).resolve_paths()


def load_settings(
    *,
    data_dir: Path | None = None,
    cli_overrides: dict[str, Any] | None = None,
) -> Settings:
    """Load settings with precedence: CLI overrides > file > defaults/env."""
    base_dir = (data_dir or Path.cwd() / ".ql").expanduser().resolve()
    file_path = base_dir / SETTINGS_FILE_NAME
    file_data: dict[str, Any] = {}
    if file_path.exists():
        try:
            loaded = json.loads(file_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            msg = f"Invalid settings file {file_path}: {exc}"
            raise ConfigurationError(msg) from exc
        if not isinstance(loaded, dict):
            msg = f"Settings file must contain a JSON object: {file_path}"
            raise ConfigurationError(msg)
        file_data = loaded

    merged: dict[str, Any] = {**file_data}
    if data_dir is not None:
        merged["data_dir"] = data_dir
    if cli_overrides:
        merged.update({k: v for k, v in cli_overrides.items() if v is not None})
    return _settings_from_mapping(merged, data_dir=data_dir)


def save_settings(settings: Settings) -> Path:
    """Persist non-secret settings to the local settings file."""
    resolved = settings.resolve_paths()
    resolved.ensure_directories()
    path = resolved.settings_file()
    payload = resolved.model_dump(mode="json")
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path


def set_setting(settings: Settings, key: str, value: str) -> Settings:
    """Set a single configuration key from a CLI string value."""
    if key in {"api_key", "api-key", "secret", "token", "password"}:
        msg = "API keys and secrets are not supported by QuantileLedger."
        raise ConfigurationError(msg)
    if key not in Settings.model_fields:
        msg = f"Unknown configuration key: {key}"
        raise ConfigurationError(msg)
    try:
        updated = Settings.model_validate({**settings.model_dump(), key: value})
    except ValidationError as exc:
        msg = f"Invalid value for {key}: {exc}"
        raise ConfigurationError(msg) from exc
    updated = updated.resolve_paths()
    save_settings(updated)
    return updated
