"""Configuration tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from quantile_ledger.config import load_settings, save_settings, set_setting
from quantile_ledger.errors import ConfigurationError


def test_defaults_resolve_under_data_dir(tmp_path: Path) -> None:
    settings = load_settings(data_dir=tmp_path / "ql-data")
    assert settings.data_dir == (tmp_path / "ql-data").resolve()
    assert settings.database_path == settings.data_dir / "quantile_ledger.db"
    assert settings.horizons_hours == [1, 6, 24, 72]
    assert 0.5 in settings.quantiles


def test_file_then_cli_override_precedence(tmp_path: Path) -> None:
    data_dir = tmp_path / "ql-data"
    settings = load_settings(data_dir=data_dir)
    settings = settings.model_copy(update={"min_metric_samples": 10})
    save_settings(settings)

    from_file = load_settings(data_dir=data_dir)
    assert from_file.min_metric_samples == 10

    overridden = load_settings(
        data_dir=data_dir,
        cli_overrides={"min_metric_samples": 99},
    )
    assert overridden.min_metric_samples == 99


def test_set_setting_rejects_secrets(tmp_path: Path) -> None:
    settings = load_settings(data_dir=tmp_path / "ql-data")
    with pytest.raises(ConfigurationError, match="not supported"):
        set_setting(settings, "api_key", "should-fail")
