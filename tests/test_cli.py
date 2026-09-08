"""CLI smoke tests for Milestone 0 commands."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from quantile_ledger.cli import app

runner = CliRunner()


def test_help_exits_zero() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "QuantileLedger" in result.stdout


def test_init_doctor_watch_roundtrip(tmp_path: Path) -> None:
    data_dir = tmp_path / "ql-data"
    init = runner.invoke(app, ["--data-dir", str(data_dir), "init"])
    assert init.exit_code == 0, init.stdout + init.stderr
    assert "Initialized" in init.stdout

    doctor = runner.invoke(app, ["--data-dir", str(data_dir), "doctor"])
    assert doctor.exit_code == 0, doctor.stdout + doctor.stderr
    assert "paper-only" in doctor.stdout
    assert "not supported" in doctor.stdout

    add = runner.invoke(app, ["--data-dir", str(data_dir), "watch", "add", "tlt"])
    assert add.exit_code == 0, add.stdout + add.stderr

    listed = runner.invoke(app, ["--data-dir", str(data_dir), "watch", "list"])
    assert listed.exit_code == 0
    assert "TLT" in listed.stdout
    assert "SPY" in listed.stdout  # seeded default

    removed = runner.invoke(
        app, ["--data-dir", str(data_dir), "watch", "remove", "TLT"]
    )
    assert removed.exit_code == 0
    listed_after = runner.invoke(app, ["--data-dir", str(data_dir), "watch", "list"])
    assert "TLT" not in listed_after.stdout


def test_milestone_stub_nonzero_exit(tmp_path: Path) -> None:
    data_dir = tmp_path / "ql-data"
    runner.invoke(app, ["--data-dir", str(data_dir), "init"])
    result = runner.invoke(app, ["--data-dir", str(data_dir), "demo", "load"])
    assert result.exit_code == 2
    assert "Milestone 1" in result.stdout + result.stderr


def test_config_show_and_set(tmp_path: Path) -> None:
    data_dir = tmp_path / "ql-data"
    runner.invoke(app, ["--data-dir", str(data_dir), "init"])
    shown = runner.invoke(app, ["--data-dir", str(data_dir), "config", "show"])
    assert shown.exit_code == 0
    assert "min_metric_samples" in shown.stdout

    updated = runner.invoke(
        app,
        ["--data-dir", str(data_dir), "config", "set", "min_metric_samples", "42"],
    )
    assert updated.exit_code == 0, updated.stdout + updated.stderr
    shown_again = runner.invoke(app, ["--data-dir", str(data_dir), "config", "show"])
    assert "42" in shown_again.stdout
