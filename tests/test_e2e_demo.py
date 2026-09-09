"""End-to-end synthetic demo: B0/B1/M0 issue, settle, pair."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from quantile_ledger.cli import app

runner = CliRunner()


def test_demo_load_and_compare(tmp_path: Path) -> None:
    data_dir = tmp_path / "ql"
    init = runner.invoke(app, ["--data-dir", str(data_dir), "init"])
    assert init.exit_code == 0, init.stdout + init.stderr
    demo = runner.invoke(app, ["--data-dir", str(data_dir), "demo", "load"])
    assert demo.exit_code == 0, demo.stdout + demo.stderr
    assert "synthetic" in demo.stdout.lower()
    compare = runner.invoke(app, ["--data-dir", str(data_dir), "compare"])
    assert compare.exit_code == 0, compare.stdout + compare.stderr
    assert "paired_n=" in compare.stdout
    assert "B1" in compare.stdout
    assert "M0" in compare.stdout
    experiments = runner.invoke(
        app, ["--data-dir", str(data_dir), "experiment", "list"]
    )
    assert experiments.exit_code == 0
    assert "M0" in experiments.stdout
    assert "Milestone 0" in experiments.stdout
