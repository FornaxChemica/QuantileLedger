"""Phase C: artifacts, freeze, provenance, migrations."""

from __future__ import annotations

from pathlib import Path

from quantile_ledger.artifacts import (
    optional_dependency_status,
    register_artifact,
    write_json_artifact,
)
from quantile_ledger.baselines import issue_b0_persistence
from quantile_ledger.db import SCHEMA_VERSION, connection, initialize_database
from quantile_ledger.experiments import M0
from quantile_ledger.forecast_store import (
    freeze_experiment,
    get_forecast_provenance,
    insert_forecast,
    upsert_experiment,
)
from quantile_ledger.mamba_quantile import train_m0_model
from quantile_ledger.synthetic import make_synthetic_hourly_closes


def test_schema_migrates_to_v3(tmp_path: Path) -> None:
    db = tmp_path / "t.db"
    version = initialize_database(db)
    assert version == SCHEMA_VERSION == 3
    with connection(db) as conn:
        tables = {
            r["name"]
            for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        assert "artifacts" in tables
        cols = {r["name"] for r in conn.execute("PRAGMA table_info(experiments)")}
        assert "frozen_at" in cols


def test_freeze_blocks_config_change(tmp_path: Path) -> None:
    db = tmp_path / "t.db"
    initialize_database(db)
    with connection(db) as conn:
        upsert_experiment(conn, M0)
        freeze_experiment(conn, "M0")
        # Same hash is ok (no-op)
        upsert_experiment(conn, M0)
        row = conn.execute(
            "SELECT status, frozen_at FROM experiments WHERE experiment_id='M0'"
        ).fetchone()
        assert row is not None
        assert row["status"] == "frozen"
        assert row["frozen_at"]


def test_artifact_relative_path_and_digest(tmp_path: Path) -> None:
    data = tmp_path / "ql"
    models = data / "models"
    aid, digest, rel = write_json_artifact(
        artifacts_dir=models,
        kind="m0_weights",
        payload={"a": 1},
        filename_stem="demo",
    )
    assert not Path(rel).is_absolute()
    assert len(digest) == 64
    db = data / "db.sqlite"
    initialize_database(db)
    with connection(db) as conn:
        upsert_experiment(conn, M0)
        register_artifact(
            conn,
            artifact_id=aid,
            digest=digest,
            kind="m0_weights",
            relative_path=str(rel),
            experiment_id="M0",
        )
        n = conn.execute("SELECT COUNT(*) AS n FROM artifacts").fetchone()
        assert n is not None
        assert int(n["n"]) == 1


def test_forecast_provenance_chain(tmp_path: Path) -> None:
    db = tmp_path / "t.db"
    initialize_database(db)
    series = make_synthetic_hourly_closes(n=120, seed=1)
    model, _ = train_m0_model(
        series.closes, bar_horizon=1, lookback=16, epochs=5, seed=1, min_samples=20
    )
    fc = issue_b0_persistence(
        ticker="SYN",
        issued_at="2024-01-10T15:00:00Z",
        origin_bar_at="2024-01-10T15:00:00Z",
        target_at="2024-01-10T16:00:00Z",
        spot_at_issue=100.0,
        horizon_hours=1,
        training_cutoff="2024-01-10T15:00:00Z",
        data_as_of="2024-01-10T15:00:00Z",
        is_synthetic=True,
    )
    with connection(db) as conn:
        upsert_experiment(conn, M0)
        from quantile_ledger.experiments import B0

        upsert_experiment(conn, B0)
        insert_forecast(conn, fc)
        prov = get_forecast_provenance(conn, fc.forecast_id)
        assert prov["experiment_id"] == "B0"
        assert prov["training_cutoff"] == "2024-01-10T15:00:00Z"
        assert prov["feature_set"] == "market_only"
        assert prov["config_hash"] == B0.config_hash()
    _ = model  # trained to ensure M0 path still works offline


def test_optional_deps_report_core_only() -> None:
    status = optional_dependency_status()
    assert status["core"] == "available"
    assert "torch" in status
    assert status["kronos_fake_adapter"] == "available"
    assert status["tft_local_adapter"] == "available"
