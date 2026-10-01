"""Phase WF: bars import, PIT, Stooq cache fetch, walk-forward, gate."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from quantile_ledger.bars import (
    DAILY_GATE_HORIZONS,
    PROVIDER_FIXTURE,
    horizon_hours_to_bar_horizon,
    import_bars_file,
    list_closes_as_of,
    parse_stooq_csv,
)
from quantile_ledger.cli import app
from quantile_ledger.db import connection, initialize_database
from quantile_ledger.errors import MalformedInputError, ModelUnavailableError
from quantile_ledger.gate import evaluate_gate
from quantile_ledger.kronos_quantile import KronosLocalPaths
from quantile_ledger.paper import (
    default_b1_control_policy,
    default_underlying_policy,
    freeze_policy,
    insert_policy,
    upsert_paper_account,
)
from quantile_ledger.providers import fetch_bars, stooq_cache_path
from quantile_ledger.walk_forward import run_bars_walk_forward, run_paper_gate_books

FIXTURES = Path(__file__).parent / "fixtures"
SPY_JSON = FIXTURES / "spy_daily_120.json"
STOOQ_CSV = FIXTURES / "stooq_spy_1d.csv"

runner = CliRunner()


def test_horizon_mapping_daily() -> None:
    assert horizon_hours_to_bar_horizon(24, "1d") == 1
    assert horizon_hours_to_bar_horizon(72, "1d") == 3
    with pytest.raises(MalformedInputError):
        horizon_hours_to_bar_horizon(6, "1d")
    assert 24 in DAILY_GATE_HORIZONS


def test_import_bars_idempotent_and_pit(tmp_path: Path) -> None:
    db = tmp_path / "b.db"
    initialize_database(db)
    with connection(db) as conn:
        r1 = import_bars_file(
            conn,
            SPY_JSON,
            default_interval="1d",
            default_provider=PROVIDER_FIXTURE,
            force_synthetic=False,
        )
        assert r1.inserted == 120
        assert r1.updated == 0
        r2 = import_bars_file(
            conn,
            SPY_JSON,
            default_interval="1d",
            default_provider=PROVIDER_FIXTURE,
            force_synthetic=False,
        )
        assert r2.inserted == 0
        assert r2.updated == 120

        mid = "2023-03-15T21:00:00Z"
        series = list_closes_as_of(
            conn,
            ticker="SPY",
            interval="1d",
            as_of=mid,
            provider=PROVIDER_FIXTURE,
        )
        assert series.bar_ends
        assert all(be <= mid for be in series.bar_ends)
        assert len(series.closes) == len(series.bar_ends)
        assert series.is_synthetic is False


def test_stooq_parse_and_cached_fetch(tmp_path: Path) -> None:
    text = STOOQ_CSV.read_text(encoding="utf-8")
    records = parse_stooq_csv(text, ticker="SPY")
    assert len(records) == 120
    assert records[0].provider == "stooq"
    assert records[0].interval == "1d"
    assert records[0].is_synthetic is False

    db = tmp_path / "f.db"
    cache = tmp_path / "cache"
    initialize_database(db)
    csv_text = text

    def fake_download(url: str) -> str:
        assert "stooq.com" in url
        return csv_text

    with connection(db) as conn:
        result = fetch_bars(
            conn,
            "SPY",
            cache_dir=cache,
            force=True,
            download_fn=fake_download,
        )
        assert result.from_cache is False
        assert result.import_result.total_rows == 120
        assert stooq_cache_path(cache, "SPY").is_file()

        # Fresh cache: no network call (download_fn would still work but from_cache).
        result2 = fetch_bars(
            conn,
            "SPY",
            cache_dir=cache,
            force=False,
            freshness_hours=10_000,
            download_fn=lambda _u: (_ for _ in ()).throw(RuntimeError("network")),
        )
        assert result2.from_cache is True


def test_walk_forward_and_gate_inconclusive(tmp_path: Path) -> None:
    db = tmp_path / "wf.db"
    initialize_database(db)
    with connection(db) as conn:
        import_bars_file(
            conn,
            SPY_JSON,
            default_provider=PROVIDER_FIXTURE,
            force_synthetic=False,
        )
        wf = run_bars_walk_forward(
            conn,
            ticker="SPY",
            interval="1d",
            horizon_hours=24,
            challenger_experiment_id="M0",
            provider=PROVIDER_FIXTURE,
            lookback=16,
            train_epochs=3,
            max_issues=5,
            min_b1_samples=20,
            min_model_samples=20,
            exclude_synthetic=True,
        )
        assert wf.issued == 10  # 5 issues * (B1 + M0)
        assert wf.settled == 10
        assert wf.is_synthetic_cohort is False

        report = evaluate_gate(
            conn,
            challenger_experiment_id="M0",
            min_metric_samples=30,
            ticker="SPY",
        )
        assert report.status == "inconclusive"
        assert report.paired_n == 5
        assert report.synthetic_excluded == 0


def test_walk_forward_k0_fake_backend(tmp_path: Path) -> None:
    db = tmp_path / "k0wf.db"
    initialize_database(db)
    with connection(db) as conn:
        import_bars_file(
            conn,
            SPY_JSON,
            default_provider=PROVIDER_FIXTURE,
            force_synthetic=False,
        )
        wf = run_bars_walk_forward(
            conn,
            ticker="SPY",
            interval="1d",
            horizon_hours=24,
            challenger_experiment_id="K0",
            provider=PROVIDER_FIXTURE,
            lookback=16,
            max_issues=5,
            min_b1_samples=20,
            min_model_samples=20,
            exclude_synthetic=True,
            kronos_backend="fake",
            kronos_paths=KronosLocalPaths.under(tmp_path / "ql"),
        )
        assert wf.challenger_experiment_id == "K0"
        assert wf.issued == 10  # 5 issues * (B1 + K0)
        assert wf.settled == 10


def test_walk_forward_k0_real_backend_refuses_without_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Selecting the real K0 backend without local weights fails clearly."""
    import quantile_ledger.kronos_quantile as kq

    # Stub deps-present so the refusal is specifically about the missing bundle
    # (deterministic whether or not the optional [ml] extra is installed).
    monkeypatch.setattr(kq, "require_kronos_runtime_deps", lambda: None)
    monkeypatch.setattr(kq, "kronos_bundle_ready", lambda _paths: False)

    db = tmp_path / "k0real.db"
    initialize_database(db)
    with connection(db) as conn:
        import_bars_file(
            conn,
            SPY_JSON,
            default_provider=PROVIDER_FIXTURE,
            force_synthetic=False,
        )
        with pytest.raises(ModelUnavailableError, match="fetch-k0"):
            run_bars_walk_forward(
                conn,
                ticker="SPY",
                interval="1d",
                horizon_hours=24,
                challenger_experiment_id="K0",
                provider=PROVIDER_FIXTURE,
                lookback=16,
                max_issues=3,
                min_b1_samples=20,
                min_model_samples=20,
                exclude_synthetic=True,
                kronos_backend="real",
                kronos_paths=KronosLocalPaths.under(tmp_path / "ql"),
            )


def test_walk_forward_k0_real_backend_requires_paths(tmp_path: Path) -> None:
    db = tmp_path / "k0nopaths.db"
    initialize_database(db)
    with connection(db) as conn:
        import_bars_file(
            conn,
            SPY_JSON,
            default_provider=PROVIDER_FIXTURE,
            force_synthetic=False,
        )
        with pytest.raises(MalformedInputError, match="kronos_paths"):
            run_bars_walk_forward(
                conn,
                ticker="SPY",
                interval="1d",
                horizon_hours=24,
                challenger_experiment_id="K0",
                provider=PROVIDER_FIXTURE,
                lookback=16,
                max_issues=3,
                min_b1_samples=20,
                min_model_samples=20,
                exclude_synthetic=True,
                kronos_backend="real",
                kronos_paths=None,
            )


def test_fit_once_trains_exactly_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """fit_once must call the trainer exactly once, not once per issuance."""
    import quantile_ledger.walk_forward as wf_mod

    calls = {"n": 0}
    real_train = wf_mod.train_t0_model

    def counting_train(*args: object, **kwargs: object) -> object:
        calls["n"] += 1
        return real_train(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(wf_mod, "train_t0_model", counting_train)

    db = tmp_path / "fitonce.db"
    initialize_database(db)
    with connection(db) as conn:
        import_bars_file(
            conn,
            SPY_JSON,
            default_provider=PROVIDER_FIXTURE,
            force_synthetic=False,
        )
        wf = wf_mod.run_bars_walk_forward(
            conn,
            ticker="SPY",
            interval="1d",
            horizon_hours=24,
            challenger_experiment_id="T0",
            provider=PROVIDER_FIXTURE,
            lookback=16,
            train_epochs=3,
            max_issues=5,
            min_b1_samples=20,
            min_model_samples=20,
            exclude_synthetic=True,
            training_mode="fit_once",
            model_dir=tmp_path / "models",
        )
    assert wf.issued == 10  # 5 * (B1 + T0)
    assert wf.settled == 10
    assert calls["n"] == 1, f"fit_once trained {calls['n']} times, expected 1"


def test_per_issuance_trains_each_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """per_issuance (default) retrains per issuance — the contrast to fit_once."""
    import quantile_ledger.walk_forward as wf_mod

    calls = {"n": 0}
    real_train = wf_mod.train_t0_model

    def counting_train(*args: object, **kwargs: object) -> object:
        calls["n"] += 1
        return real_train(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(wf_mod, "train_t0_model", counting_train)

    db = tmp_path / "periss.db"
    initialize_database(db)
    with connection(db) as conn:
        import_bars_file(
            conn,
            SPY_JSON,
            default_provider=PROVIDER_FIXTURE,
            force_synthetic=False,
        )
        wf_mod.run_bars_walk_forward(
            conn,
            ticker="SPY",
            interval="1d",
            horizon_hours=24,
            challenger_experiment_id="T0",
            provider=PROVIDER_FIXTURE,
            lookback=16,
            train_epochs=3,
            max_issues=5,
            min_b1_samples=20,
            min_model_samples=20,
            exclude_synthetic=True,
            training_mode="per_issuance",
        )
    assert calls["n"] == 5, f"per_issuance trained {calls['n']} times, expected 5"


def test_fit_once_pit_safe_and_monotone(tmp_path: Path) -> None:
    """Every fit_once forecast has training_cutoff <= issued_at and no crossing."""
    db = tmp_path / "fitpit.db"
    initialize_database(db)
    with connection(db) as conn:
        import_bars_file(
            conn,
            SPY_JSON,
            default_provider=PROVIDER_FIXTURE,
            force_synthetic=False,
        )
        run_bars_walk_forward(
            conn,
            ticker="SPY",
            interval="1d",
            horizon_hours=24,
            challenger_experiment_id="M0",
            provider=PROVIDER_FIXTURE,
            lookback=16,
            train_epochs=3,
            max_issues=5,
            min_b1_samples=20,
            min_model_samples=20,
            exclude_synthetic=True,
            training_mode="fit_once",
            model_dir=tmp_path / "models",
        )
        rows = conn.execute(
            """
            SELECT f.forecast_id, f.training_cutoff, f.issued_at, f.target_at
            FROM forecasts f WHERE f.experiment_id = 'M0'
            """,
        ).fetchall()
        assert rows, "no M0 forecasts issued"
        for row in rows:
            assert row["training_cutoff"] <= row["issued_at"], "PIT violation"
            assert row["target_at"] > row["issued_at"], "target not strictly future"
            # Quantiles live in forecast_quantiles; verify non-crossing per row.
            qvals = [
                r["return_value"]
                for r in conn.execute(
                    "SELECT return_value FROM forecast_quantiles "
                    "WHERE forecast_id = ? ORDER BY q",
                    (row["forecast_id"],),
                ).fetchall()
            ]
            assert qvals, "no quantiles stored"
            assert all(
                qvals[i] <= qvals[i + 1] + 1e-9 for i in range(len(qvals) - 1)
            ), "quantiles cross"


def test_fit_once_registers_artifact(tmp_path: Path) -> None:
    """fit_once with a model_dir persists and registers a weights artifact."""
    db = tmp_path / "fitart.db"
    model_dir = tmp_path / "models"
    initialize_database(db)
    with connection(db) as conn:
        import_bars_file(
            conn,
            SPY_JSON,
            default_provider=PROVIDER_FIXTURE,
            force_synthetic=False,
        )
        run_bars_walk_forward(
            conn,
            ticker="SPY",
            interval="1d",
            horizon_hours=24,
            challenger_experiment_id="T0",
            provider=PROVIDER_FIXTURE,
            lookback=16,
            train_epochs=3,
            max_issues=4,
            min_b1_samples=20,
            min_model_samples=20,
            exclude_synthetic=True,
            training_mode="fit_once",
            model_dir=model_dir,
        )
        art = conn.execute(
            "SELECT kind, experiment_id, metadata_json FROM artifacts "
            "WHERE experiment_id = 'T0'",
        ).fetchone()
        assert art is not None
        assert art["kind"] == "t0_weights"
        assert "fit_once" in art["metadata_json"]
    assert model_dir.is_dir() and any(model_dir.iterdir())


def test_invalid_training_mode_rejected(tmp_path: Path) -> None:
    db = tmp_path / "badmode.db"
    initialize_database(db)
    with connection(db) as conn:
        import_bars_file(
            conn,
            SPY_JSON,
            default_provider=PROVIDER_FIXTURE,
            force_synthetic=False,
        )
        with pytest.raises(MalformedInputError, match="training_mode"):
            run_bars_walk_forward(
                conn,
                ticker="SPY",
                interval="1d",
                horizon_hours=24,
                challenger_experiment_id="M0",
                provider=PROVIDER_FIXTURE,
                lookback=16,
                max_issues=3,
                min_b1_samples=20,
                min_model_samples=20,
                exclude_synthetic=True,
                training_mode="whenever",  # type: ignore[arg-type]
            )


def test_gate_excludes_synthetic(tmp_path: Path) -> None:
    db = tmp_path / "syn.db"
    initialize_database(db)
    with connection(db) as conn:
        import_bars_file(
            conn,
            SPY_JSON,
            default_provider="local_import",
            force_synthetic=True,
        )
        with pytest.raises(MalformedInputError, match="synthetic"):
            run_bars_walk_forward(
                conn,
                ticker="SPY",
                interval="1d",
                horizon_hours=24,
                lookback=16,
                train_epochs=2,
                max_issues=3,
                min_b1_samples=20,
                min_model_samples=20,
                exclude_synthetic=True,
            )


def test_b1_control_paper_gate_books(tmp_path: Path) -> None:
    db = tmp_path / "paper.db"
    initialize_database(db)
    with connection(db) as conn:
        import_bars_file(
            conn,
            SPY_JSON,
            default_provider=PROVIDER_FIXTURE,
            force_synthetic=False,
        )
        run_bars_walk_forward(
            conn,
            ticker="SPY",
            interval="1d",
            horizon_hours=24,
            challenger_experiment_id="M0",
            provider=PROVIDER_FIXTURE,
            lookback=16,
            train_epochs=3,
            max_issues=6,
            min_b1_samples=20,
            min_model_samples=20,
            exclude_synthetic=True,
        )
        ch_acct = upsert_paper_account(conn, name="ch", starting_cash="100000")
        b1_acct = upsert_paper_account(conn, name="b1", starting_cash="100000")
        ch_pol = default_underlying_policy()
        b1_pol = default_b1_control_policy()
        insert_policy(conn, ch_pol)
        insert_policy(conn, b1_pol)
        freeze_policy(conn, ch_pol.policy_id)
        freeze_policy(conn, b1_pol.policy_id)
        assert "B1" in b1_pol.challenger_experiment_ids

        result = run_paper_gate_books(
            conn,
            challenger_account_id=ch_acct,
            challenger_policy_id=ch_pol.policy_id,
            b1_account_id=b1_acct,
            b1_policy_id=b1_pol.policy_id,
            challenger_experiment_id="M0",
        )
        assert result["challenger"].decisions >= 1
        assert result["b1"].decisions >= 1

        report = evaluate_gate(
            conn,
            challenger_experiment_id="M0",
            min_metric_samples=30,
            challenger_account_id=ch_acct,
            b1_account_id=b1_acct,
        )
        # N=6 < 30 → inconclusive even with paper books
        assert report.status == "inconclusive"
        assert report.challenger_equity_mid is not None
        assert report.b1_equity_mid is not None


def test_cli_import_status_walk_forward_gate(tmp_path: Path) -> None:
    data_dir = tmp_path / "ql"
    init = runner.invoke(app, ["--data-dir", str(data_dir), "init"])
    assert init.exit_code == 0, init.stdout
    imp = runner.invoke(
        app,
        [
            "--data-dir",
            str(data_dir),
            "data",
            "import-bars",
            str(SPY_JSON),
            "--provider",
            "fixture",
        ],
    )
    assert imp.exit_code == 0, imp.stdout
    status = runner.invoke(app, ["--data-dir", str(data_dir), "data", "status"])
    assert status.exit_code == 0, status.stdout
    assert "bars_total=" in status.stdout
    assert "SPY" in status.stdout

    wf = runner.invoke(
        app,
        [
            "--data-dir",
            str(data_dir),
            "experiment",
            "walk-forward",
            "--ticker",
            "SPY",
            "--experiment",
            "M0",
            "--horizon-hours",
            "24",
            "--provider",
            "fixture",
            "--max-issues",
            "4",
            "--train-epochs",
            "2",
        ],
    )
    # Default min samples in walk-forward CLI use defaults (30/40) which may skip
    # or fail with insufficient data depending on lookback. Use library path if CLI
    # uses hard defaults — check exit.
    # CLI doesn't expose min_samples; with 120 bars and defaults it should work.
    assert wf.exit_code == 0, wf.stdout + wf.stderr
    assert "Walk-forward" in wf.stdout

    gate = runner.invoke(
        app,
        [
            "--data-dir",
            str(data_dir),
            "report",
            "gate",
            "--experiment",
            "M0",
            "--ticker",
            "SPY",
        ],
    )
    assert gate.exit_code == 0, gate.stdout
    assert "gate_status=" in gate.stdout
    assert "synthetic_excluded=" in gate.stdout
    assert "Unlock checklist" in gate.stdout


def test_cli_policy_b1_control(tmp_path: Path) -> None:
    data_dir = tmp_path / "ql"
    assert runner.invoke(app, ["--data-dir", str(data_dir), "init"]).exit_code == 0
    out = runner.invoke(
        app, ["--data-dir", str(data_dir), "paper", "policy-init", "--b1-control"]
    )
    assert out.exit_code == 0, out.stdout
    assert "B1 control" in out.stdout
    assert "B1" in out.stdout
