"""Typer CLI entrypoint: `ql`."""

from __future__ import annotations

import json
import math
import sqlite3
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from quantile_ledger import __version__
from quantile_ledger.artifacts import (
    optional_dependency_status,
    register_artifact,
    write_json_artifact,
)
from quantile_ledger.compare import SettledForecast, build_paired_cohort
from quantile_ledger.config import load_settings, save_settings, set_setting
from quantile_ledger.db import (
    add_watchlist_ticker,
    connection,
    doctor_database,
    initialize_database,
    list_watchlist,
    remove_watchlist_ticker,
)
from quantile_ledger.errors import (
    ConfigurationError,
    DatabaseError,
    InsufficientDataError,
    ModelUnavailableError,
    QuantileLedgerError,
)
from quantile_ledger.experiments import K0, M0, T0, get_experiment, list_experiments
from quantile_ledger.forecast_store import (
    freeze_experiment,
    get_forecast_provenance,
    insert_forecast,
    insert_outcome,
    upsert_experiment,
)
from quantile_ledger.forecasting import run_baselines
from quantile_ledger.kronos_quantile import (
    FakeKronosSampler,
    KronosLocalPaths,
    closes_to_ohlc_bars,
    fetch_kronos_assets,
    issue_k0_forecast,
    kronos_bundle_ready,
)
from quantile_ledger.logging_setup import configure_logging
from quantile_ledger.mamba_quantile import issue_m0_forecast, train_m0_model
from quantile_ledger.runs import close_run, fail_run, open_run
from quantile_ledger.synthetic import make_synthetic_hourly_closes, one_step_log_returns
from quantile_ledger.tft_quantile import issue_t0_forecast, train_t0_model

app = typer.Typer(
    name="ql",
    help=(
        "QuantileLedger — point-in-time probabilistic forecasts, "
        "calibration, and paper-only trades. Never places real orders."
    ),
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,
)
watch_app = typer.Typer(help="Manage the local equity/ETF watchlist.")
config_app = typer.Typer(help="Show or update non-secret local settings.")
data_app = typer.Typer(help="Local data fetch/import status (Milestone 2+).")
model_app = typer.Typer(help="Model registry commands.")
forecast_app = typer.Typer(help="Forecast issuance and settlement.")
paper_app = typer.Typer(help="Manual paper-only long options (Milestone 6).")
mechanical_app = typer.Typer(help="Mechanical paper benchmark (Milestone 7).")
demo_app = typer.Typer(help="Deterministic offline synthetic demo.")
experiment_app = typer.Typer(help="Research-matrix experiment registry.")

app.add_typer(watch_app, name="watch")
app.add_typer(config_app, name="config")
app.add_typer(data_app, name="data")
app.add_typer(model_app, name="model")
app.add_typer(forecast_app, name="forecast")
app.add_typer(paper_app, name="paper")
app.add_typer(mechanical_app, name="mechanical")
app.add_typer(demo_app, name="demo")
app.add_typer(experiment_app, name="experiment")

console = Console(stderr=False)
err_console = Console(stderr=True)

DataDirOption = Annotated[
    Path | None,
    typer.Option(
        "--data-dir",
        help="Local data directory (database, cache, logs). Default: ./.ql",
        show_default=False,
    ),
]


def _fail(message: str, code: int = 1) -> None:
    err_console.print(f"[red]error:[/red] {message}")
    raise typer.Exit(code)


def _milestone_stub(feature: str, milestone: str) -> None:
    _fail(f"{feature} is not implemented until {milestone}.", code=2)


@app.callback()
def main_callback(
    ctx: typer.Context,
    data_dir: DataDirOption = None,
    verbose: Annotated[
        bool, typer.Option("--verbose", help="Enable debug logging.")
    ] = False,
) -> None:
    """QuantileLedger CLI root."""
    ctx.ensure_object(dict)
    ctx.obj["data_dir"] = data_dir
    level = "DEBUG" if verbose else "INFO"
    configure_logging(level=level)


@app.command("init")
def init_cmd(
    ctx: typer.Context,
    seed_watchlist: Annotated[
        bool,
        typer.Option(
            "--seed-watchlist/--no-seed-watchlist",
            help="Seed default watchlist tickers when empty.",
        ),
    ] = True,
) -> None:
    """Create local directories, settings, and SQLite schema."""
    try:
        settings = load_settings(data_dir=ctx.obj.get("data_dir"))
        settings.ensure_directories()
        assert settings.database_path is not None
        version = initialize_database(settings.database_path)
        save_settings(settings)
        with connection(settings.database_path) as conn:
            for spec in list_experiments():
                upsert_experiment(conn, spec)
            if seed_watchlist:
                existing = list_watchlist(conn, active_only=False)
                if not existing:
                    for ticker in settings.watchlist:
                        add_watchlist_ticker(conn, ticker)
        console.print(
            "[green]Initialized[/green] QuantileLedger "
            f"(schema v{version}) at {settings.data_dir}"
        )
        console.print(
            "[dim]Paper-only · local-only · no API keys · "
            "no measurable edge is an acceptable result.[/dim]"
        )
    except (ConfigurationError, DatabaseError, OSError) as exc:
        _fail(str(exc))


@app.command("doctor")
def doctor_cmd(ctx: typer.Context) -> None:
    """Check local environment health without network access."""
    try:
        settings = load_settings(data_dir=ctx.obj.get("data_dir")).resolve_paths()
    except ConfigurationError as exc:
        _fail(str(exc))

    assert settings.database_path is not None
    db_info = doctor_database(settings.database_path)

    table = Table(title="QuantileLedger doctor")
    table.add_column("Check")
    table.add_column("Value")
    table.add_row("version", __version__)
    table.add_row("python_requires", ">=3.11,<3.13")
    table.add_row("data_dir", str(settings.data_dir))
    table.add_row("database", str(db_info["path"]))
    table.add_row("database_exists", str(db_info["exists"]))
    table.add_row("schema_version", str(db_info["schema_version"]))
    table.add_row("foreign_keys", str(db_info["foreign_keys"]))
    table.add_row("watchlist_active", str(db_info["watchlist_active_count"]))
    table.add_row("paper_trading", "paper-only (no live orders)")
    table.add_row("api_keys", "not supported")
    table.add_row("telemetry", "disabled")
    for name, state in optional_dependency_status().items():
        table.add_row(f"extra:{name}", state)
    if db_info["error"]:
        table.add_row("database_error", str(db_info["error"]))
    console.print(table)

    if not db_info["exists"]:
        _fail("Database missing. Run: ql init", code=1)
    if db_info["error"]:
        _fail(str(db_info["error"]), code=1)
    if db_info["schema_version"] is None:
        _fail("Schema not initialized. Run: ql init", code=1)


@app.command("nightly")
def nightly_cmd() -> None:
    """Run the local walk-forward job (Milestone 9)."""
    _milestone_stub("ql nightly", "Milestone 9")


@app.command("report")
def report_cmd() -> None:
    """Generate a local evaluation report (Milestone 1+)."""
    _milestone_stub("ql report", "Milestone 1")


@app.command("signal")
def signal_cmd(
    ticker: Annotated[str, typer.Argument()],
    horizon: Annotated[int | None, typer.Option("--horizon")] = None,
) -> None:
    """Show the latest forecast signal for a ticker (Milestone 1)."""
    _ = (ticker, horizon)
    _milestone_stub("ql signal", "Milestone 1")


@app.command("calibration")
def calibration_cmd(ctx: typer.Context) -> None:
    """Show paired B1 vs M0/K0/T0 calibration summary from stored settled forecasts."""
    try:
        settings = load_settings(data_dir=ctx.obj.get("data_dir")).resolve_paths()
        assert settings.database_path is not None
        settled = _load_settled_forecasts(settings.database_path)
        paired = build_paired_cohort(settled, experiment_ids=["B1", "M0", "K0", "T0"])
    except (ConfigurationError, DatabaseError, sqlite3.Error, OSError) as exc:
        _fail(str(exc))
    _print_paired(paired)


@app.command("compare")
def compare_cmd(ctx: typer.Context) -> None:
    """Compare B0/B1/M0/K0/T0 on the strict paired settled cohort."""
    try:
        settings = load_settings(data_dir=ctx.obj.get("data_dir")).resolve_paths()
        assert settings.database_path is not None
        settled = _load_settled_forecasts(settings.database_path)
        paired = build_paired_cohort(
            settled, experiment_ids=["B0", "B1", "M0", "K0", "T0"]
        )
    except (ConfigurationError, DatabaseError, sqlite3.Error, OSError) as exc:
        _fail(str(exc))
    _print_paired(paired)


@experiment_app.command("list")
def experiment_list_cmd(ctx: typer.Context) -> None:
    """List research-matrix experiment identities."""
    table = Table(title="Experiments")
    table.add_column("ID")
    table.add_column("Name")
    table.add_column("Family")
    table.add_column("Features")
    table.add_column("Status")
    for spec in list_experiments():
        table.add_row(
            spec.experiment_id,
            spec.name,
            spec.model_family,
            spec.feature_set,
            spec.status,
        )
    console.print(table)
    console.print(
        "[dim]Note: foundation build 'Milestone 0' ≠ research experiment M0 "
        "(MambaQuantile). K0 = Kronos; T0 = local TFT-style.[/dim]"
    )
    _ = ctx


@experiment_app.command("inspect")
def experiment_inspect_cmd(experiment_id: Annotated[str, typer.Argument()]) -> None:
    """Show immutable experiment configuration."""
    try:
        spec = get_experiment(experiment_id.upper())
    except KeyError as exc:
        _fail(str(exc))
    console.print_json(json.dumps(spec.to_record(), indent=2, sort_keys=True))


@experiment_app.command("freeze")
def experiment_freeze_cmd(
    ctx: typer.Context,
    experiment_id: Annotated[str, typer.Argument()],
) -> None:
    """Freeze an experiment configuration (immutable thereafter)."""
    try:
        settings = load_settings(data_dir=ctx.obj.get("data_dir")).resolve_paths()
        assert settings.database_path is not None
        eid = experiment_id.upper()
        get_experiment(eid)  # validate known id for now
        with connection(settings.database_path) as conn:
            upsert_experiment(conn, get_experiment(eid))
            freeze_experiment(conn, eid)
        console.print(f"[green]Frozen[/green] experiment {eid}")
    except (KeyError, ConfigurationError, DatabaseError) as exc:
        _fail(str(exc))


@forecast_app.command("inspect")
def forecast_inspect_cmd(
    ctx: typer.Context,
    forecast_id: Annotated[str, typer.Argument()],
) -> None:
    """Trace a forecast to experiment, cutoffs, and artifact digest."""
    try:
        settings = load_settings(data_dir=ctx.obj.get("data_dir")).resolve_paths()
        assert settings.database_path is not None
        with connection(settings.database_path) as conn:
            prov = get_forecast_provenance(conn, forecast_id)
        console.print_json(json.dumps(prov, indent=2, sort_keys=True, default=str))
    except (ConfigurationError, DatabaseError) as exc:
        _fail(str(exc))


@experiment_app.command("audit-m0")
def experiment_audit_m0_cmd() -> None:
    """Print the M0 (MambaQuantile market-only) audit checklist status."""
    console.print("[bold]M0 audit (MambaQuantile, market-only)[/bold]")
    console.print(f"experiment_id: {M0.experiment_id}")
    console.print(f"version: {M0.version}")
    console.print(f"config_hash: {M0.config_hash()}")
    console.print(f"target: {M0.target_definition}")
    console.print(f"feature_set: {M0.feature_set}/{M0.feature_version}")
    console.print(f"backbone: {M0.hyperparameters.get('backbone')}")
    console.print(
        "loss: joint pinball across quantiles; "
        "crossing handled by explicit isotonic_pav_v1"
    )
    console.print("status: candidate (greenfield local SSM; not CUDA mamba-ssm)")


@experiment_app.command("audit-k0")
def experiment_audit_k0_cmd() -> None:
    """Print the K0 (Kronos sample-quantile) audit checklist status."""
    console.print("[bold]K0 audit (Kronos sample → quantile, market-only)[/bold]")
    console.print(f"experiment_id: {K0.experiment_id}")
    console.print(f"version: {K0.version}")
    console.print(f"config_hash: {K0.config_hash()}")
    console.print(f"target: {K0.target_definition}")
    console.print(f"feature_set: {K0.feature_set}/{K0.feature_version}")
    console.print(f"quantile_method: {K0.hyperparameters.get('quantile_method')}")
    console.print(f"distribution_claim: {K0.hyperparameters.get('distribution_claim')}")
    console.print(f"sample_count: {K0.hyperparameters.get('sample_count')}")
    console.print(f"upstream_license: {K0.hyperparameters.get('upstream_license')}")
    console.print(
        "status: candidate (demo uses fake sampler; real Kronos via "
        "`uv sync --extra ml` + `ql experiment fetch-k0`)"
    )
    console.print(
        "note: official KronosPredictor averages sample_count; "
        "real adapter issues single-path calls to retain samples"
    )


@experiment_app.command("audit-t0")
def experiment_audit_t0_cmd() -> None:
    """Print the T0 (TFT-style market-only) audit checklist status."""
    console.print("[bold]T0 audit (local TFT-style quantile, market-only)[/bold]")
    console.print(f"experiment_id: {T0.experiment_id}")
    console.print(f"version: {T0.version}")
    console.print(f"config_hash: {T0.config_hash()}")
    console.print(f"target: {T0.target_definition}")
    console.print(f"feature_set: {T0.feature_set}/{T0.feature_version}")
    console.print(f"backbone: {T0.hyperparameters.get('backbone')}")
    console.print(
        "loss: joint pinball across quantiles; "
        "crossing handled by explicit isotonic_pav_v1"
    )
    console.print(
        "status: candidate (local gated-attention TFT-style; not pytorch-forecasting)"
    )


@experiment_app.command("fetch-k0")
def experiment_fetch_k0_cmd(
    ctx: typer.Context,
    force: Annotated[
        bool, typer.Option("--force", help="Re-download even if cache exists")
    ] = False,
) -> None:
    """Download Kronos source + public weights once into local .ql cache."""
    try:
        settings = load_settings(data_dir=ctx.obj.get("data_dir")).resolve_paths()
        settings.ensure_directories()
        paths = KronosLocalPaths.under(settings.data_dir)
        console.print(
            "[bold]Fetching Kronos (public, no API key) into local cache…[/bold]"
        )
        status = fetch_kronos_assets(paths, force=force)
        for key, value in status.items():
            console.print(f"  {key}: {value}")
        if kronos_bundle_ready(paths):
            console.print(
                "[green]Ready[/green] — real K0 can load from local cache. "
                "Demo still uses the fast fake sampler by default."
            )
        else:
            _fail("Kronos bundle incomplete after fetch")
    except (ConfigurationError, ModelUnavailableError, OSError) as exc:
        _fail(str(exc))


@app.command("export")
def export_cmd(
    format: Annotated[str, typer.Option("--format")] = "csv",
    table: Annotated[str, typer.Option("--table")] = "forecasts",
) -> None:
    """Export a local table (Milestone 1+)."""
    _ = (format, table)
    _milestone_stub("ql export", "Milestone 1")


@app.command("reset")
def reset_cmd(
    confirm: Annotated[bool, typer.Option("--confirm")] = False,
    scope: Annotated[str, typer.Option("--scope")] = "all",
) -> None:
    """Scoped local reset with backup (Milestone 1+)."""
    _ = (confirm, scope)
    _milestone_stub("ql reset", "Milestone 1")


@watch_app.command("list")
def watch_list_cmd(
    ctx: typer.Context,
    all_tickers: Annotated[
        bool, typer.Option("--all", help="Include inactive tickers.")
    ] = False,
) -> None:
    """List watchlist tickers."""
    try:
        settings = load_settings(data_dir=ctx.obj.get("data_dir")).resolve_paths()
        assert settings.database_path is not None
        if not settings.database_path.exists():
            _fail("Database missing. Run: ql init")
        with connection(settings.database_path) as conn:
            rows = list_watchlist(conn, active_only=not all_tickers)
    except (ConfigurationError, DatabaseError, sqlite3.Error, OSError) as exc:
        _fail(str(exc))

    if not rows:
        console.print(
            "[yellow]Watchlist empty.[/yellow] Add tickers with: ql watch add TICKER"
        )
        return
    table = Table(title="Watchlist")
    table.add_column("Ticker")
    table.add_column("Active")
    table.add_column("Added at")
    table.add_column("Note")
    for row in rows:
        table.add_row(
            row["ticker"],
            "yes" if row["active"] else "no",
            row["added_at"],
            row["note"] or "",
        )
    console.print(table)


@watch_app.command("add")
def watch_add_cmd(
    ctx: typer.Context,
    ticker: Annotated[str, typer.Argument(help="US-listed equity or ETF symbol.")],
) -> None:
    """Add or reactivate a ticker on the watchlist."""
    try:
        settings = load_settings(data_dir=ctx.obj.get("data_dir")).resolve_paths()
        assert settings.database_path is not None
        if not settings.database_path.exists():
            _fail("Database missing. Run: ql init")
        with connection(settings.database_path) as conn:
            add_watchlist_ticker(conn, ticker)
        console.print(f"[green]Added[/green] {ticker.strip().upper()} to watchlist")
    except (ConfigurationError, DatabaseError) as exc:
        _fail(str(exc))


@watch_app.command("remove")
def watch_remove_cmd(
    ctx: typer.Context,
    ticker: Annotated[str, typer.Argument()],
) -> None:
    """Deactivate a ticker without deleting historical rows."""
    try:
        settings = load_settings(data_dir=ctx.obj.get("data_dir")).resolve_paths()
        assert settings.database_path is not None
        with connection(settings.database_path) as conn:
            changed = remove_watchlist_ticker(conn, ticker)
        if not changed:
            _fail(f"Active ticker not found: {ticker.strip().upper()}")
        console.print(
            f"[green]Removed[/green] {ticker.strip().upper()} (history retained)"
        )
    except (ConfigurationError, DatabaseError) as exc:
        _fail(str(exc))


@config_app.command("show")
def config_show_cmd(ctx: typer.Context) -> None:
    """Show effective non-secret settings."""
    try:
        settings = load_settings(data_dir=ctx.obj.get("data_dir")).resolve_paths()
    except ConfigurationError as exc:
        _fail(str(exc))
    payload = settings.model_dump(mode="json")
    console.print_json(json.dumps(payload, indent=2, sort_keys=True))


@config_app.command("set")
def config_set_cmd(
    ctx: typer.Context,
    key: Annotated[str, typer.Argument()],
    value: Annotated[str, typer.Argument()],
) -> None:
    """Set a non-secret configuration value."""
    try:
        settings = load_settings(data_dir=ctx.obj.get("data_dir")).resolve_paths()
        updated = set_setting(settings, key, value)
        console.print(f"[green]Updated[/green] {key}={getattr(updated, key)}")
    except (ConfigurationError, QuantileLedgerError) as exc:
        _fail(str(exc))


@demo_app.command("load")
def demo_load_cmd(ctx: typer.Context) -> None:
    """Load synthetic bars, issue B0/B1/M0/K0/T0, settle, label synthetic."""
    try:
        settings = load_settings(data_dir=ctx.obj.get("data_dir")).resolve_paths()
        assert settings.database_path is not None
        if not settings.database_path.exists():
            _fail("Database missing. Run: ql init")
        series = make_synthetic_hourly_closes(ticker="SYN", n=320, seed=7)
        bar_horizon = 6
        lookback = 32
        k0_lookback = int(K0.hyperparameters["lookback"])
        issue_idx = len(series.closes) - bar_horizon - 1
        if issue_idx <= max(lookback, k0_lookback) + 50:
            _fail("synthetic series too short")
        closes_known = series.closes[: issue_idx + 1]
        issued_at = series.bar_ends[issue_idx]
        origin = issued_at
        target_at = series.bar_ends[issue_idx + bar_horizon]
        spot = closes_known[-1]
        outcome_price = series.closes[issue_idx + bar_horizon]
        actual_return = math.log(outcome_price / spot)
        training_cutoff = issued_at
        data_as_of = issued_at

        model, train_metrics = train_m0_model(
            closes_known,
            bar_horizon=bar_horizon,
            lookback=lookback,
            epochs=25,
            seed=7,
            min_samples=40,
        )
        t0_model, t0_train_metrics = train_t0_model(
            closes_known,
            bar_horizon=bar_horizon,
            lookback=lookback,
            epochs=25,
            seed=7,
            min_samples=40,
        )
        rets = one_step_log_returns(closes_known)
        assert settings.model_dir is not None
        artifact_id, digest, rel = write_json_artifact(
            artifacts_dir=settings.model_dir,
            kind="m0_weights",
            payload=model.to_dict(),
            filename_stem="m0-demo",
        )
        t0_artifact_id, t0_digest, t0_rel = write_json_artifact(
            artifacts_dir=settings.model_dir,
            kind="t0_weights",
            payload=t0_model.to_dict(),
            filename_stem="t0-demo",
        )
        k0_sampler = FakeKronosSampler(lookback=k0_lookback)
        ohlc = closes_to_ohlc_bars(closes_known)
        future_ends = series.bar_ends[issue_idx + 1 : issue_idx + bar_horizon + 1]
        with connection(settings.database_path) as conn:
            run_id = open_run(conn, run_type="demo_load", provider="synthetic")
            try:
                for spec in list_experiments():
                    upsert_experiment(conn, spec)
                register_artifact(
                    conn,
                    artifact_id=artifact_id,
                    digest=digest,
                    kind="m0_weights",
                    relative_path=str(rel),
                    experiment_id=M0.experiment_id,
                    model_version_id=None,
                    metadata={"is_synthetic": True},
                )
                register_artifact(
                    conn,
                    artifact_id=t0_artifact_id,
                    digest=t0_digest,
                    kind="t0_weights",
                    relative_path=str(t0_rel),
                    experiment_id=T0.experiment_id,
                    model_version_id=None,
                    metadata={"is_synthetic": True},
                )
                b0 = run_baselines(
                    mode="B0",
                    ticker=series.ticker,
                    issued_at=issued_at,
                    origin_bar_at=origin,
                    target_at=target_at,
                    spot_at_issue=spot,
                    horizon_hours=bar_horizon,
                    training_cutoff=training_cutoff,
                    data_as_of=data_as_of,
                    is_synthetic=True,
                )
                b1 = run_baselines(
                    mode="B1",
                    ticker=series.ticker,
                    issued_at=issued_at,
                    origin_bar_at=origin,
                    target_at=target_at,
                    spot_at_issue=spot,
                    horizon_hours=bar_horizon,
                    training_cutoff=training_cutoff,
                    data_as_of=data_as_of,
                    closes_known_by_issue=closes_known,
                    bar_horizon=bar_horizon,
                    is_synthetic=True,
                )
                m0 = issue_m0_forecast(
                    model,
                    ticker=series.ticker,
                    issued_at=issued_at,
                    origin_bar_at=origin,
                    target_at=target_at,
                    spot_at_issue=spot,
                    horizon_hours=bar_horizon,
                    recent_one_step_log_returns=rets,
                    training_cutoff=training_cutoff,
                    data_as_of=data_as_of,
                    is_synthetic=True,
                    run_id=run_id,
                )
                m0 = m0.model_copy(update={"artifact_digest": digest})
                k0 = issue_k0_forecast(
                    k0_sampler,
                    bars=ohlc,
                    bar_ends=series.bar_ends[: issue_idx + 1],
                    future_bar_ends=future_ends,
                    ticker=series.ticker,
                    issued_at=issued_at,
                    origin_bar_at=origin,
                    target_at=target_at,
                    spot_at_issue=spot,
                    horizon_hours=bar_horizon,
                    pred_len=bar_horizon,
                    training_cutoff=training_cutoff,
                    data_as_of=data_as_of,
                    seed=7,
                    is_synthetic=True,
                    run_id=run_id,
                )
                t0 = issue_t0_forecast(
                    t0_model,
                    ticker=series.ticker,
                    issued_at=issued_at,
                    origin_bar_at=origin,
                    target_at=target_at,
                    spot_at_issue=spot,
                    horizon_hours=bar_horizon,
                    recent_one_step_log_returns=rets,
                    training_cutoff=training_cutoff,
                    data_as_of=data_as_of,
                    is_synthetic=True,
                    run_id=run_id,
                )
                t0 = t0.model_copy(update={"artifact_digest": t0_digest})
                b0 = b0.model_copy(update={"run_id": run_id})
                b1 = b1.model_copy(update={"run_id": run_id})
                for fc in (b0, b1, m0, k0, t0):
                    insert_forecast(conn, fc)
                    qmap = dict(
                        zip(fc.quantile_levels, fc.quantile_values, strict=True)
                    )
                    insert_outcome(
                        conn,
                        forecast_id=fc.forecast_id,
                        outcome_price=outcome_price,
                        actual_return=actual_return,
                        outcome_bar_at=target_at,
                        settled_at=target_at,
                        p10=qmap[0.10],
                        p90=qmap[0.90],
                    )
                close_run(
                    conn,
                    run_id,
                    status="succeeded",
                    row_counts={"forecasts": 5, "outcomes": 5},
                )
            except Exception as exc:
                fail_run(conn, run_id, str(exc))
                raise
        console.print(
            "[green]Loaded synthetic demo[/green] "
            f"ticker={series.ticker} horizon={bar_horizon}h "
            f"M0_train_pinball={train_metrics['mean_pinball']:.6f} "
            f"T0_train_pinball={t0_train_metrics['mean_pinball']:.6f} "
            f"K0_samples={k0.generation_metadata.get('sample_count')} "
            f"artifact={digest[:12]}"
        )
        console.print(
            "[dim]All demo rows are labeled is_synthetic=1. "
            "K0 uses FakeKronosSampler (not pretrained weights). "
            "T0 is local TFT-style (not pytorch-forecasting). "
            "Paper-only · no measurable edge claimed.[/dim]"
        )
    except (ConfigurationError, DatabaseError, InsufficientDataError, OSError) as exc:
        _fail(str(exc))


def _load_settled_forecasts(database_path: Path) -> list[SettledForecast]:
    with connection(database_path) as conn:
        rows = conn.execute(
            """
            SELECT f.forecast_id, f.experiment_id, f.ticker, f.issued_at,
                   f.origin_bar_at, f.target_at, f.horizon_hours, f.spot_at_issue,
                   f.price_type, o.actual_return
            FROM forecasts f
            JOIN outcomes o ON o.forecast_id = f.forecast_id
            WHERE o.actual_return IS NOT NULL
              AND f.experiment_id IS NOT NULL
            """
        ).fetchall()
        out: list[SettledForecast] = []
        for row in rows:
            qrows = conn.execute(
                """
                SELECT q, return_value FROM forecast_quantiles
                WHERE forecast_id = ? ORDER BY q
                """,
                (row["forecast_id"],),
            ).fetchall()
            levels = tuple(float(q["q"]) for q in qrows)
            values = tuple(float(q["return_value"]) for q in qrows)
            out.append(
                SettledForecast(
                    forecast_id=row["forecast_id"],
                    experiment_id=row["experiment_id"],
                    ticker=row["ticker"],
                    issued_at=row["issued_at"],
                    origin_bar_at=row["origin_bar_at"],
                    target_at=row["target_at"],
                    horizon_hours=int(row["horizon_hours"]),
                    spot_at_issue=float(row["spot_at_issue"]),
                    quantile_levels=levels,
                    quantile_values=values,
                    actual_return=float(row["actual_return"]),
                    price_type=row["price_type"],
                )
            )
        return out


def _print_paired(paired: object) -> None:
    from quantile_ledger.compare import PairedComparison

    assert isinstance(paired, PairedComparison)
    console.print(f"paired_n={len(paired.cohort)} exclusions={len(paired.exclusions)}")
    table = Table(title="Paired comparison (return space)")
    table.add_column("Experiment")
    table.add_column("N")
    table.add_column("Pinball")
    table.add_column("Cov P10-P90")
    table.add_column("Width")
    table.add_column("Skill vs B1")
    for eid, metrics in paired.metrics_by_experiment.items():
        table.add_row(
            eid,
            str(metrics.get("sample_count")),
            _fmt(metrics.get("mean_pinball")),
            _fmt(metrics.get("coverage_p10_p90")),
            _fmt(metrics.get("mean_width_return")),
            _fmt(metrics.get("skill_vs_B1")),
        )
    console.print(table)
    console.print(
        "[dim]Coverage is shown beside width. Approx CRPS is quantile-grid only. "
        "Negative skill means worse than B1.[/dim]"
    )


def _fmt(value: object) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


@data_app.command("fetch")
def data_fetch_cmd(
    ticker: Annotated[str | None, typer.Option("--ticker")] = None,
) -> None:
    _ = ticker
    _milestone_stub("ql data fetch", "Milestone 2")


@data_app.command("import-news")
def data_import_news_cmd(path: Annotated[Path, typer.Argument()]) -> None:
    _ = path
    _milestone_stub("ql data import-news", "Milestone 2")


@data_app.command("status")
def data_status_cmd() -> None:
    _milestone_stub("ql data status", "Milestone 2")


@model_app.command("list")
def model_list_cmd() -> None:
    _milestone_stub("ql model list", "Milestone 1")


@model_app.command("train")
def model_train_cmd(
    model: Annotated[str, typer.Option("--model")] = "baseline",
) -> None:
    _ = model
    _milestone_stub("ql model train", "Milestone 1")


@model_app.command("activate")
def model_activate_cmd(version: Annotated[str, typer.Argument()]) -> None:
    _ = version
    _milestone_stub("ql model activate", "Milestone 4")


@model_app.command("inspect")
def model_inspect_cmd(version: Annotated[str, typer.Argument()]) -> None:
    _ = version
    _milestone_stub("ql model inspect", "Milestone 1")


@forecast_app.command("run")
def forecast_run_cmd(
    ticker: Annotated[str | None, typer.Option("--ticker")] = None,
) -> None:
    _ = ticker
    _milestone_stub("ql forecast run", "Milestone 1")


@forecast_app.command("backfill")
def forecast_backfill_cmd() -> None:
    _milestone_stub("ql forecast backfill", "Milestone 1")


@paper_app.command("buy")
def paper_buy_cmd() -> None:
    _milestone_stub("ql paper buy", "Milestone 6")


@paper_app.command("positions")
def paper_positions_cmd() -> None:
    _milestone_stub("ql paper positions", "Milestone 6")


@paper_app.command("history")
def paper_history_cmd() -> None:
    _milestone_stub("ql paper history", "Milestone 6")


@paper_app.command("stats")
def paper_stats_cmd() -> None:
    _milestone_stub("ql paper stats", "Milestone 6")


@paper_app.command("mark")
def paper_mark_cmd() -> None:
    _milestone_stub("ql paper mark", "Milestone 6")


@paper_app.command("backfill")
def paper_backfill_cmd() -> None:
    _milestone_stub("ql paper backfill", "Milestone 6")


@mechanical_app.command("run")
def mechanical_run_cmd() -> None:
    _milestone_stub("ql mechanical run", "Milestone 7")


@mechanical_app.command("positions")
def mechanical_positions_cmd() -> None:
    _milestone_stub("ql mechanical positions", "Milestone 7")


@mechanical_app.command("history")
def mechanical_history_cmd() -> None:
    _milestone_stub("ql mechanical history", "Milestone 7")


@mechanical_app.command("stats")
def mechanical_stats_cmd() -> None:
    _milestone_stub("ql mechanical stats", "Milestone 7")
