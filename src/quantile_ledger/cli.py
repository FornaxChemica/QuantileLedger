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
from quantile_ledger.compare import (
    SettledForecast,
    build_paired_cohort,
    score_by_regime,
)
from quantile_ledger.config import load_settings, save_settings, set_setting
from quantile_ledger.contract import ForecastContract
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
    ForecastValidationError,
    InsufficientDataError,
    MalformedInputError,
    ModelUnavailableError,
    PaperRiskRejectionError,
    ProviderError,
    QuantileLedgerError,
)
from quantile_ledger.experiments import (
    K0,
    M0,
    M1,
    N0,
    PARENT_TO_CALIBRATED,
    T0,
    T1,
    T2,
    K0c,
    M0c,
    T0c,
    get_experiment,
    list_experiments,
)
from quantile_ledger.forecast_store import (
    freeze_experiment,
    get_forecast_provenance,
    insert_forecast,
    insert_outcome,
    load_forecast_contract,
    upsert_experiment,
)
from quantile_ledger.forecasting import run_baselines
from quantile_ledger.kronos_quantile import (
    KronosLocalPaths,
    build_kronos_sampler,
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
sentiment_app = typer.Typer(
    help="Local FinBERT / fake news sentiment (missing ≠ neutral)."
)
model_app = typer.Typer(help="Model registry commands.")
forecast_app = typer.Typer(help="Forecast issuance and settlement.")
paper_app = typer.Typer(
    help="Paper-only underlying long/flat (options deferred until evidence)."
)
mechanical_app = typer.Typer(
    help="Mechanical equity long/flat paper runner (frozen policy required)."
)
demo_app = typer.Typer(help="Deterministic offline synthetic demo.")
experiment_app = typer.Typer(help="Research-matrix experiment registry.")
report_app = typer.Typer(
    help="Local evaluation reports (Phase WF gate; no cloud upload)."
)

app.add_typer(watch_app, name="watch")
app.add_typer(config_app, name="config")
app.add_typer(data_app, name="data")
app.add_typer(sentiment_app, name="sentiment")
app.add_typer(model_app, name="model")
app.add_typer(forecast_app, name="forecast")
app.add_typer(paper_app, name="paper")
app.add_typer(mechanical_app, name="mechanical")
app.add_typer(demo_app, name="demo")
app.add_typer(experiment_app, name="experiment")
app.add_typer(report_app, name="report")

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
    """Show raw vs isotonic-PIT calibrated cohorts and regime subgroups."""
    try:
        settings = load_settings(data_dir=ctx.obj.get("data_dir")).resolve_paths()
        assert settings.database_path is not None
        settled = _load_settled_forecasts(settings.database_path)
        raw = build_paired_cohort(settled, experiment_ids=["B1", "M0", "K0", "T0"])
        console.print("[bold]Raw market cohort[/bold]")
        _print_paired(raw)
        present = {f.experiment_id for f in settled}
        if {"B1", "M0c", "T0c", "K0c"}.issubset(present):
            cal = build_paired_cohort(
                settled, experiment_ids=["B1", "M0c", "T0c", "K0c"]
            )
            console.print("[bold]Calibrated cohort (isotonic PIT)[/bold]")
            _print_paired(cal)
        for eid in ("M0", "T0", "K0", "M0c", "T0c", "K0c"):
            if eid not in present:
                continue
            by_reg = score_by_regime(settled, experiment_id=eid)
            if not by_reg:
                continue
            console.print(f"[bold]Regime subgroups — {eid}[/bold]")
            _print_regime_table(by_reg)
    except (ConfigurationError, DatabaseError, sqlite3.Error, OSError) as exc:
        _fail(str(exc))


@app.command("compare")
def compare_cmd(ctx: typer.Context) -> None:
    """Compare market, calibration, and news-ablation cohorts on paired keys."""
    try:
        settings = load_settings(data_dir=ctx.obj.get("data_dir")).resolve_paths()
        assert settings.database_path is not None
        settled = _load_settled_forecasts(settings.database_path)
        market = build_paired_cohort(
            settled, experiment_ids=["B0", "B1", "M0", "K0", "T0"]
        )
        console.print("[bold]Market cohort[/bold]")
        _print_paired(market)
        present = {f.experiment_id for f in settled}
        if {"B1", "M0c", "T0c", "K0c"}.issubset(present):
            cal = build_paired_cohort(
                settled, experiment_ids=["B1", "M0c", "T0c", "K0c"]
            )
            console.print("[bold]Calibration cohort (M0c/T0c/K0c)[/bold]")
            _print_paired(cal)
        if {"T0", "T1", "T2"}.issubset(present):
            tft = build_paired_cohort(settled, experiment_ids=["B1", "T0", "T1", "T2"])
            console.print("[bold]TFT news ablation cohort[/bold]")
            _print_paired(tft)
        if {"M0", "M1"}.issubset(present):
            mamba = build_paired_cohort(settled, experiment_ids=["B1", "M0", "M1"])
            console.print("[bold]Mamba news ablation cohort[/bold]")
            _print_paired(mamba)
        if {"B1", "N0"}.issubset(present):
            news = build_paired_cohort(settled, experiment_ids=["B1", "N0"])
            console.print("[bold]News-only vs B1 cohort[/bold]")
            _print_paired(news)
    except (ConfigurationError, DatabaseError, sqlite3.Error, OSError) as exc:
        _fail(str(exc))


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
        "(MambaQuantile). K0 = Kronos; T0/T1/T2 = TFT ladder; "
        "M1/N0 = news ablations.[/dim]"
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


@experiment_app.command("audit-n0")
def experiment_audit_n0_cmd() -> None:
    """Print the N0 (news-only baseline) audit checklist status."""
    console.print("[bold]N0 audit (news-only FinBERT baseline)[/bold]")
    console.print(f"experiment_id: {N0.experiment_id}")
    console.print(f"version: {N0.version}")
    console.print(f"config_hash: {N0.config_hash()}")
    console.print(f"feature_set: {N0.feature_set}/{N0.feature_version}")
    console.print(f"missing_policy: {N0.hyperparameters.get('missing_policy')}")
    console.print(
        "issues forecasts from PIT FinBERT polarity + volume width; "
        "no market returns; missing ≠ neutral"
    )
    console.print("status: candidate (Phase G news-only baseline vs B1)")


@experiment_app.command("audit-g")
def experiment_audit_g_cmd() -> None:
    """Print Phase G ablation ladder status (T1/T2/M1/N0)."""
    console.print("[bold]Phase G ablation audit[/bold]")
    for spec in (T1, T2, M1, N0):
        console.print(
            f"{spec.experiment_id}: feature_set={spec.feature_set} "
            f"parent={spec.hyperparameters.get('parent_experiment_id', '—')} "
            f"hash={spec.config_hash()[:12]}"
        )
    console.print(
        "[dim]T0/M0 stay market-only. News variants refuse missing/partial "
        "context. Coverage always with width + N.[/dim]"
    )


@experiment_app.command("audit-h")
def experiment_audit_h_cmd() -> None:
    """Print Phase H recalibration + regime status (M0c/T0c/K0c)."""
    console.print("[bold]Phase H calibration audit[/bold]")
    for spec in (M0c, T0c, K0c):
        parent = spec.hyperparameters.get("parent_experiment_id")
        method = spec.hyperparameters.get("calibration_method")
        variant = spec.hyperparameters.get("variant")
        console.print(
            f"{spec.experiment_id}: parent={parent} method={method} "
            f"variant={variant} hash={spec.config_hash()[:12]}"
        )
    console.print(
        "[dim]Raw parents immutable. Fit on validation issued_at only; "
        "children only on eval. Regimes = PIT realized-vol terciles with N.[/dim]"
    )


@experiment_app.command("walk-forward")
def experiment_walk_forward_cmd(
    ctx: typer.Context,
    ticker: Annotated[str, typer.Option("--ticker", help="Ticker with bars in DB.")],
    experiment: Annotated[
        str,
        typer.Option("--experiment", help="Challenger: M0, T0, or K0."),
    ] = "M0",
    horizon_hours: Annotated[
        int,
        typer.Option("--horizon-hours", help="24 or 72 for daily bars."),
    ] = 24,
    interval: Annotated[str, typer.Option("--interval")] = "1d",
    provider: Annotated[
        str | None,
        typer.Option("--provider", help="Optional bars.provider filter."),
    ] = None,
    max_issues: Annotated[
        int | None,
        typer.Option(
            "--max-issues", help="Cap issuance count (chronological subsample)."
        ),
    ] = None,
    allow_synthetic: Annotated[
        bool,
        typer.Option(
            "--allow-synthetic",
            help="Allow synthetic bars (excluded from gate by default).",
        ),
    ] = False,
    train_epochs: Annotated[int, typer.Option("--train-epochs")] = 12,
    backend: Annotated[
        str,
        typer.Option(
            "--backend",
            help="K0 sampler backend: 'fake' (offline, default) or 'real'.",
        ),
    ] = "fake",
    training_mode: Annotated[
        str,
        typer.Option(
            "--training-mode",
            help=(
                "M0/T0 fitting: 'per_issuance' (default, retrain each issuance) "
                "or 'fit_once' (train once, reuse; required for heavy models)."
            ),
        ),
    ] = "per_issuance",
) -> None:
    """Issue B1 + challenger on stored bars and settle from later bars."""
    try:
        from quantile_ledger.kronos_quantile import KronosLocalPaths
        from quantile_ledger.walk_forward import run_bars_walk_forward

        if training_mode not in ("per_issuance", "fit_once"):
            _fail("--training-mode must be 'per_issuance' or 'fit_once'")

        settings = load_settings(data_dir=ctx.obj.get("data_dir")).resolve_paths()
        assert settings.database_path is not None
        assert settings.data_dir is not None
        initialize_database(settings.database_path)
        kronos_paths = KronosLocalPaths.under(settings.data_dir)
        with connection(settings.database_path) as conn:
            result = run_bars_walk_forward(
                conn,
                ticker=ticker,
                interval=interval,
                horizon_hours=horizon_hours,
                challenger_experiment_id=experiment,
                provider=provider,
                max_issues=max_issues,
                train_epochs=train_epochs,
                exclude_synthetic=not allow_synthetic,
                kronos_backend=backend,
                kronos_paths=kronos_paths,
                training_mode=training_mode,  # type: ignore[arg-type]
                model_dir=settings.model_dir,
            )
        console.print(
            f"[green]Walk-forward[/green] ticker={result.ticker} "
            f"challenger={result.challenger_experiment_id} "
            f"horizon_h={result.horizon_hours} "
            f"issued={result.issued} settled={result.settled} "
            f"skipped={result.skipped_insufficient} "
            f"synthetic_cohort={result.is_synthetic_cohort} "
            f"run_id={result.run_id[:8]}…"
        )
        console.print(
            "[dim]Next: freeze policies, ql mechanical run on challenger + B1 "
            "accounts, then ql report gate.[/dim]"
        )
    except (
        ConfigurationError,
        DatabaseError,
        InsufficientDataError,
        MalformedInputError,
        PaperRiskRejectionError,
        ForecastValidationError,
    ) as exc:
        _fail(str(exc))


@experiment_app.command("recalibrate")
def experiment_recalibrate_cmd(ctx: typer.Context) -> None:
    """Fit isotonic PIT maps on validation parents; insert eval-window children."""
    try:
        from quantile_ledger.recalibration import (
            fit_isotonic_pit_map,
            issue_recalibrated_from_parent,
            split_issuance_periods,
        )

        settings = load_settings(data_dir=ctx.obj.get("data_dir")).resolve_paths()
        assert settings.database_path is not None
        if not settings.database_path.exists():
            _fail("Database missing. Run: ql init")
        settled = _load_settled_forecasts(settings.database_path)
        parents = [f for f in settled if f.experiment_id in PARENT_TO_CALIBRATED]
        if not parents:
            _fail("No settled M0/T0/K0 parents found")
        issued = sorted({f.issued_at for f in parents})
        period = split_issuance_periods(issued)
        inserted = 0
        with connection(settings.database_path) as conn:
            for parent_eid, child_eid in PARENT_TO_CALIBRATED.items():
                upsert_experiment(conn, get_experiment(child_eid))
                val_rows = [
                    f
                    for f in parents
                    if f.experiment_id == parent_eid
                    and f.issued_at in period.validation_issued_ats
                ]
                if len(val_rows) < 2:
                    console.print(
                        f"[yellow]skip {parent_eid}: need >=2 validation rows[/yellow]"
                    )
                    continue
                pit_map = fit_isotonic_pit_map(
                    parent_experiment_id=parent_eid,
                    levels_list=[list(f.quantile_levels) for f in val_rows],
                    values_list=[list(f.quantile_values) for f in val_rows],
                    actuals=[f.actual_return for f in val_rows],
                    issued_ats=[f.issued_at for f in val_rows],
                    period=period,
                )
                eval_rows = [
                    f
                    for f in parents
                    if f.experiment_id == parent_eid
                    and f.issued_at in period.eval_issued_ats
                ]
                for row in eval_rows:
                    existing = conn.execute(
                        """
                        SELECT 1 FROM forecasts
                        WHERE parent_forecast_id = ? AND experiment_id = ?
                        """,
                        (row.forecast_id, child_eid),
                    ).fetchone()
                    if existing is not None:
                        continue
                    parent = load_forecast_contract(conn, row.forecast_id)
                    child = issue_recalibrated_from_parent(
                        parent, pit_map=pit_map, period=period
                    )
                    insert_forecast(conn, child)
                    qmap = dict(
                        zip(child.quantile_levels, child.quantile_values, strict=True)
                    )
                    insert_outcome(
                        conn,
                        forecast_id=child.forecast_id,
                        outcome_price=row.spot_at_issue * math.exp(row.actual_return),
                        actual_return=row.actual_return,
                        outcome_bar_at=row.target_at,
                        settled_at=row.target_at,
                        p10=qmap[0.10],
                        p90=qmap[0.90],
                    )
                    inserted += 1
            conn.commit()
        console.print(
            f"[green]Recalibrated[/green] inserted={inserted} "
            f"val=[{period.validation_start} … {period.validation_end}] "
            f"eval=[{period.eval_start} … {period.eval_end}]"
        )
        console.print("[dim]Raw rows unchanged. No measurable edge claimed.[/dim]")
    except (
        ConfigurationError,
        DatabaseError,
        ForecastValidationError,
        InsufficientDataError,
        sqlite3.Error,
        OSError,
    ) as exc:
        _fail(str(exc))


@experiment_app.command("fetch-n0")
def experiment_fetch_n0_cmd(
    ctx: typer.Context,
    force: Annotated[
        bool, typer.Option("--force", help="Re-download even if cache exists")
    ] = False,
) -> None:
    """Download ProsusAI/finbert weights once into local .ql cache (no API key)."""
    try:
        from quantile_ledger.sentiment import fetch_finbert_weights, finbert_local_dir

        settings = load_settings(data_dir=ctx.obj.get("data_dir")).resolve_paths()
        settings.ensure_directories()
        console.print(
            "[bold]Fetching FinBERT (public, no API key) into local cache…[/bold]"
        )
        path = fetch_finbert_weights(settings.data_dir, force=force)
        console.print(f"  local_dir: {finbert_local_dir(settings.data_dir).name}/…")
        console.print(f"[green]Ready[/green] — weights under {path.name}/ (relative)")
        console.print(
            "[dim]Default scoring still uses FakeFinBERT unless "
            "`ql sentiment score --backend finbert`.[/dim]"
        )
    except (ConfigurationError, ModelUnavailableError, OSError) as exc:
        _fail(str(exc))


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
    """Load multi-issuance synthetic demo with Phase G ablations + Phase H cal."""
    try:
        from quantile_ledger.ablations import (
            issue_m1_from_m0,
            issue_n0_news_only,
            issue_t1_from_t0,
            issue_t2_from_t0,
            load_context_for_ablation,
        )
        from quantile_ledger.news import import_news_file
        from quantile_ledger.recalibration import (
            fit_isotonic_pit_map,
            issue_recalibrated_from_parent,
            split_issuance_periods,
        )
        from quantile_ledger.regimes import (
            fit_vol_tercile_thresholds,
            realized_vol,
            regime_for_closes,
        )
        from quantile_ledger.sentiment import FakeFinBERT, score_unscored_news

        settings = load_settings(data_dir=ctx.obj.get("data_dir")).resolve_paths()
        assert settings.database_path is not None
        if not settings.database_path.exists():
            _fail("Database missing. Run: ql init")
        series = make_synthetic_hourly_closes(ticker="SYN", n=320, seed=7)
        bar_horizon = 6
        lookback = 32
        vol_lookback = 20
        n_issues = 5
        k0_lookback = int(K0.hyperparameters["lookback"])
        start_idx = max(lookback, k0_lookback) + 50
        end_idx = len(series.closes) - bar_horizon - 1
        if end_idx <= start_idx + n_issues:
            _fail("synthetic series too short for multi-issuance demo")
        issue_indices = [
            start_idx + round(i * (end_idx - start_idx) / (n_issues - 1))
            for i in range(n_issues)
        ]
        issued_ats = [series.bar_ends[i] for i in issue_indices]
        period = split_issuance_periods(issued_ats)

        # Train once on closes through the first issuance so
        # training_cutoff <= issued_at for every later demo forecast.
        first_idx = issue_indices[0]
        train_closes = series.closes[: first_idx + 1]
        training_cutoff = series.bar_ends[first_idx]
        model, train_metrics = train_m0_model(
            train_closes,
            bar_horizon=bar_horizon,
            lookback=lookback,
            epochs=20,
            seed=7,
            min_samples=40,
        )
        t0_model, t0_train_metrics = train_t0_model(
            train_closes,
            bar_horizon=bar_horizon,
            lookback=lookback,
            epochs=20,
            seed=7,
            min_samples=40,
        )
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
        # The demo is deterministic, offline, and synthetic-only, so K0 always
        # uses the fake sampler here. Real Kronos is exercised via
        # `ql experiment walk-forward --backend real` on real bars.
        assert settings.data_dir is not None
        k0_sampler = build_kronos_sampler(
            "fake",
            paths=KronosLocalPaths.under(settings.data_dir),
            lookback=k0_lookback,
        )

        val_vols: list[float] = []
        for idx, ts in zip(issue_indices, issued_ats, strict=True):
            if ts not in period.validation_issued_ats:
                continue
            val_vols.append(
                realized_vol(series.closes[: idx + 1], lookback=vol_lookback)
            )
        vol_thresholds = fit_vol_tercile_thresholds(
            val_vols,
            fit_as_of_max=period.validation_end,
            lookback=vol_lookback,
        )

        news_items: list[dict[str, object]] = []
        for issue_idx in issue_indices:
            news_items.extend(
                [
                    {
                        "ticker": series.ticker,
                        "headline": "SYN beats estimates; shares surge",
                        "published_at": series.bar_ends[max(0, issue_idx - 8)],
                        "ingested_at": series.bar_ends[max(0, issue_idx - 7)],
                        "is_synthetic": True,
                    },
                    {
                        "ticker": series.ticker,
                        "headline": "Analyst upgrade lifts outlook",
                        "published_at": series.bar_ends[max(0, issue_idx - 3)],
                        "ingested_at": series.bar_ends[max(0, issue_idx - 2)],
                        "is_synthetic": True,
                    },
                ]
            )
        news_path = settings.data_dir / "demo_news.json"
        news_path.write_text(json.dumps(news_items), encoding="utf-8")

        raw_count = 0
        cal_count = 0
        with connection(settings.database_path) as conn:
            run_id = open_run(conn, run_type="demo_load", provider="synthetic")
            try:
                for spec in list_experiments():
                    upsert_experiment(conn, spec)
                import_news_file(conn, news_path, force_synthetic=True)
                score_unscored_news(conn, scorer=FakeFinBERT())
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

                parents_by_eid: dict[
                    str, list[tuple[ForecastContract, float, float]]
                ] = {"M0": [], "T0": [], "K0": []}
                for issue_idx in issue_indices:
                    closes_known = series.closes[: issue_idx + 1]
                    issued_at = series.bar_ends[issue_idx]
                    origin = issued_at
                    target_at = series.bar_ends[issue_idx + bar_horizon]
                    spot = closes_known[-1]
                    outcome_price = series.closes[issue_idx + bar_horizon]
                    actual_return = math.log(outcome_price / spot)
                    data_as_of = issued_at
                    rets = one_step_log_returns(closes_known)
                    ohlc = closes_to_ohlc_bars(closes_known)
                    future_ends = series.bar_ends[
                        issue_idx + 1 : issue_idx + bar_horizon + 1
                    ]
                    regime, regime_reasons = regime_for_closes(
                        closes_known,
                        thresholds=vol_thresholds,
                        bar_ends=series.bar_ends[: issue_idx + 1],
                        issued_at=issued_at,
                    )
                    regime_update = {"regime": regime, "regime_reasons": regime_reasons}

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
                    ).model_copy(update={**regime_update, "run_id": run_id})
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
                    ).model_copy(update={**regime_update, "run_id": run_id})
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
                    ).model_copy(update={**regime_update, "artifact_digest": digest})
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
                        seed=7 + issue_idx,
                        is_synthetic=True,
                        run_id=run_id,
                    ).model_copy(update=regime_update)
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
                    ).model_copy(update={**regime_update, "artifact_digest": t0_digest})

                    news_ctx = load_context_for_ablation(
                        conn,
                        ticker=series.ticker,
                        issued_at=issued_at,
                        lookback_hours=72,
                    )
                    t1 = issue_t1_from_t0(t0, news_ctx).model_copy(update=regime_update)
                    t2 = issue_t2_from_t0(t0, news_ctx).model_copy(update=regime_update)
                    m1 = issue_m1_from_m0(m0, news_ctx).model_copy(update=regime_update)
                    n0 = issue_n0_news_only(
                        ticker=series.ticker,
                        issued_at=issued_at,
                        origin_bar_at=origin,
                        target_at=target_at,
                        spot_at_issue=spot,
                        horizon_hours=bar_horizon,
                        training_cutoff=training_cutoff,
                        data_as_of=data_as_of,
                        ctx=news_ctx,
                        is_synthetic=True,
                        run_id=run_id,
                    ).model_copy(update=regime_update)

                    forecasts = (b0, b1, m0, m1, k0, t0, t1, t2, n0)
                    for fc in forecasts:
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
                        raw_count += 1
                    parents_by_eid["M0"].append((m0, actual_return, outcome_price))
                    parents_by_eid["T0"].append((t0, actual_return, outcome_price))
                    parents_by_eid["K0"].append((k0, actual_return, outcome_price))

                for parent_eid in ("M0", "T0", "K0"):
                    rows = parents_by_eid[parent_eid]
                    val_rows = [
                        (p, y, op)
                        for p, y, op in rows
                        if p.issued_at in period.validation_issued_ats
                    ]
                    pit_map = fit_isotonic_pit_map(
                        parent_experiment_id=parent_eid,
                        levels_list=[list(p.quantile_levels) for p, _, _ in val_rows],
                        values_list=[list(p.quantile_values) for p, _, _ in val_rows],
                        actuals=[y for _, y, _ in val_rows],
                        issued_ats=[p.issued_at for p, _, _ in val_rows],
                        period=period,
                    )
                    for parent, actual_return, outcome_price in rows:
                        if parent.issued_at not in period.eval_issued_ats:
                            continue
                        child = issue_recalibrated_from_parent(
                            parent, pit_map=pit_map, period=period
                        )
                        insert_forecast(conn, child)
                        qmap = dict(
                            zip(
                                child.quantile_levels,
                                child.quantile_values,
                                strict=True,
                            )
                        )
                        insert_outcome(
                            conn,
                            forecast_id=child.forecast_id,
                            outcome_price=outcome_price,
                            actual_return=actual_return,
                            outcome_bar_at=child.target_at,
                            settled_at=child.target_at,
                            p10=qmap[0.10],
                            p90=qmap[0.90],
                        )
                        cal_count += 1

                close_run(
                    conn,
                    run_id,
                    status="succeeded",
                    row_counts={
                        "forecasts": raw_count + cal_count,
                        "outcomes": raw_count + cal_count,
                        "issuances": n_issues,
                        "calibrated": cal_count,
                    },
                )
            except Exception as exc:
                fail_run(conn, run_id, str(exc))
                raise
        console.print(
            "[green]Loaded synthetic demo[/green] "
            f"ticker={series.ticker} horizon={bar_horizon}h "
            f"issuances={n_issues} "
            f"val_n={len(period.validation_issued_ats)} "
            f"eval_n={len(period.eval_issued_ats)} "
            f"calibrated={cal_count} "
            f"M0_train_pinball={train_metrics['mean_pinball']:.6f} "
            f"T0_train_pinball={t0_train_metrics['mean_pinball']:.6f} "
            f"ablations=T1,T2,M1,N0 "
            f"cal=M0c,T0c,K0c "
            f"artifact={digest[:12]}"
        )
        console.print(
            "[dim]All demo rows are labeled is_synthetic=1. "
            "K0 uses FakeKronosSampler (not pretrained weights). "
            "T0 is local TFT-style (not pytorch-forecasting). "
            "News ablations use FakeFinBERT; missing≠neutral. "
            "Calibrated children fit on validation issuances only. "
            "Paper-only · no measurable edge claimed.[/dim]"
        )
    except (
        ConfigurationError,
        DatabaseError,
        ForecastValidationError,
        InsufficientDataError,
        OSError,
    ) as exc:
        _fail(str(exc))


def _load_settled_forecasts(database_path: Path) -> list[SettledForecast]:
    with connection(database_path) as conn:
        rows = conn.execute(
            """
            SELECT f.forecast_id, f.experiment_id, f.ticker, f.issued_at,
                   f.origin_bar_at, f.target_at, f.horizon_hours, f.spot_at_issue,
                   f.price_type, f.variant, f.regime, f.parent_forecast_id,
                   o.actual_return
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
                    variant=row["variant"] or "raw",
                    regime=row["regime"],
                    parent_forecast_id=row["parent_forecast_id"],
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


def _print_regime_table(by_reg: dict[str, dict[str, float | int | None]]) -> None:
    table = Table(title="Regime subgroups (N + coverage + width)")
    table.add_column("Regime")
    table.add_column("N")
    table.add_column("Pinball")
    table.add_column("Cov P10-P90")
    table.add_column("Width")
    for label, metrics in by_reg.items():
        table.add_row(
            label,
            str(metrics.get("sample_count")),
            _fmt(metrics.get("mean_pinball")),
            _fmt(metrics.get("coverage_p10_p90")),
            _fmt(metrics.get("mean_width_return")),
        )
    console.print(table)


def _fmt(value: object) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


@data_app.command("fetch")
def data_fetch_cmd(
    ctx: typer.Context,
    ticker: Annotated[str | None, typer.Option("--ticker")] = None,
    force: Annotated[
        bool,
        typer.Option("--force", help="Re-download even if cache is fresh."),
    ] = False,
) -> None:
    """Fetch keyless Stooq daily bars into local SQLite (no API key)."""
    try:
        from quantile_ledger.providers import fetch_bars

        settings = load_settings(data_dir=ctx.obj.get("data_dir")).resolve_paths()
        assert settings.database_path is not None
        assert settings.cache_dir is not None
        initialize_database(settings.database_path)
        tickers = [ticker.upper()] if ticker else list(settings.watchlist)
        with connection(settings.database_path) as conn:
            for t in tickers:
                result = fetch_bars(
                    conn,
                    t,
                    cache_dir=settings.cache_dir,
                    force=force,
                    freshness_hours=settings.bar_freshness_hours,
                )
                console.print(
                    f"[green]Fetched[/green] {result.ticker} "
                    f"provider={result.provider} "
                    f"from_cache={result.from_cache} "
                    f"inserted={result.import_result.inserted} "
                    f"updated={result.import_result.updated} "
                    f"rows={result.import_result.total_rows}"
                )
        console.print(
            "[dim]Stooq daily only (interval=1d). Gate horizons: 24h / 72h. "
            "Cached under .ql/cache/bars/.[/dim]"
        )
    except (
        ConfigurationError,
        DatabaseError,
        MalformedInputError,
        ProviderError,
        QuantileLedgerError,
        OSError,
    ) as exc:
        _fail(str(exc))


@data_app.command("import-bars")
def data_import_bars_cmd(
    ctx: typer.Context,
    path: Annotated[Path, typer.Argument(help="Local CSV/JSON/JSONL OHLCV file.")],
    synthetic: Annotated[
        bool,
        typer.Option(
            "--synthetic",
            help="Force is_synthetic=1 on imported rows (demo/fixtures).",
        ),
    ] = False,
    interval: Annotated[
        str,
        typer.Option("--interval", help="Default interval when row omits it."),
    ] = "1d",
    provider: Annotated[
        str,
        typer.Option("--provider", help="Default provider label."),
    ] = "local_import",
) -> None:
    """Import local OHLCV bars (no network). Idempotent upsert."""
    try:
        from quantile_ledger.bars import import_bars_file

        settings = load_settings(data_dir=ctx.obj.get("data_dir")).resolve_paths()
        assert settings.database_path is not None
        initialize_database(settings.database_path)
        with connection(settings.database_path) as conn:
            result = import_bars_file(
                conn,
                path,
                default_interval=interval,
                default_provider=provider,
                force_synthetic=synthetic,
            )
        console.print(
            "[green]Imported bars[/green] "
            f"inserted={result.inserted} "
            f"updated={result.updated} "
            f"rows={result.total_rows}"
        )
        if synthetic:
            console.print(
                "[dim]Labeled is_synthetic=1 — excluded from ql report gate.[/dim]"
            )
    except (
        ConfigurationError,
        DatabaseError,
        MalformedInputError,
        OSError,
    ) as exc:
        _fail(str(exc))


@data_app.command("import-news")
def data_import_news_cmd(
    ctx: typer.Context,
    path: Annotated[Path, typer.Argument(help="Local JSON or JSONL news file.")],
    synthetic: Annotated[
        bool,
        typer.Option(
            "--synthetic",
            help="Force is_synthetic=1 on imported rows (demo/fixtures).",
        ),
    ] = False,
) -> None:
    """Import local news JSON/JSONL (no network). Enforces ingest ≥ publish."""
    try:
        from quantile_ledger.forecast_store import upsert_experiment
        from quantile_ledger.news import import_news_file

        settings = load_settings(data_dir=ctx.obj.get("data_dir")).resolve_paths()
        assert settings.database_path is not None
        initialize_database(settings.database_path)
        with connection(settings.database_path) as conn:
            upsert_experiment(conn, N0)
            result = import_news_file(conn, path, force_synthetic=synthetic)
        console.print(
            "[green]Imported news[/green] "
            f"inserted={result.inserted} "
            f"duplicates={result.skipped_duplicate} "
            f"rows={result.total_rows}"
        )
        console.print(
            "[dim]Point-in-time: eligible only when published_at and "
            "ingested_at ≤ issued_at. Missing ≠ neutral.[/dim]"
        )
    except (
        ConfigurationError,
        DatabaseError,
        MalformedInputError,
        OSError,
    ) as exc:
        _fail(str(exc))


@data_app.command("status")
def data_status_cmd(ctx: typer.Context) -> None:
    """Show local bars / news / sentiment row counts (no private paths)."""
    try:
        from quantile_ledger.bars import bars_counts
        from quantile_ledger.news import news_counts

        settings = load_settings(data_dir=ctx.obj.get("data_dir")).resolve_paths()
        assert settings.database_path is not None
        initialize_database(settings.database_path)
        with connection(settings.database_path) as conn:
            news = news_counts(conn)
            bars = bars_counts(conn)
        console.print(
            f"bars_total={bars['bars_total']} "
            f"bars_non_synthetic={bars['bars_non_synthetic']} "
            f"bars_synthetic={bars['bars_synthetic']}"
        )
        for row in bars["by_provider_interval"]:
            console.print(
                f"  bars provider={row['provider']} interval={row['interval']} "
                f"n={row['n']}"
            )
        for row in bars["by_ticker"]:
            console.print(f"  bars ticker={row['ticker']} n={row['n']}")
        console.print(
            f"news_items={news['news_items']} news_sentiment={news['news_sentiment']}"
        )
    except (ConfigurationError, DatabaseError) as exc:
        _fail(str(exc))


@sentiment_app.command("score")
def sentiment_score_cmd(
    ctx: typer.Context,
    backend: Annotated[
        str,
        typer.Option("--backend", help="fake (default) or finbert"),
    ] = "fake",
    ticker: Annotated[str | None, typer.Option("--ticker")] = None,
    limit: Annotated[int | None, typer.Option("--limit")] = None,
) -> None:
    """Score unscored local headlines (FakeFinBERT or optional ProsusAI/finbert)."""
    try:
        from quantile_ledger.sentiment import resolve_scorer, score_unscored_news

        be = backend.lower().strip()
        if be not in {"fake", "finbert"}:
            _fail("--backend must be fake or finbert")
        settings = load_settings(data_dir=ctx.obj.get("data_dir")).resolve_paths()
        assert settings.database_path is not None
        initialize_database(settings.database_path)
        scorer = resolve_scorer(backend=be, data_dir=settings.data_dir)  # type: ignore[arg-type]
        with connection(settings.database_path) as conn:
            result = score_unscored_news(
                conn, scorer=scorer, ticker=ticker, limit=limit
            )
        console.print(
            "[green]Sentiment score[/green] "
            f"backend={be} scorer={scorer.scorer_id} "
            f"scored={result['scored']} failed={result['failed']} "
            f"considered={result['considered']}"
        )
        if be == "fake":
            console.print("[dim]FakeFinBERT is synthetic lexicon — not ProsusAI.[/dim]")
    except (ConfigurationError, DatabaseError, ModelUnavailableError) as exc:
        _fail(str(exc))


@sentiment_app.command("context")
def sentiment_context_cmd(
    ctx: typer.Context,
    ticker: Annotated[str, typer.Option("--ticker")],
    issued_at: Annotated[
        str, typer.Option("--issued-at", help="Forecast issuance UTC ISO timestamp")
    ],
    scorer: Annotated[
        str,
        typer.Option("--scorer", help="Scorer id (default fake_finbert_v1)"),
    ] = "fake_finbert_v1",
) -> None:
    """Show PIT sentiment context for a ticker (missing ≠ neutral)."""
    try:
        from quantile_ledger.sentiment import build_sentiment_context

        settings = load_settings(data_dir=ctx.obj.get("data_dir")).resolve_paths()
        assert settings.database_path is not None
        initialize_database(settings.database_path)
        with connection(settings.database_path) as conn:
            ctx_row = build_sentiment_context(
                conn,
                ticker=ticker,
                issued_at=issued_at,
                scorer_id=scorer,
            )
        console.print(json.dumps(ctx_row.as_dict(), indent=2, sort_keys=True))
        if ctx_row.status == "missing":
            console.print(
                "[yellow]status=missing[/yellow] — no eligible news; "
                "do not treat as neutral."
            )
        elif ctx_row.status == "unscored":
            console.print(
                "[yellow]status=unscored[/yellow] — news present but no scores; "
                "run `ql sentiment score`."
            )
        elif (
            ctx_row.n_scored is not None
            and ctx_row.n_items > 0
            and ctx_row.n_scored < ctx_row.n_items
        ):
            console.print(
                f"[yellow]partial[/yellow] — scored {ctx_row.n_scored}/"
                f"{ctx_row.n_items} eligible items."
            )
    except (ConfigurationError, DatabaseError, MalformedInputError) as exc:
        _fail(str(exc))


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


@paper_app.command("account-init")
def paper_account_init_cmd(
    ctx: typer.Context,
    name: Annotated[str, typer.Option("--name")] = "default",
    cash: Annotated[str | None, typer.Option("--cash")] = None,
) -> None:
    """Create a paper equity account with starting cash."""
    try:
        from quantile_ledger.paper import upsert_paper_account

        settings = load_settings(data_dir=ctx.obj.get("data_dir")).resolve_paths()
        assert settings.database_path is not None
        starting = cash or settings.paper_starting_cash
        with connection(settings.database_path) as conn:
            account_id = upsert_paper_account(
                conn,
                name=name,
                starting_cash=starting,
                note="paper equity account",
            )
        console.print(
            f"[green]Paper account[/green] name={name} id={account_id} "
            f"cash={starting} (hypothetical)"
        )
    except (ConfigurationError, DatabaseError, ValueError) as exc:
        _fail(str(exc))


@paper_app.command("policy-init")
def paper_policy_init_cmd(
    ctx: typer.Context,
    b1_control: Annotated[
        bool,
        typer.Option(
            "--b1-control",
            help="Insert B1-only control policy (same costs) for gate comparison.",
        ),
    ] = False,
) -> None:
    """Insert the default underlying long/flat policy as draft."""
    try:
        from dataclasses import replace

        from quantile_ledger.paper import (
            default_b1_control_policy,
            default_underlying_policy,
            insert_policy,
        )

        settings = load_settings(data_dir=ctx.obj.get("data_dir")).resolve_paths()
        assert settings.database_path is not None
        base = (
            default_b1_control_policy() if b1_control else default_underlying_policy()
        )
        policy = replace(
            base,
            half_spread_bps=settings.paper_equity_half_spread_bps,
            slippage_bps=settings.paper_equity_slippage_bps,
            commission_per_share=settings.paper_equity_commission_per_share,
            shares_per_entry=settings.paper_equity_shares_per_entry,
            max_notional_per_trade=settings.paper_equity_max_notional_per_trade,
            min_p50_log_return=settings.paper_min_p50_log_return,
            quote_max_age_seconds=settings.quote_max_age_seconds,
        )
        with connection(settings.database_path) as conn:
            insert_policy(conn, policy)
        kind = "B1 control" if b1_control else "challenger"
        console.print(
            f"[green]Draft {kind} policy[/green] id={policy.policy_id} "
            f"experiments={list(policy.challenger_experiment_ids)} "
            f"hash={policy.config_hash()[:12]} — freeze before forward tests"
        )
    except (ConfigurationError, DatabaseError, ValueError) as exc:
        _fail(str(exc))


@paper_app.command("policy-freeze")
def paper_policy_freeze_cmd(
    ctx: typer.Context,
    policy_id: Annotated[str, typer.Argument()],
) -> None:
    """Freeze a draft paper policy (immutable thereafter)."""
    try:
        from quantile_ledger.paper import freeze_policy

        settings = load_settings(data_dir=ctx.obj.get("data_dir")).resolve_paths()
        assert settings.database_path is not None
        with connection(settings.database_path) as conn:
            policy = freeze_policy(conn, policy_id)
        console.print(
            f"[green]Frozen[/green] policy {policy.name}/v{policy.version} "
            f"at {policy.frozen_at} hash={policy.config_hash()[:12]}"
        )
    except (ConfigurationError, DatabaseError, QuantileLedgerError) as exc:
        _fail(str(exc))


@paper_app.command("policy-show")
def paper_policy_show_cmd(
    ctx: typer.Context,
    policy_id: Annotated[str, typer.Argument()],
) -> None:
    """Show a paper policy configuration."""
    try:
        from quantile_ledger.paper import get_policy

        settings = load_settings(data_dir=ctx.obj.get("data_dir")).resolve_paths()
        assert settings.database_path is not None
        with connection(settings.database_path) as conn:
            policy = get_policy(conn, policy_id)
        console.print_json(
            json.dumps(
                {
                    "policy_id": policy.policy_id,
                    "name": policy.name,
                    "version": policy.version,
                    "status": policy.status,
                    "config_hash": policy.config_hash(),
                    "frozen_at": policy.frozen_at,
                    "config": policy.config_payload(),
                },
                indent=2,
                sort_keys=True,
            )
        )
    except (ConfigurationError, DatabaseError) as exc:
        _fail(str(exc))


@paper_app.command("buy")
def paper_buy_cmd() -> None:
    """Options paper buy — blocked until underlying evidence (Phase J2)."""
    _fail(
        "Options paper trading is blocked until underlying long/flat forward "
        "results show economic value after costs vs B1. Use equity policy path."
    )


@paper_app.command("positions")
def paper_positions_cmd(ctx: typer.Context) -> None:
    """Show open paper equity share quantities by ticker."""
    try:
        settings = load_settings(data_dir=ctx.obj.get("data_dir")).resolve_paths()
        assert settings.database_path is not None
        with connection(settings.database_path) as conn:
            accounts = conn.execute(
                "SELECT account_id, name FROM paper_accounts ORDER BY name"
            ).fetchall()
            if not accounts:
                console.print("[yellow]No paper accounts.[/yellow]")
                return
            from quantile_ledger.paper import open_share_quantity

            for acct in accounts:
                tickers = conn.execute(
                    """
                    SELECT DISTINCT ticker FROM paper_fills
                    WHERE account_id = ? ORDER BY ticker
                    """,
                    (acct["account_id"],),
                ).fetchall()
                console.print(
                    f"[bold]{acct['name']}[/bold] ({acct['account_id'][:8]}…)"
                )
                if not tickers:
                    console.print("  (no fills)")
                    continue
                for trow in tickers:
                    qty = open_share_quantity(conn, acct["account_id"], trow["ticker"])
                    console.print(f"  {trow['ticker']}: {qty} shares (paper)")
    except (ConfigurationError, DatabaseError) as exc:
        _fail(str(exc))


@paper_app.command("history")
def paper_history_cmd(ctx: typer.Context) -> None:
    """List recent paper decisions (forward flag included)."""
    try:
        settings = load_settings(data_dir=ctx.obj.get("data_dir")).resolve_paths()
        assert settings.database_path is not None
        with connection(settings.database_path) as conn:
            rows = conn.execute(
                """
                SELECT decided_at, ticker, action, reason, experiment_id,
                       is_forward, is_synthetic
                FROM paper_decisions
                ORDER BY decided_at DESC, rowid DESC
                LIMIT 50
                """
            ).fetchall()
        if not rows:
            console.print("[yellow]No paper decisions yet.[/yellow]")
            return
        table = Table(title="Paper decisions (hypothetical)")
        table.add_column("When")
        table.add_column("Ticker")
        table.add_column("Action")
        table.add_column("Exp")
        table.add_column("Fwd")
        table.add_column("Reason")
        for row in rows:
            table.add_row(
                row["decided_at"],
                row["ticker"],
                row["action"],
                row["experiment_id"] or "",
                "yes" if row["is_forward"] else "no",
                row["reason"][:60],
            )
        console.print(table)
    except (ConfigurationError, DatabaseError, sqlite3.Error) as exc:
        _fail(str(exc))


@paper_app.command("stats")
def paper_stats_cmd(ctx: typer.Context) -> None:
    """Show paper cash, mid/bid equity, and forward decision/fill counts."""
    try:
        from quantile_ledger.paper import paper_equity_summary

        settings = load_settings(data_dir=ctx.obj.get("data_dir")).resolve_paths()
        assert settings.database_path is not None
        with connection(settings.database_path) as conn:
            accounts = conn.execute(
                "SELECT account_id, name FROM paper_accounts ORDER BY name"
            ).fetchall()
            if not accounts:
                console.print("[yellow]No paper accounts.[/yellow]")
                return
            for acct in accounts:
                summary = paper_equity_summary(conn, acct["account_id"])
                mid = (
                    str(summary.equity_mid) if summary.equity_mid is not None else "n/a"
                )
                bid = (
                    str(summary.equity_bid) if summary.equity_bid is not None else "n/a"
                )
                ret_mid = (
                    f"{summary.return_mid:.6f}"
                    if summary.return_mid is not None
                    else "n/a"
                )
                ret_bid = (
                    f"{summary.return_bid:.6f}"
                    if summary.return_bid is not None
                    else "n/a"
                )
                console.print(
                    f"[bold]{summary.name}[/bold] "
                    f"cash={summary.cash} start={summary.starting_cash} "
                    f"equity_mid={mid} equity_bid={bid} "
                    f"ret_mid={ret_mid} ret_bid={ret_bid} "
                    f"marks={summary.mark_count} "
                    f"decisions={summary.decision_count}"
                    f"/{summary.forward_decision_count}fwd "
                    f"fills={summary.fill_count}/{summary.forward_fill_count}fwd "
                    "[dim]paper[/dim]"
                )
    except (ConfigurationError, DatabaseError, QuantileLedgerError) as exc:
        _fail(str(exc))


@paper_app.command("mark")
def paper_mark_cmd(
    ctx: typer.Context,
    account_id: Annotated[str, typer.Option("--account-id")],
    spot: Annotated[
        list[str] | None,
        typer.Option(
            "--spot",
            help="Ticker=spot mid (repeatable). Bid/ask from policy half-spread.",
        ),
    ] = None,
    from_last_fill: Annotated[
        bool,
        typer.Option(
            "--from-last-fill",
            help="Mark open positions using bid/ask from the latest fill per ticker.",
        ),
    ] = False,
    half_spread_bps: Annotated[
        str | None,
        typer.Option(
            "--half-spread-bps",
            help="Half-spread bps when using --spot (default: policy or 5).",
        ),
    ] = None,
    exploratory: Annotated[
        bool,
        typer.Option("--exploratory", help="Store mark with is_forward=0."),
    ] = False,
) -> None:
    """Record mid accounting vs bid liquidation marks (hypothetical)."""
    try:
        from quantile_ledger.mechanical import quote_from_spot
        from quantile_ledger.paper import (
            get_policy,
            list_open_positions,
            quotes_from_last_fills,
            record_account_mark,
        )

        if bool(spot) == from_last_fill:
            _fail("Provide exactly one of --spot TICKER=PRICE or --from-last-fill")

        settings = load_settings(data_dir=ctx.obj.get("data_dir")).resolve_paths()
        assert settings.database_path is not None
        with connection(settings.database_path) as conn:
            positions = list_open_positions(conn, account_id)
            if from_last_fill:
                quotes = quotes_from_last_fills(
                    conn, account_id, list(positions.keys())
                )
                source = "last_fill"
            else:
                assert spot is not None
                spread = half_spread_bps
                if spread is None:
                    row = conn.execute(
                        """
                        SELECT policy_id FROM paper_decisions
                        WHERE account_id = ?
                        ORDER BY decided_at DESC, rowid DESC LIMIT 1
                        """,
                        (account_id,),
                    ).fetchone()
                    if row is not None:
                        spread = get_policy(conn, row["policy_id"]).half_spread_bps
                    else:
                        spread = settings.paper_equity_half_spread_bps
                quotes = {}
                for item in spot:
                    if "=" not in item:
                        _fail(f"invalid --spot {item!r}; expected TICKER=PRICE")
                    ticker, price_s = item.split("=", 1)
                    quotes[ticker.upper()] = quote_from_spot(
                        ticker=ticker,
                        spot=float(price_s),
                        half_spread_bps=spread,
                        quote_time="",
                        quality="cli_spot",
                    )
                source = "cli_spot"
            missing = [t for t in positions if t not in quotes]
            if missing:
                _fail(f"missing quotes for open positions: {', '.join(missing)}")
            mark = record_account_mark(
                conn,
                account_id=account_id,
                quotes=quotes,
                quote_source=source,
                is_forward=not exploratory,
                is_synthetic=False,
            )
        console.print(
            "[green]Paper mark[/green] "
            f"cash={mark.cash} equity_mid={mark.equity_mid} "
            f"equity_bid={mark.equity_bid} positions={len(mark.positions)} "
            f"source={source} [dim]paper[/dim]"
        )
    except (ConfigurationError, DatabaseError, QuantileLedgerError, ValueError) as exc:
        _fail(str(exc))


@paper_app.command("equity")
def paper_equity_cmd(
    ctx: typer.Context,
    account_id: Annotated[str, typer.Option("--account-id")],
) -> None:
    """Print the stored mid vs bid equity curve (hypothetical)."""
    try:
        from quantile_ledger.paper import list_marks

        settings = load_settings(data_dir=ctx.obj.get("data_dir")).resolve_paths()
        assert settings.database_path is not None
        with connection(settings.database_path) as conn:
            marks = list_marks(conn, account_id)
        if not marks:
            console.print("[yellow]No paper marks yet.[/yellow]")
            return
        table = Table(title="Paper equity curve (hypothetical)")
        table.add_column("Marked at")
        table.add_column("Cash")
        table.add_column("Equity mid")
        table.add_column("Equity bid")
        table.add_column("Pos")
        table.add_column("Fwd")
        table.add_column("Source")
        for mark in marks:
            table.add_row(
                mark.marked_at,
                str(mark.cash),
                str(mark.equity_mid),
                str(mark.equity_bid),
                str(len(mark.positions)),
                "yes" if mark.is_forward else "no",
                mark.quote_source,
            )
        console.print(table)
        console.print(
            f"[dim]N={len(marks)} marks · mid ≠ bid liquidation · paper[/dim]"
        )
    except (ConfigurationError, DatabaseError) as exc:
        _fail(str(exc))


@paper_app.command("forward-demo")
def paper_forward_demo_cmd(
    ctx: typer.Context,
    account_id: Annotated[str, typer.Option("--account-id")],
    policy_id: Annotated[str, typer.Option("--policy-id")],
    experiment: Annotated[
        str, typer.Option("--experiment", help="Challenger id: M0, K0, or T0")
    ] = "M0",
    steps: Annotated[
        int, typer.Option("--steps", help="Number of sequential issuance steps (>=2)")
    ] = 8,
) -> None:
    """Accumulate multi-day synthetic forward paper results (labeled)."""
    try:
        from quantile_ledger.paper_forward import run_synthetic_forward_book

        if steps < 2:
            _fail("--steps must be >= 2")
        settings = load_settings(data_dir=ctx.obj.get("data_dir")).resolve_paths()
        assert settings.database_path is not None
        with connection(settings.database_path) as conn:
            result = run_synthetic_forward_book(
                conn,
                account_id=account_id,
                policy_id=policy_id,
                experiment_id=experiment.upper(),
                steps=steps,
            )
        s = result.summary
        mid = str(s.equity_mid) if s.equity_mid is not None else "n/a"
        bid = str(s.equity_bid) if s.equity_bid is not None else "n/a"
        console.print(
            "[green]Forward paper demo[/green] "
            f"experiment={result.experiment_id} forecasts={result.forecasts} "
            f"issue_days={result.distinct_issue_days} "
            f"decisions={result.decisions} entries={result.entries} "
            f"exits={result.exits} flats={result.flats_recorded} "
            f"marks={result.marks} equity_mid={mid} equity_bid={bid}"
        )
        console.print(
            "[dim]All rows labeled is_synthetic=1 · is_forward=1 · "
            "paper-only · no measurable edge claimed.[/dim]"
        )
    except (ConfigurationError, DatabaseError, QuantileLedgerError, ValueError) as exc:
        _fail(str(exc))


@paper_app.command("backfill")
def paper_backfill_cmd() -> None:
    _fail(
        "Options expiration backfill is Phase J2 (blocked until underlying evidence)."
    )


@mechanical_app.command("run")
def mechanical_run_cmd(
    ctx: typer.Context,
    account_id: Annotated[str, typer.Option("--account-id")],
    policy_id: Annotated[str, typer.Option("--policy-id")],
    experiment: Annotated[
        str, typer.Option("--experiment", help="Challenger id: M0, K0, or T0")
    ] = "M0",
    exploratory: Annotated[
        bool,
        typer.Option(
            "--exploratory",
            help="Mark decisions as not forward (is_forward=0).",
        ),
    ] = False,
) -> None:
    """Run long/flat decisions on undecided settled challenger forecasts."""
    try:
        from quantile_ledger.mechanical import run_mechanical_underlying

        settings = load_settings(data_dir=ctx.obj.get("data_dir")).resolve_paths()
        assert settings.database_path is not None
        eid = experiment.upper()
        with connection(settings.database_path) as conn:
            result = run_mechanical_underlying(
                conn,
                account_id=account_id,
                policy_id=policy_id,
                experiment_id=eid,
                is_forward=not exploratory,
            )
        console.print(
            "[green]Mechanical run[/green] "
            f"experiment={eid} decisions={result.decisions} "
            f"entries={result.entries} exits={result.exits} "
            f"flat_holds={result.flats_recorded} "
            f"forward={'no' if exploratory else 'yes'}"
        )
        if result.cash_after is not None:
            console.print(f"cash_after={result.cash_after} [dim]paper[/dim]")
    except (ConfigurationError, DatabaseError, QuantileLedgerError) as exc:
        _fail(str(exc))


@mechanical_app.command("positions")
def mechanical_positions_cmd(ctx: typer.Context) -> None:
    """Alias: show paper equity positions."""
    paper_positions_cmd(ctx)


@mechanical_app.command("history")
def mechanical_history_cmd(ctx: typer.Context) -> None:
    """Alias: show paper decision history."""
    paper_history_cmd(ctx)


@mechanical_app.command("stats")
def mechanical_stats_cmd(ctx: typer.Context) -> None:
    """Alias: show paper cash / mid vs bid equity stats."""
    paper_stats_cmd(ctx)


@report_app.command("gate")
def report_gate_cmd(
    ctx: typer.Context,
    experiment: Annotated[
        str,
        typer.Option("--experiment", help="Challenger experiment id (e.g. M0)."),
    ] = "M0",
    ticker: Annotated[str | None, typer.Option("--ticker")] = None,
    challenger_account: Annotated[
        str | None,
        typer.Option("--challenger-account", help="Paper account for challenger book."),
    ] = None,
    b1_account: Annotated[
        str | None,
        typer.Option("--b1-account", help="Paper account for B1 control book."),
    ] = None,
) -> None:
    """Evidence gate: skill vs B1 and paper equity delta (excludes synthetic)."""
    try:
        from quantile_ledger.gate import evaluate_gate

        settings = load_settings(data_dir=ctx.obj.get("data_dir")).resolve_paths()
        assert settings.database_path is not None
        if not settings.database_path.exists():
            _fail("Database missing. Run: ql init")
        with connection(settings.database_path) as conn:
            report = evaluate_gate(
                conn,
                challenger_experiment_id=experiment,
                min_metric_samples=settings.min_metric_samples,
                ticker=ticker,
                challenger_account_id=challenger_account,
                b1_account_id=b1_account,
            )
        color = {
            "pass": "green",
            "fail": "red",
            "inconclusive": "yellow",
        }[report.status]
        console.print(
            f"[{color}]gate_status={report.status}[/{color}] "
            f"challenger={report.challenger_experiment_id} "
            f"paired_n={report.paired_n} "
            f"min_n={report.min_metric_samples} "
            f"synthetic_excluded={report.synthetic_excluded}"
        )
        console.print(
            f"  skill_vs_B1={_fmt(report.skill_vs_b1)} "
            f"coverage={_fmt(report.coverage)} "
            f"mean_width={_fmt(report.mean_width)} "
            f"approx_crps={_fmt(report.approx_crps)} "
            f"pinball={_fmt(report.pinball)}"
        )
        console.print(
            f"  challenger_equity_mid={report.challenger_equity_mid} "
            f"b1_equity_mid={report.b1_equity_mid} "
            f"equity_delta_mid={report.equity_delta_mid} "
            f"fills_ch={report.challenger_fills} fills_b1={report.b1_fills}"
        )
        for reason in report.reasons:
            console.print(f"  reason: {reason}")
        console.print("[bold]Unlock checklist (human review; not automatic)[/bold]")
        for item in report.checklist:
            console.print(f"  • {item}")
    except (ConfigurationError, DatabaseError) as exc:
        _fail(str(exc))


@paper_app.command("gate-run")
def paper_gate_run_cmd(
    ctx: typer.Context,
    challenger_account: Annotated[str, typer.Option("--challenger-account")],
    challenger_policy: Annotated[str, typer.Option("--challenger-policy")],
    b1_account: Annotated[str, typer.Option("--b1-account")],
    b1_policy: Annotated[str, typer.Option("--b1-policy")],
    experiment: Annotated[str, typer.Option("--experiment")] = "M0",
    exploratory: Annotated[
        bool,
        typer.Option("--exploratory", help="Mark decisions is_forward=0."),
    ] = False,
) -> None:
    """Run mechanical long/flat for challenger and B1 control books."""
    try:
        from quantile_ledger.walk_forward import run_paper_gate_books

        settings = load_settings(data_dir=ctx.obj.get("data_dir")).resolve_paths()
        assert settings.database_path is not None
        with connection(settings.database_path) as conn:
            result = run_paper_gate_books(
                conn,
                challenger_account_id=challenger_account,
                challenger_policy_id=challenger_policy,
                b1_account_id=b1_account,
                b1_policy_id=b1_policy,
                challenger_experiment_id=experiment,
                is_forward=not exploratory,
            )
        ch = result["challenger"]
        b1 = result["b1"]
        console.print(
            f"[green]Gate paper[/green] challenger decisions={ch.decisions} "
            f"entries={ch.entries} exits={ch.exits} flats={ch.flats_recorded}"
        )
        console.print(
            f"  B1 control decisions={b1.decisions} entries={b1.entries} "
            f"exits={b1.exits} flats={b1.flats_recorded}"
        )
    except (
        ConfigurationError,
        DatabaseError,
        PaperRiskRejectionError,
    ) as exc:
        _fail(str(exc))
