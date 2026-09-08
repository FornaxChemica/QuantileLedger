"""Typer CLI entrypoint: `ql`."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from quantile_ledger import __version__
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
    QuantileLedgerError,
)
from quantile_ledger.logging_setup import configure_logging

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
model_app = typer.Typer(help="Model registry commands (Milestone 1/4).")
forecast_app = typer.Typer(help="Forecast issuance and settlement (Milestone 1+).")
paper_app = typer.Typer(help="Manual paper-only long options (Milestone 6).")
mechanical_app = typer.Typer(help="Mechanical paper benchmark (Milestone 7).")
demo_app = typer.Typer(help="Deterministic offline synthetic demo (Milestone 1).")

app.add_typer(watch_app, name="watch")
app.add_typer(config_app, name="config")
app.add_typer(data_app, name="data")
app.add_typer(model_app, name="model")
app.add_typer(forecast_app, name="forecast")
app.add_typer(paper_app, name="paper")
app.add_typer(mechanical_app, name="mechanical")
app.add_typer(demo_app, name="demo")

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
        if seed_watchlist:
            with connection(settings.database_path) as conn:
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
def calibration_cmd() -> None:
    """Show walk-forward calibration summary (Milestone 1)."""
    _milestone_stub("ql calibration", "Milestone 1")


@app.command("compare")
def compare_cmd() -> None:
    """Compare manual vs mechanical paper series (Milestone 7)."""
    _milestone_stub("ql compare", "Milestone 7")


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
def demo_load_cmd() -> None:
    """Load deterministic synthetic demo data (Milestone 1)."""
    _milestone_stub("ql demo load", "Milestone 1")


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
