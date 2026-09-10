"""SQLite persistence helpers for QuantileLedger."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from importlib import resources
from pathlib import Path

from quantile_ledger.errors import DatabaseError
from quantile_ledger.timeutil import to_iso_utc, utc_now

SCHEMA_VERSION = 4
SCHEMA_DESCRIPTION = "underlying long/flat paper policy, decisions, fills, cash"
BUSY_TIMEOUT_MS = 5000

_V2_FORECAST_COLUMNS: tuple[tuple[str, str], ...] = (
    ("experiment_id", "TEXT"),
    ("feature_set", "TEXT"),
    ("feature_version", "TEXT"),
    ("training_cutoff", "TEXT"),
    ("data_as_of", "TEXT"),
    ("target_definition", "TEXT"),
    ("target_transform", "TEXT"),
    ("forecast_space", "TEXT"),
    ("calibration_method", "TEXT"),
    ("calibration_version", "TEXT"),
    ("parent_forecast_id", "TEXT"),
    ("random_seed", "INTEGER"),
    ("artifact_digest", "TEXT"),
    ("generation_metadata_json", "TEXT"),
)

_V2_SCHEMA_DESCRIPTION = (
    "foundation + forecast contract columns for multi-model comparison"
)


def _load_schema_sql() -> str:
    schema_path = resources.files("quantile_ledger").joinpath("schema.sql")
    return schema_path.read_text(encoding="utf-8")


def connect(database_path: Path) -> sqlite3.Connection:
    """Open a SQLite connection with foreign keys and WAL."""
    database_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        conn = sqlite3.connect(database_path, timeout=BUSY_TIMEOUT_MS / 1000)
    except sqlite3.Error as exc:
        msg = f"Failed to open database at {database_path}: {exc}"
        raise DatabaseError(msg) from exc
    conn.row_factory = sqlite3.Row
    conn.execute(f"PRAGMA busy_timeout = {BUSY_TIMEOUT_MS}")
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    return conn


@contextmanager
def connection(database_path: Path) -> Iterator[sqlite3.Connection]:
    conn = connect(database_path)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def get_schema_version(conn: sqlite3.Connection) -> int | None:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='schema_migrations'"
    ).fetchone()
    if row is None:
        return None
    version_row = conn.execute(
        "SELECT COALESCE(MAX(version), 0) AS version FROM schema_migrations"
    ).fetchone()
    assert version_row is not None
    value = int(version_row["version"])
    return value if value > 0 else None


def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return {str(row["name"]) for row in rows}


def _migrate_to_v2(conn: sqlite3.Connection) -> None:
    cols = _table_columns(conn, "forecasts")
    for name, decl in _V2_FORECAST_COLUMNS:
        if name not in cols:
            conn.execute(f"ALTER TABLE forecasts ADD COLUMN {name} {decl}")


def _migrate_to_v3(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS artifacts (
            artifact_id TEXT PRIMARY KEY,
            digest TEXT NOT NULL UNIQUE,
            kind TEXT NOT NULL,
            relative_path TEXT NOT NULL,
            created_at TEXT NOT NULL,
            experiment_id TEXT REFERENCES experiments (experiment_id),
            model_version_id TEXT REFERENCES model_versions (model_version_id),
            metadata_json TEXT
        )
        """
    )
    exp_cols = _table_columns(conn, "experiments")
    if "frozen_at" not in exp_cols:
        conn.execute("ALTER TABLE experiments ADD COLUMN frozen_at TEXT")


_V3_SCHEMA_DESCRIPTION = "artifacts table + experiment freeze support"


def _migrate_to_v4(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS paper_policies (
            policy_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            version TEXT NOT NULL,
            status TEXT NOT NULL CHECK (status IN ('draft', 'frozen', 'retired')),
            instrument TEXT NOT NULL DEFAULT 'equity',
            config_hash TEXT NOT NULL,
            config_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            frozen_at TEXT,
            note TEXT,
            UNIQUE (name, version)
        );
        CREATE TABLE IF NOT EXISTS paper_decisions (
            decision_id TEXT PRIMARY KEY,
            account_id TEXT NOT NULL REFERENCES paper_accounts (account_id),
            policy_id TEXT NOT NULL REFERENCES paper_policies (policy_id),
            policy_hash TEXT NOT NULL,
            forecast_id TEXT REFERENCES forecasts (forecast_id),
            experiment_id TEXT,
            ticker TEXT NOT NULL,
            decided_at TEXT NOT NULL,
            action TEXT NOT NULL CHECK (action IN ('long', 'flat')),
            reason TEXT NOT NULL,
            is_forward INTEGER NOT NULL DEFAULT 1 CHECK (is_forward IN (0, 1)),
            is_synthetic INTEGER NOT NULL DEFAULT 0 CHECK (is_synthetic IN (0, 1)),
            metadata_json TEXT
        );
        CREATE TABLE IF NOT EXISTS paper_fills (
            fill_id TEXT PRIMARY KEY,
            account_id TEXT NOT NULL REFERENCES paper_accounts (account_id),
            decision_id TEXT REFERENCES paper_decisions (decision_id),
            policy_id TEXT NOT NULL REFERENCES paper_policies (policy_id),
            ticker TEXT NOT NULL,
            side TEXT NOT NULL CHECK (side IN ('buy', 'sell')),
            quantity TEXT NOT NULL,
            fill_price TEXT NOT NULL,
            bid TEXT,
            ask TEXT,
            mid TEXT,
            half_spread_bps TEXT NOT NULL,
            slippage_bps TEXT NOT NULL,
            commission TEXT NOT NULL,
            notional TEXT NOT NULL,
            cash_delta TEXT NOT NULL,
            quote_time TEXT,
            fill_time TEXT NOT NULL,
            fill_source TEXT NOT NULL,
            data_quality TEXT NOT NULL DEFAULT 'ok',
            is_forward INTEGER NOT NULL DEFAULT 1 CHECK (is_forward IN (0, 1)),
            is_synthetic INTEGER NOT NULL DEFAULT 0 CHECK (is_synthetic IN (0, 1)),
            metadata_json TEXT
        );
        CREATE TABLE IF NOT EXISTS paper_cash_ledger (
            entry_id TEXT PRIMARY KEY,
            account_id TEXT NOT NULL REFERENCES paper_accounts (account_id),
            fill_id TEXT REFERENCES paper_fills (fill_id),
            created_at TEXT NOT NULL,
            amount TEXT NOT NULL,
            balance_after TEXT NOT NULL,
            kind TEXT NOT NULL,
            note TEXT
        );
        """
    )


def initialize_database(database_path: Path) -> int:
    """Create schema if needed and apply additive migrations. Idempotent."""
    with connection(database_path) as conn:
        current = get_schema_version(conn)
        try:
            if current is None:
                conn.executescript(_load_schema_sql())
                conn.execute("PRAGMA foreign_keys = ON")
                conn.execute(
                    """
                    INSERT INTO schema_migrations (version, applied_at, description)
                    VALUES (?, ?, ?)
                    ON CONFLICT(version) DO NOTHING
                    """,
                    (
                        1,
                        to_iso_utc(utc_now()),
                        "foundation schema with forecast-ready DDL",
                    ),
                )
                current = 1

            if current < 2:
                _migrate_to_v2(conn)
                conn.execute(
                    """
                    INSERT INTO schema_migrations (version, applied_at, description)
                    VALUES (?, ?, ?)
                    ON CONFLICT(version) DO NOTHING
                    """,
                    (2, to_iso_utc(utc_now()), _V2_SCHEMA_DESCRIPTION),
                )
                current = 2

            if current < 3:
                _migrate_to_v3(conn)
                conn.execute(
                    """
                    INSERT INTO schema_migrations (version, applied_at, description)
                    VALUES (?, ?, ?)
                    ON CONFLICT(version) DO NOTHING
                    """,
                    (3, to_iso_utc(utc_now()), _V3_SCHEMA_DESCRIPTION),
                )
                current = 3

            if current < 4:
                _migrate_to_v4(conn)
                conn.execute(
                    """
                    INSERT INTO schema_migrations (version, applied_at, description)
                    VALUES (?, ?, ?)
                    ON CONFLICT(version) DO NOTHING
                    """,
                    (4, to_iso_utc(utc_now()), SCHEMA_DESCRIPTION),
                )
                current = 4

            if current < SCHEMA_VERSION:
                msg = (
                    f"schema migration incomplete: at {current}, want {SCHEMA_VERSION}"
                )
                raise DatabaseError(msg)
        except sqlite3.Error as exc:
            msg = f"Failed to initialize schema: {exc}"
            raise DatabaseError(msg) from exc

        _assert_foreign_keys(conn)
        version = get_schema_version(conn)
        if version is None:
            msg = "Schema initialization did not record a version"
            raise DatabaseError(msg)
        return version


def _assert_foreign_keys(conn: sqlite3.Connection) -> None:
    row = conn.execute("PRAGMA foreign_keys").fetchone()
    if row is None or int(row[0]) != 1:
        msg = "Foreign keys are not enabled on this connection"
        raise DatabaseError(msg)


def doctor_database(database_path: Path) -> dict[str, object]:
    """Return local database health details for ql doctor."""
    exists = database_path.exists()
    result: dict[str, object] = {
        "path": str(database_path),
        "exists": exists,
        "schema_version": None,
        "foreign_keys": None,
        "watchlist_active_count": None,
        "error": None,
    }
    if not exists:
        return result
    try:
        with connection(database_path) as conn:
            result["schema_version"] = get_schema_version(conn)
            fk = conn.execute("PRAGMA foreign_keys").fetchone()
            result["foreign_keys"] = bool(fk and int(fk[0]) == 1)
            count = conn.execute(
                "SELECT COUNT(*) AS n FROM watchlist WHERE active = 1"
            ).fetchone()
            result["watchlist_active_count"] = int(count["n"]) if count else 0
    except (sqlite3.Error, DatabaseError) as exc:
        result["error"] = str(exc)
    return result


def list_watchlist(
    conn: sqlite3.Connection, *, active_only: bool = True
) -> list[sqlite3.Row]:
    if active_only:
        return list(
            conn.execute(
                """
                SELECT ticker, active, added_at, removed_at, exchange, currency, note
                FROM watchlist
                WHERE active = 1
                ORDER BY ticker
                """
            )
        )
    return list(
        conn.execute(
            """
            SELECT ticker, active, added_at, removed_at, exchange, currency, note
            FROM watchlist
            ORDER BY ticker
            """
        )
    )


def add_watchlist_ticker(
    conn: sqlite3.Connection,
    ticker: str,
    *,
    exchange: str | None = None,
    currency: str | None = "USD",
    note: str | None = None,
) -> None:
    symbol = ticker.strip().upper()
    if not symbol:
        msg = "Ticker must be non-empty"
        raise DatabaseError(msg)
    now = to_iso_utc(utc_now())
    conn.execute(
        """
        INSERT INTO watchlist (
            ticker, active, added_at, removed_at, exchange, currency, note
        )
        VALUES (?, 1, ?, NULL, ?, ?, ?)
        ON CONFLICT(ticker) DO UPDATE SET
            active = 1,
            removed_at = NULL,
            exchange = COALESCE(excluded.exchange, watchlist.exchange),
            currency = COALESCE(excluded.currency, watchlist.currency),
            note = COALESCE(excluded.note, watchlist.note)
        """,
        (symbol, now, exchange, currency, note),
    )


def remove_watchlist_ticker(conn: sqlite3.Connection, ticker: str) -> bool:
    """Deactivate a ticker without deleting history. Returns True if updated."""
    symbol = ticker.strip().upper()
    now = to_iso_utc(utc_now())
    cur = conn.execute(
        """
        UPDATE watchlist
        SET active = 0, removed_at = ?
        WHERE ticker = ? AND active = 1
        """,
        (now, symbol),
    )
    return cur.rowcount > 0
