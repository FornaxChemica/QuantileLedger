"""SQLite persistence helpers for QuantileLedger."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from importlib import resources
from pathlib import Path

from quantile_ledger.errors import DatabaseError
from quantile_ledger.timeutil import to_iso_utc, utc_now

SCHEMA_VERSION = 1
SCHEMA_DESCRIPTION = "foundation schema with forecast-ready DDL"
BUSY_TIMEOUT_MS = 5000


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


def initialize_database(database_path: Path) -> int:
    """Create schema if needed. Idempotent for repeated init."""
    with connection(database_path) as conn:
        current = get_schema_version(conn)
        if current is not None and current >= SCHEMA_VERSION:
            _assert_foreign_keys(conn)
            return current

        try:
            # executescript auto-commits and can clear connection PRAGMAs.
            conn.executescript(_load_schema_sql())
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute(
                """
                INSERT INTO schema_migrations (version, applied_at, description)
                VALUES (?, ?, ?)
                ON CONFLICT(version) DO NOTHING
                """,
                (SCHEMA_VERSION, to_iso_utc(utc_now()), SCHEMA_DESCRIPTION),
            )
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
