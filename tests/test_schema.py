"""Database schema initialization tests."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from quantile_ledger.db import (
    SCHEMA_VERSION,
    add_watchlist_ticker,
    connection,
    get_schema_version,
    initialize_database,
    list_watchlist,
    remove_watchlist_ticker,
)


def test_initialize_database_creates_schema(tmp_path: Path) -> None:
    db_path = tmp_path / "test.db"
    version = initialize_database(db_path)
    assert version == SCHEMA_VERSION
    assert db_path.exists()

    with connection(db_path) as conn:
        assert get_schema_version(conn) == SCHEMA_VERSION
        fk = conn.execute("PRAGMA foreign_keys").fetchone()
        assert fk is not None
        assert int(fk[0]) == 1
        tables = {
            row["name"]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        assert "forecasts" in tables
        assert "outcomes" in tables
        assert "watchlist" in tables
        assert "schema_migrations" in tables


def test_initialize_database_is_idempotent(tmp_path: Path) -> None:
    db_path = tmp_path / "test.db"
    first = initialize_database(db_path)
    second = initialize_database(db_path)
    assert first == second == SCHEMA_VERSION
    with connection(db_path) as conn:
        count = conn.execute("SELECT COUNT(*) AS n FROM schema_migrations").fetchone()
        assert count is not None
        assert int(count["n"]) == 1


def test_foreign_keys_reject_orphan_outcome(tmp_path: Path) -> None:
    db_path = tmp_path / "test.db"
    initialize_database(db_path)
    with connection(db_path) as conn, pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            """
            INSERT INTO outcomes (
                forecast_id, settled_at, resolution_rule, quality
            ) VALUES ('missing', '2024-01-01T00:00:00Z', 'test', 'ok')
            """
        )


def test_watchlist_add_remove_preserves_row(tmp_path: Path) -> None:
    db_path = tmp_path / "test.db"
    initialize_database(db_path)
    with connection(db_path) as conn:
        add_watchlist_ticker(conn, "spy")
        active = list_watchlist(conn, active_only=True)
        assert [row["ticker"] for row in active] == ["SPY"]
        assert remove_watchlist_ticker(conn, "SPY") is True
        assert list_watchlist(conn, active_only=True) == []
        all_rows = list_watchlist(conn, active_only=False)
        assert len(all_rows) == 1
        assert all_rows[0]["active"] == 0
