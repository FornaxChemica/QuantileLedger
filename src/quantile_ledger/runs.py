"""Auditable forecast/train run records."""

from __future__ import annotations

import json
import sqlite3
import uuid
from typing import Any

from quantile_ledger import __version__
from quantile_ledger.errors import DatabaseError
from quantile_ledger.timeutil import to_iso_utc, utc_now


def open_run(
    conn: sqlite3.Connection,
    *,
    run_type: str,
    config_snapshot_id: str | None = None,
    provider: str | None = None,
) -> str:
    run_id = str(uuid.uuid4())
    try:
        conn.execute(
            """
            INSERT INTO runs (
                run_id, run_type, started_at, ended_at, status, provider,
                software_version, config_snapshot_id, row_counts_json, error_sanitized
            ) VALUES (?, ?, ?, NULL, 'running', ?, ?, ?, NULL, NULL)
            """,
            (
                run_id,
                run_type,
                to_iso_utc(utc_now()),
                provider,
                __version__,
                config_snapshot_id,
            ),
        )
    except sqlite3.Error as exc:
        msg = f"failed to open run: {exc}"
        raise DatabaseError(msg) from exc
    return run_id


def close_run(
    conn: sqlite3.Connection,
    run_id: str,
    *,
    status: str = "succeeded",
    row_counts: dict[str, Any] | None = None,
    error_sanitized: str | None = None,
) -> None:
    try:
        conn.execute(
            """
            UPDATE runs
            SET ended_at = ?, status = ?, row_counts_json = ?, error_sanitized = ?
            WHERE run_id = ?
            """,
            (
                to_iso_utc(utc_now()),
                status,
                json.dumps(row_counts or {}, sort_keys=True),
                error_sanitized,
                run_id,
            ),
        )
    except sqlite3.Error as exc:
        msg = f"failed to close run: {exc}"
        raise DatabaseError(msg) from exc


def fail_run(conn: sqlite3.Connection, run_id: str, error_sanitized: str) -> None:
    close_run(
        conn,
        run_id,
        status="failed",
        error_sanitized=error_sanitized[:500],
    )
