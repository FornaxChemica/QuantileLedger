"""OHLCV bar ingest, PIT queries, and local import (Milestone 2 / Phase WF)."""

from __future__ import annotations

import csv
import io
import json
import math
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from quantile_ledger.errors import DatabaseError, MalformedInputError
from quantile_ledger.timeutil import parse_iso_utc, to_iso_utc, utc_now

Interval = Literal["1d", "1h"]
PROVIDER_LOCAL_IMPORT = "local_import"
PROVIDER_STOOQ = "stooq"
PROVIDER_FIXTURE = "fixture"
ALLOWED_INTERVALS: frozenset[str] = frozenset({"1d", "1h"})
# Daily walk-forward gate: horizon hours that map cleanly to trading-day bars.
DAILY_GATE_HORIZONS: frozenset[int] = frozenset({24, 72})


@dataclass(frozen=True)
class BarRecord:
    ticker: str
    interval: str
    bar_start: str
    bar_end: str
    open: float | None
    high: float | None
    low: float | None
    close: float
    volume: float | None
    adjustment: str
    provider: str
    first_fetched_at: str
    last_observed_at: str
    revision_state: str = "initial"
    quality: str = "ok"
    is_synthetic: bool = False


@dataclass(frozen=True)
class BarsImportResult:
    inserted: int
    updated: int
    total_rows: int


@dataclass(frozen=True)
class CloseSeries:
    """Point-in-time completed closes (bar_end ascending)."""

    ticker: str
    interval: str
    bar_ends: list[str]
    closes: list[float]
    is_synthetic: bool


def normalize_interval(value: str) -> str:
    iv = value.strip().lower()
    if iv not in ALLOWED_INTERVALS:
        msg = f"unsupported interval {value!r}; allowed: {sorted(ALLOWED_INTERVALS)}"
        raise MalformedInputError(msg)
    return iv


def horizon_hours_to_bar_horizon(horizon_hours: int, interval: str) -> int:
    """Map contract horizon_hours to number of bars for the given interval."""
    iv = normalize_interval(interval)
    if iv == "1d":
        if horizon_hours not in DAILY_GATE_HORIZONS:
            msg = (
                f"daily bars only support horizons {sorted(DAILY_GATE_HORIZONS)}h "
                f"(got {horizon_hours})"
            )
            raise MalformedInputError(msg)
        return horizon_hours // 24
    # 1h: one bar per hour
    if horizon_hours < 1:
        msg = "horizon_hours must be >= 1"
        raise MalformedInputError(msg)
    return horizon_hours


def _normalize_iso(value: str, *, field: str) -> str:
    try:
        return to_iso_utc(parse_iso_utc(value))
    except (ValueError, TypeError) as exc:
        msg = f"invalid {field} timestamp: {value!r}"
        raise MalformedInputError(msg) from exc


def _optional_float(raw: Any, *, field: str) -> float | None:
    if raw is None or raw == "":
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        msg = f"invalid {field}: {raw!r}"
        raise MalformedInputError(msg) from exc
    if not math.isfinite(value):
        msg = f"non-finite {field}: {raw!r}"
        raise MalformedInputError(msg)
    return value


def _require_positive_close(close: float) -> float:
    if close <= 0:
        msg = f"close must be positive, got {close}"
        raise MalformedInputError(msg)
    return close


def _validate_ohlc(
    *,
    open_: float | None,
    high: float | None,
    low: float | None,
    close: float,
) -> None:
    if high is not None and low is not None and high < low:
        msg = f"high {high} < low {low}"
        raise MalformedInputError(msg)
    for name, value in (("open", open_), ("high", high), ("low", low)):
        if value is not None and value <= 0:
            msg = f"{name} must be positive when present, got {value}"
            raise MalformedInputError(msg)
    if high is not None and close > high + 1e-12:
        msg = f"close {close} above high {high}"
        raise MalformedInputError(msg)
    if low is not None and close < low - 1e-12:
        msg = f"close {close} below low {low}"
        raise MalformedInputError(msg)


def bar_record_from_mapping(
    raw: dict[str, Any],
    *,
    default_interval: str = "1d",
    default_provider: str = PROVIDER_LOCAL_IMPORT,
    default_adjustment: str = "adjusted_research",
    force_synthetic: bool | None = None,
    fetched_at: str | None = None,
) -> BarRecord:
    """Parse one bar row (dict from CSV/JSON)."""
    try:
        ticker = str(raw["ticker"]).strip().upper()
        close = _require_positive_close(float(raw["close"]))
    except KeyError as exc:
        msg = f"bar row missing required field: {exc}"
        raise MalformedInputError(msg) from exc
    except (TypeError, ValueError) as exc:
        msg = f"invalid close: {raw.get('close')!r}"
        raise MalformedInputError(msg) from exc
    if not ticker:
        msg = "ticker must be non-empty"
        raise MalformedInputError(msg)

    if "bar_end" in raw and raw["bar_end"] is not None:
        bar_end = _normalize_iso(str(raw["bar_end"]), field="bar_end")
    elif "date" in raw and raw["date"] is not None:
        # Stooq / CSV date → end of UTC day for daily bars.
        date_s = str(raw["date"]).strip()
        if "T" in date_s:
            bar_end = _normalize_iso(date_s, field="date")
        else:
            bar_end = _normalize_iso(f"{date_s}T21:00:00Z", field="date")
    else:
        msg = "bar row requires bar_end or date"
        raise MalformedInputError(msg)

    interval = normalize_interval(str(raw.get("interval", default_interval)))
    if "bar_start" in raw and raw["bar_start"] is not None:
        bar_start = _normalize_iso(str(raw["bar_start"]), field="bar_start")
    elif interval == "1d":
        # Session start placeholder: same calendar day 14:30Z (approx US open).
        day = bar_end[:10]
        bar_start = _normalize_iso(f"{day}T14:30:00Z", field="bar_start")
    else:
        bar_start = bar_end

    if bar_start > bar_end:
        msg = f"bar_start {bar_start} after bar_end {bar_end}"
        raise MalformedInputError(msg)

    open_ = _optional_float(raw.get("open"), field="open")
    high = _optional_float(raw.get("high"), field="high")
    low = _optional_float(raw.get("low"), field="low")
    volume = _optional_float(raw.get("volume"), field="volume")
    _validate_ohlc(open_=open_, high=high, low=low, close=close)

    provider = str(raw.get("provider", default_provider)).strip() or default_provider
    adjustment = (
        str(raw.get("adjustment", default_adjustment)).strip() or default_adjustment
    )
    now = fetched_at or to_iso_utc(utc_now())
    if force_synthetic is None:
        is_synthetic = bool(raw.get("is_synthetic", False))
    else:
        is_synthetic = force_synthetic
    quality = str(raw.get("quality", "ok")).strip() or "ok"
    revision = str(raw.get("revision_state", "initial")).strip() or "initial"

    return BarRecord(
        ticker=ticker,
        interval=interval,
        bar_start=bar_start,
        bar_end=bar_end,
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=volume,
        adjustment=adjustment,
        provider=provider,
        first_fetched_at=now,
        last_observed_at=now,
        revision_state=revision,
        quality=quality,
        is_synthetic=is_synthetic,
    )


def load_bars_file(path: Path) -> list[dict[str, Any]]:
    """Load local bars from CSV, JSON array, JSONL, or {bars: [...]}."""
    if not path.is_file():
        msg = f"bars file not found: {path}"
        raise MalformedInputError(msg)
    text = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return _parse_csv_text(text)
    if suffix in {".jsonl", ".ndjson"}:
        rows: list[dict[str, Any]] = []
        for line_no, line in enumerate(text.splitlines(), start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                obj = json.loads(stripped)
            except json.JSONDecodeError as exc:
                msg = f"invalid JSONL at line {line_no}: {exc}"
                raise MalformedInputError(msg) from exc
            if not isinstance(obj, dict):
                msg = f"JSONL line {line_no} must be an object"
                raise MalformedInputError(msg)
            rows.append(obj)
        return rows
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        msg = f"invalid JSON bars file: {exc}"
        raise MalformedInputError(msg) from exc
    if isinstance(payload, dict) and "bars" in payload:
        payload = payload["bars"]
    if not isinstance(payload, list):
        msg = "bars JSON must be an array or {bars: [...]}"
        raise MalformedInputError(msg)
    out: list[dict[str, Any]] = []
    for i, obj in enumerate(payload):
        if not isinstance(obj, dict):
            msg = f"bar item {i} must be an object"
            raise MalformedInputError(msg)
        out.append(obj)
    return out


def _parse_csv_text(text: str) -> list[dict[str, Any]]:
    reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames is None:
        msg = "CSV has no header row"
        raise MalformedInputError(msg)
    # Normalize headers (Stooq uses Date,Open,High,Low,Close,Volume).
    rows: list[dict[str, Any]] = []
    for raw in reader:
        normalized: dict[str, Any] = {}
        for key, value in raw.items():
            if key is None:
                continue
            k = key.strip().lower()
            normalized[k] = value.strip() if isinstance(value, str) else value
        if not any(v not in (None, "") for v in normalized.values()):
            continue
        rows.append(normalized)
    return rows


def parse_stooq_csv(
    text: str,
    *,
    ticker: str,
    fetched_at: str | None = None,
) -> list[BarRecord]:
    """Parse Stooq daily CSV into BarRecords (provider=stooq, interval=1d)."""
    rows = _parse_csv_text(text)
    now = fetched_at or to_iso_utc(utc_now())
    out: list[BarRecord] = []
    for raw in rows:
        raw = dict(raw)
        raw.setdefault("ticker", ticker)
        raw.setdefault("interval", "1d")
        raw.setdefault("provider", PROVIDER_STOOQ)
        raw.setdefault("adjustment", "adjusted_research")
        raw.setdefault("is_synthetic", False)
        out.append(
            bar_record_from_mapping(
                raw,
                default_interval="1d",
                default_provider=PROVIDER_STOOQ,
                force_synthetic=False,
                fetched_at=now,
            )
        )
    return out


def upsert_bar(conn: Any, bar: BarRecord) -> Literal["inserted", "updated"]:
    """Idempotent upsert on (provider, ticker, interval, bar_end)."""
    existing = conn.execute(
        """
        SELECT bar_id, first_fetched_at FROM bars
        WHERE provider = ? AND ticker = ? AND interval = ? AND bar_end = ?
        """,
        (bar.provider, bar.ticker, bar.interval, bar.bar_end),
    ).fetchone()
    if existing is None:
        try:
            conn.execute(
                """
                INSERT INTO bars (
                    ticker, interval, bar_start, bar_end, open, high, low, close,
                    volume, adjustment, provider, first_fetched_at, last_observed_at,
                    revision_state, quality, is_synthetic
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    bar.ticker,
                    bar.interval,
                    bar.bar_start,
                    bar.bar_end,
                    bar.open,
                    bar.high,
                    bar.low,
                    bar.close,
                    bar.volume,
                    bar.adjustment,
                    bar.provider,
                    bar.first_fetched_at,
                    bar.last_observed_at,
                    bar.revision_state,
                    bar.quality,
                    1 if bar.is_synthetic else 0,
                ),
            )
        except Exception as exc:
            msg = f"bars insert failed: {exc}"
            raise DatabaseError(msg) from exc
        return "inserted"

    try:
        conn.execute(
            """
            UPDATE bars SET
                bar_start = ?,
                open = ?,
                high = ?,
                low = ?,
                close = ?,
                volume = ?,
                adjustment = ?,
                last_observed_at = ?,
                revision_state = ?,
                quality = ?,
                is_synthetic = ?
            WHERE provider = ? AND ticker = ? AND interval = ? AND bar_end = ?
            """,
            (
                bar.bar_start,
                bar.open,
                bar.high,
                bar.low,
                bar.close,
                bar.volume,
                bar.adjustment,
                bar.last_observed_at,
                bar.revision_state,
                bar.quality,
                1 if bar.is_synthetic else 0,
                bar.provider,
                bar.ticker,
                bar.interval,
                bar.bar_end,
            ),
        )
    except Exception as exc:
        msg = f"bars update failed: {exc}"
        raise DatabaseError(msg) from exc
    return "updated"


def upsert_bars(conn: Any, bars: list[BarRecord]) -> BarsImportResult:
    inserted = 0
    updated = 0
    for bar in bars:
        result = upsert_bar(conn, bar)
        if result == "inserted":
            inserted += 1
        else:
            updated += 1
    return BarsImportResult(inserted=inserted, updated=updated, total_rows=len(bars))


def import_bars_file(
    conn: Any,
    path: Path,
    *,
    default_interval: str = "1d",
    default_provider: str = PROVIDER_LOCAL_IMPORT,
    force_synthetic: bool = False,
) -> BarsImportResult:
    """Import local CSV/JSON/JSONL bars (idempotent upsert)."""
    rows = load_bars_file(path)
    now = to_iso_utc(utc_now())
    records: list[BarRecord] = []
    for raw in rows:
        records.append(
            bar_record_from_mapping(
                raw,
                default_interval=default_interval,
                default_provider=default_provider,
                force_synthetic=True if force_synthetic else None,
                fetched_at=now,
            )
        )
    return upsert_bars(conn, records)


def list_closes_as_of(
    conn: Any,
    *,
    ticker: str,
    interval: str,
    as_of: str,
    provider: str | None = None,
) -> CloseSeries:
    """Completed bars with bar_end <= as_of (point-in-time)."""
    iv = normalize_interval(interval)
    as_of_iso = _normalize_iso(as_of, field="as_of")
    params: list[Any] = [ticker.upper(), iv, as_of_iso]
    provider_clause = ""
    if provider is not None:
        provider_clause = " AND provider = ?"
        params.append(provider)
    try:
        rows = conn.execute(
            f"""
            SELECT bar_end, close, is_synthetic, provider
            FROM bars
            WHERE ticker = ?
              AND interval = ?
              AND bar_end <= ?
              {provider_clause}
            ORDER BY bar_end ASC, provider ASC
            """,
            params,
        ).fetchall()
    except Exception as exc:
        msg = f"bars query failed: {exc}"
        raise DatabaseError(msg) from exc

    # If multiple providers share bar_end, keep last in sort (stable prefer explicit).
    by_end: dict[str, Any] = {}
    for row in rows:
        by_end[str(row["bar_end"])] = row
    ordered = [by_end[k] for k in sorted(by_end)]
    bar_ends = [str(r["bar_end"]) for r in ordered]
    closes = [float(r["close"]) for r in ordered]
    is_synthetic = any(bool(r["is_synthetic"]) for r in ordered) if ordered else False
    return CloseSeries(
        ticker=ticker.upper(),
        interval=iv,
        bar_ends=bar_ends,
        closes=closes,
        is_synthetic=is_synthetic,
    )


def list_all_bar_ends(
    conn: Any,
    *,
    ticker: str,
    interval: str,
    provider: str | None = None,
) -> list[str]:
    series = list_closes_as_of(
        conn,
        ticker=ticker,
        interval=interval,
        as_of="9999-12-31T23:59:59Z",
        provider=provider,
    )
    return series.bar_ends


def bars_counts(conn: Any) -> dict[str, Any]:
    """Aggregate bar counts for ql data status (no private paths)."""
    try:
        total = int(conn.execute("SELECT COUNT(*) AS n FROM bars").fetchone()["n"])
        synthetic = int(
            conn.execute(
                "SELECT COUNT(*) AS n FROM bars WHERE is_synthetic = 1"
            ).fetchone()["n"]
        )
        by_provider_rows = conn.execute(
            """
            SELECT provider, interval, COUNT(*) AS n
            FROM bars
            GROUP BY provider, interval
            ORDER BY provider, interval
            """
        ).fetchall()
        by_ticker_rows = conn.execute(
            """
            SELECT ticker, COUNT(*) AS n
            FROM bars
            GROUP BY ticker
            ORDER BY ticker
            """
        ).fetchall()
    except Exception as exc:
        msg = f"bars status query failed: {exc}"
        raise DatabaseError(msg) from exc
    return {
        "bars_total": total,
        "bars_synthetic": synthetic,
        "bars_non_synthetic": total - synthetic,
        "by_provider_interval": [
            {
                "provider": r["provider"],
                "interval": r["interval"],
                "n": int(r["n"]),
            }
            for r in by_provider_rows
        ],
        "by_ticker": [
            {"ticker": r["ticker"], "n": int(r["n"])} for r in by_ticker_rows
        ],
    }


def new_run_id() -> str:
    return str(uuid.uuid4())


__all__ = [
    "ALLOWED_INTERVALS",
    "DAILY_GATE_HORIZONS",
    "PROVIDER_FIXTURE",
    "PROVIDER_LOCAL_IMPORT",
    "PROVIDER_STOOQ",
    "BarRecord",
    "BarsImportResult",
    "CloseSeries",
    "bar_record_from_mapping",
    "bars_counts",
    "horizon_hours_to_bar_horizon",
    "import_bars_file",
    "list_all_bar_ends",
    "list_closes_as_of",
    "load_bars_file",
    "new_run_id",
    "normalize_interval",
    "parse_stooq_csv",
    "upsert_bar",
    "upsert_bars",
]
