"""Local news import and point-in-time eligibility (no network fetch)."""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from quantile_ledger.errors import DatabaseError, MalformedInputError
from quantile_ledger.timeutil import parse_iso_utc, to_iso_utc, utc_now

SentimentAvailability = Literal["missing", "unscored", "scored"]


@dataclass(frozen=True)
class NewsItem:
    news_id: str
    ticker: str
    headline: str
    published_at: str
    ingested_at: str
    content_hash: str
    dedupe_key: str
    source: str | None = None
    url: str | None = None
    is_retrospective: bool = False
    quality: str = "ok"
    is_synthetic: bool = False


@dataclass(frozen=True)
class NewsImportResult:
    inserted: int
    skipped_duplicate: int
    total_rows: int


def _normalize_iso(value: str, *, field: str) -> str:
    try:
        return to_iso_utc(parse_iso_utc(value))
    except (ValueError, TypeError) as exc:
        msg = f"invalid {field} timestamp: {value!r}"
        raise MalformedInputError(msg) from exc


def content_hash_for(headline: str, published_at: str, ticker: str) -> str:
    blob = f"{ticker.upper()}|{published_at}|{headline.strip()}".encode()
    return hashlib.sha256(blob).hexdigest()


def dedupe_key_for(
    *,
    ticker: str,
    published_at: str,
    content_hash: str,
    source: str | None,
) -> str:
    src = (source or "").strip().lower()
    return f"{ticker.upper()}|{published_at}|{content_hash[:16]}|{src}"


def news_item_from_mapping(
    raw: dict[str, Any],
    *,
    default_ingested_at: str | None = None,
    force_synthetic: bool | None = None,
) -> NewsItem:
    """Parse one local news row (JSON object)."""
    try:
        ticker = str(raw["ticker"]).strip().upper()
        headline = str(raw["headline"]).strip()
        published_at = _normalize_iso(str(raw["published_at"]), field="published_at")
    except KeyError as exc:
        msg = f"news row missing required field: {exc}"
        raise MalformedInputError(msg) from exc
    if not ticker or not headline:
        msg = "ticker and headline must be non-empty"
        raise MalformedInputError(msg)

    if "ingested_at" in raw and raw["ingested_at"] is not None:
        ingested_at = _normalize_iso(str(raw["ingested_at"]), field="ingested_at")
    else:
        ingested_at = default_ingested_at or to_iso_utc(utc_now())

    # Point-in-time: ingested_at must not precede published_at for non-retrospective.
    is_retrospective = bool(raw.get("is_retrospective", False))
    if not is_retrospective and ingested_at < published_at:
        msg = (
            f"ingested_at {ingested_at} precedes published_at {published_at}; "
            "set is_retrospective=1 only for labeled backfill rows"
        )
        raise MalformedInputError(msg)

    source = str(raw["source"]).strip() if raw.get("source") else None
    url = str(raw["url"]).strip() if raw.get("url") else None
    chash = content_hash_for(headline, published_at, ticker)
    if raw.get("content_hash"):
        chash = str(raw["content_hash"])
    dkey = (
        str(raw["dedupe_key"])
        if raw.get("dedupe_key")
        else dedupe_key_for(
            ticker=ticker,
            published_at=published_at,
            content_hash=chash,
            source=source,
        )
    )
    is_synthetic = (
        bool(force_synthetic)
        if force_synthetic is not None
        else bool(raw.get("is_synthetic", False))
    )
    return NewsItem(
        news_id=str(raw.get("news_id") or uuid.uuid4()),
        ticker=ticker,
        headline=headline,
        published_at=published_at,
        ingested_at=ingested_at,
        content_hash=chash,
        dedupe_key=dkey,
        source=source,
        url=url,
        is_retrospective=is_retrospective,
        quality=str(raw.get("quality") or "ok"),
        is_synthetic=is_synthetic,
    )


def load_news_file(path: Path) -> list[dict[str, Any]]:
    """Load JSON array or JSONL from a local file (no network)."""
    if not path.is_file():
        msg = f"news file not found: {path}"
        raise MalformedInputError(msg)
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if path.suffix.lower() == ".jsonl" or (not text.startswith("[") and "\n" in text):
        rows: list[dict[str, Any]] = []
        for line_no, line in enumerate(text.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
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
        msg = f"invalid JSON news file: {exc}"
        raise MalformedInputError(msg) from exc
    if isinstance(payload, dict) and "items" in payload:
        payload = payload["items"]
    if not isinstance(payload, list):
        msg = "news JSON must be an array or {items: [...]}"
        raise MalformedInputError(msg)
    out: list[dict[str, Any]] = []
    for i, obj in enumerate(payload):
        if not isinstance(obj, dict):
            msg = f"news item {i} must be an object"
            raise MalformedInputError(msg)
        out.append(obj)
    return out


def insert_news_item(conn: Any, item: NewsItem) -> bool:
    """Insert one news item. Returns True if inserted, False if duplicate dedupe_key."""
    import sqlite3

    try:
        conn.execute(
            """
            INSERT INTO news_items (
                news_id, ticker, headline, source, url, published_at, ingested_at,
                content_hash, dedupe_key, is_retrospective, quality, is_synthetic
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                item.news_id,
                item.ticker,
                item.headline,
                item.source,
                item.url,
                item.published_at,
                item.ingested_at,
                item.content_hash,
                item.dedupe_key,
                1 if item.is_retrospective else 0,
                item.quality,
                1 if item.is_synthetic else 0,
            ),
        )
        return True
    except sqlite3.IntegrityError as exc:
        # Only treat unique-constraint collisions as idempotent skips.
        if "UNIQUE" in str(exc).upper() or "unique" in str(exc).lower():
            return False
        msg = f"news_items insert failed: {exc}"
        raise DatabaseError(msg) from exc


def import_news_file(
    conn: Any,
    path: Path,
    *,
    force_synthetic: bool = False,
) -> NewsImportResult:
    """Import local news JSON/JSONL into news_items (idempotent on dedupe_key)."""
    rows = load_news_file(path)
    ingested_default = to_iso_utc(utc_now())
    inserted = 0
    skipped = 0
    for raw in rows:
        item = news_item_from_mapping(
            raw,
            default_ingested_at=ingested_default,
            force_synthetic=True if force_synthetic else None,
        )
        if insert_news_item(conn, item):
            inserted += 1
        else:
            skipped += 1
    return NewsImportResult(
        inserted=inserted,
        skipped_duplicate=skipped,
        total_rows=len(rows),
    )


def assert_news_pit_eligible(
    *,
    published_at: str,
    ingested_at: str,
    issued_at: str,
) -> None:
    """Raise if news timestamps violate forecast issuance cutoffs."""
    pub = _normalize_iso(published_at, field="published_at")
    ing = _normalize_iso(ingested_at, field="ingested_at")
    issued = _normalize_iso(issued_at, field="issued_at")
    if pub > issued:
        msg = f"news leakage: published_at {pub} > issued_at {issued}"
        raise MalformedInputError(msg)
    if ing > issued:
        msg = f"news leakage: ingested_at {ing} > issued_at {issued}"
        raise MalformedInputError(msg)


def list_eligible_news(
    conn: Any,
    *,
    ticker: str,
    issued_at: str,
) -> list[dict[str, Any]]:
    """News visible at issued_at: published_at and ingested_at both <= issued_at."""
    issued = _normalize_iso(issued_at, field="issued_at")
    rows = conn.execute(
        """
        SELECT news_id, ticker, headline, source, url, published_at, ingested_at,
               content_hash, dedupe_key, is_retrospective, quality, is_synthetic
        FROM news_items
        WHERE ticker = ?
          AND published_at <= ?
          AND ingested_at <= ?
        ORDER BY published_at ASC, news_id ASC
        """,
        (ticker.upper(), issued, issued),
    ).fetchall()
    return [dict(row) for row in rows]


def news_counts(conn: Any) -> dict[str, int]:
    try:
        n_items = int(
            conn.execute("SELECT COUNT(*) AS n FROM news_items").fetchone()["n"]
        )
        n_sent = int(
            conn.execute("SELECT COUNT(*) AS n FROM news_sentiment").fetchone()["n"]
        )
    except Exception as exc:
        msg = f"news status query failed: {exc}"
        raise DatabaseError(msg) from exc
    return {"news_items": n_items, "news_sentiment": n_sent}


__all__ = [
    "NewsImportResult",
    "NewsItem",
    "SentimentAvailability",
    "assert_news_pit_eligible",
    "content_hash_for",
    "dedupe_key_for",
    "import_news_file",
    "insert_news_item",
    "list_eligible_news",
    "load_news_file",
    "news_counts",
    "news_item_from_mapping",
]
