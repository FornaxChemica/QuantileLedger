"""Read-only market data providers (Milestone 2 / Phase WF).

Keyless Stooq daily CSV fetch — no API keys. Default tests use fixtures /
cached bytes and must not require network.
"""

from __future__ import annotations

import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from quantile_ledger.bars import (
    PROVIDER_STOOQ,
    BarsImportResult,
    parse_stooq_csv,
    upsert_bars,
)
from quantile_ledger.errors import MalformedInputError, ProviderError, StaleDataError
from quantile_ledger.timeutil import to_iso_utc, utc_now

# Stooq US tickers use .us suffix (e.g. spy.us).
STOOQ_DAILY_URL = "https://stooq.com/q/d/l/?s={symbol}&i=d"
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_SLEEP_SECONDS = 0.4


@dataclass(frozen=True)
class FetchResult:
    ticker: str
    provider: str
    cache_path: Path
    from_cache: bool
    bytes_read: int
    import_result: BarsImportResult


def stooq_symbol(ticker: str) -> str:
    t = ticker.strip().lower()
    if not t:
        msg = "ticker must be non-empty"
        raise MalformedInputError(msg)
    if "." in t:
        return t
    return f"{t}.us"


def stooq_cache_path(cache_dir: Path, ticker: str) -> Path:
    return cache_dir / "bars" / f"stooq_{ticker.upper()}_1d.csv"


def _cache_is_fresh(path: Path, *, freshness_hours: int) -> bool:
    if not path.is_file() or freshness_hours < 0:
        return False
    age_seconds = time.time() - path.stat().st_mtime
    return age_seconds <= freshness_hours * 3600


def download_url_text(
    url: str,
    *,
    max_retries: int = DEFAULT_MAX_RETRIES,
    retry_sleep_seconds: float = DEFAULT_RETRY_SLEEP_SECONDS,
    timeout_seconds: float = 30.0,
) -> str:
    """Bounded HTTPS GET returning UTF-8 text. No API key headers."""
    last_error: Exception | None = None
    for attempt in range(max(1, max_retries)):
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "QuantileLedger/0.1 (local research; keyless)"},
                method="GET",
            )
            with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
                raw = bytes(resp.read())
            return raw.decode("utf-8", errors="replace")
        except (
            urllib.error.URLError,
            urllib.error.HTTPError,
            TimeoutError,
            OSError,
        ) as exc:
            last_error = exc
            if attempt + 1 < max_retries:
                time.sleep(retry_sleep_seconds * (attempt + 1))
    msg = f"provider download failed after {max_retries} attempts: {last_error}"
    raise ProviderError(msg) from last_error


def fetch_stooq_daily_csv(
    ticker: str,
    *,
    cache_dir: Path,
    force: bool = False,
    freshness_hours: int = 36,
    download_fn: Any | None = None,
) -> tuple[str, Path, bool]:
    """
    Return (csv_text, cache_path, from_cache).

    download_fn is injectable for offline tests: (url: str) -> str.
    """
    cache_dir = cache_dir.expanduser().resolve()
    path = stooq_cache_path(cache_dir, ticker)
    path.parent.mkdir(parents=True, exist_ok=True)

    if not force and _cache_is_fresh(path, freshness_hours=freshness_hours):
        return path.read_text(encoding="utf-8"), path, True

    symbol = stooq_symbol(ticker)
    url = STOOQ_DAILY_URL.format(symbol=symbol)
    fetcher = download_fn or (lambda u: download_url_text(u))
    try:
        text = str(fetcher(url))
    except ProviderError:
        if path.is_file():
            # Stale cache fallback when network fails.
            return path.read_text(encoding="utf-8"), path, True
        raise

    if "Date" not in text and "date" not in text.lower():
        msg = f"Stooq response for {ticker} missing CSV header"
        raise ProviderError(msg)
    path.write_text(text, encoding="utf-8")
    return text, path, False


def fetch_bars(
    conn: Any,
    ticker: str,
    *,
    cache_dir: Path,
    force: bool = False,
    freshness_hours: int = 36,
    download_fn: Any | None = None,
) -> FetchResult:
    """
    Fetch Stooq daily bars into SQLite (provider=stooq, is_synthetic=0).

    Uses local cache under cache_dir/bars/. Injectable download_fn for tests.
    """
    text, path, from_cache = fetch_stooq_daily_csv(
        ticker,
        cache_dir=cache_dir,
        force=force,
        freshness_hours=freshness_hours,
        download_fn=download_fn,
    )
    fetched_at = to_iso_utc(utc_now())
    try:
        records = parse_stooq_csv(text, ticker=ticker.upper(), fetched_at=fetched_at)
    except MalformedInputError as exc:
        msg = f"Stooq CSV malformed for {ticker}: {exc}"
        raise ProviderError(msg) from exc
    if not records:
        msg = f"Stooq returned no bars for {ticker}"
        raise ProviderError(msg)

    result = upsert_bars(conn, records)
    return FetchResult(
        ticker=ticker.upper(),
        provider=PROVIDER_STOOQ,
        cache_path=path,
        from_cache=from_cache,
        bytes_read=len(text.encode("utf-8")),
        import_result=result,
    )


def assert_bars_not_stale(
    last_bar_end: str,
    *,
    freshness_hours: int,
    now: str | None = None,
) -> None:
    """Raise StaleDataError if last bar is older than freshness_hours."""
    from datetime import timedelta

    from quantile_ledger.timeutil import parse_iso_utc

    now_dt = parse_iso_utc(now) if now else utc_now()
    last = parse_iso_utc(last_bar_end)
    if now_dt - last > timedelta(hours=freshness_hours):
        msg = (
            f"last bar_end {last_bar_end} older than "
            f"{freshness_hours}h freshness window"
        )
        raise StaleDataError(msg)


__all__ = [
    "STOOQ_DAILY_URL",
    "FetchResult",
    "assert_bars_not_stale",
    "download_url_text",
    "fetch_bars",
    "fetch_stooq_daily_csv",
    "stooq_cache_path",
    "stooq_symbol",
]
