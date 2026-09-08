"""Timezone-aware UTC helpers. Never rely on the workstation local timezone."""

from __future__ import annotations

from datetime import UTC, datetime


def ensure_utc(value: datetime) -> datetime:
    """Return a timezone-aware UTC datetime; reject naive inputs."""
    if value.tzinfo is None:
        msg = "Naive datetime rejected; pass timezone-aware UTC timestamps."
        raise ValueError(msg)
    return value.astimezone(UTC)


def utc_now() -> datetime:
    """Current time in UTC."""
    return datetime.now(UTC)


def to_iso_utc(value: datetime) -> str:
    """Serialize as unambiguous ISO 8601 UTC with trailing Z."""
    aware = ensure_utc(value)
    return aware.strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_iso_utc(value: str) -> datetime:
    """Parse ISO 8601 into timezone-aware UTC."""
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    parsed = datetime.fromisoformat(text)
    return ensure_utc(parsed)
