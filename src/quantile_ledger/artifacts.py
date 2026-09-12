"""Local artifact digests — relative paths only, no private absolute paths in DB."""

from __future__ import annotations

import hashlib
import json
import uuid
from pathlib import Path
from typing import Any

from quantile_ledger.errors import DatabaseError
from quantile_ledger.timeutil import to_iso_utc, utc_now


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_json(payload: dict[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256_bytes(blob)


def write_json_artifact(
    *,
    artifacts_dir: Path,
    kind: str,
    payload: dict[str, Any],
    filename_stem: str,
) -> tuple[str, str, Path]:
    """
    Write JSON under artifacts_dir and return (artifact_id, digest, relative_path).

    relative_path is relative to artifacts_dir's parent (data_dir) when possible,
    otherwise relative to artifacts_dir itself — never an absolute home path.
    """
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    digest = sha256_json(payload)
    safe_stem = "".join(c if c.isalnum() or c in "-_" else "_" for c in filename_stem)
    filename = f"{safe_stem}-{digest[:12]}.json"
    path = artifacts_dir / filename
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    # Prefer path relative to data_dir (parent of artifacts/).
    data_dir = artifacts_dir.parent
    try:
        relative = str(path.relative_to(data_dir))
    except ValueError:
        relative = str(Path(artifacts_dir.name) / filename)
    artifact_id = str(uuid.uuid4())
    return artifact_id, digest, Path(relative)


def register_artifact(
    conn: Any,
    *,
    artifact_id: str,
    digest: str,
    kind: str,
    relative_path: str,
    experiment_id: str | None = None,
    model_version_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    import sqlite3

    try:
        conn.execute(
            """
            INSERT INTO artifacts (
                artifact_id, digest, kind, relative_path, created_at,
                experiment_id, model_version_id, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(digest) DO NOTHING
            """,
            (
                artifact_id,
                digest,
                kind,
                relative_path,
                to_iso_utc(utc_now()),
                experiment_id,
                model_version_id,
                json.dumps(metadata or {}, sort_keys=True),
            ),
        )
    except sqlite3.Error as exc:
        msg = f"artifact register failed: {exc}"
        raise DatabaseError(msg) from exc


def optional_dependency_status() -> dict[str, str]:
    """Report optional stacks without revealing install paths."""
    import importlib.util

    from quantile_ledger.kronos_quantile import kronos_optional_status
    from quantile_ledger.sentiment import finbert_optional_status

    names = (
        "torch",
        "mamba_ssm",
        "transformers",
        "pytorch_forecasting",
        "streamlit",
    )
    status: dict[str, str] = {"core": "available"}
    for name in names:
        status[name] = (
            "available"
            if importlib.util.find_spec(name) is not None
            else "not_installed"
        )
    # K0 fake adapter is always available; real Kronos stack is optional.
    status.update(kronos_optional_status())
    status["tft_local_adapter"] = "available"
    status.update(finbert_optional_status())
    return status
