"""Local logging setup without telemetry or remote handlers."""

from __future__ import annotations

import logging
import time
from pathlib import Path


def configure_logging(*, level: str = "INFO", log_file: Path | None = None) -> None:
    """Configure root logging for local CLI use."""
    root = logging.getLogger()
    if root.handlers:
        return

    numeric = getattr(logging, level.upper(), logging.INFO)
    root.setLevel(numeric)

    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%SZ",
    )
    formatter.converter = time.gmtime

    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    root.addHandler(stream)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)
