"""
Structured logging for VGA v17.2.
All pipeline events are JSON lines written to /workspace/logs/.
Console output uses human-readable formatting.
Spec: VGA Coding Standards RULE-18 (no print()), RULE-19 (structured logging)
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Attempt to use the settings log path; fall back to /tmp for local dev
try:
    from vga.config.settings import settings
    _LOG_DIR = settings.LOGS_DIR
except Exception:
    _LOG_DIR = Path("/tmp/vga_logs")

_LOG_DIR_STR = str(_LOG_DIR)


class _JSONFormatter(logging.Formatter):
    """Formats log records as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        # Attach extra VGA context fields if present
        for field in ("stage_id", "scene_id", "job_id", "segment_id"):
            if hasattr(record, field):
                payload[field] = getattr(record, field)

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, default=str)


class _HumanFormatter(logging.Formatter):
    """Human-readable console formatter."""

    LEVEL_COLORS = {
        "DEBUG": "\033[36m",
        "INFO": "\033[32m",
        "WARNING": "\033[33m",
        "ERROR": "\033[31m",
        "CRITICAL": "\033[35m",
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        color = self.LEVEL_COLORS.get(record.levelname, "")
        ts = datetime.now(tz=timezone.utc).strftime("%H:%M:%S")
        stage = getattr(record, "stage_id", "")
        scene = getattr(record, "scene_id", "")
        ctx = f"[{stage}:{scene}]" if stage or scene else ""
        msg = record.getMessage()
        line = f"{color}{ts} {record.levelname:8s}{self.RESET} {ctx} {msg}"
        if record.exc_info:
            line += "\n" + self.formatException(record.exc_info)
        return line


def _build_file_handler(log_dir: Path) -> logging.FileHandler | None:
    """Create a rotating-like JSON file handler, returns None if dir unavailable."""
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        date_str = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
        log_path = log_dir / f"vga_{date_str}.jsonl"
        handler = logging.FileHandler(log_path, encoding="utf-8")
        handler.setFormatter(_JSONFormatter())
        return handler
    except (OSError, PermissionError):
        return None


def _configure_root() -> None:
    """Configure root logging once (idempotent)."""
    root = logging.getLogger()
    if root.handlers:
        return  # already configured

    root.setLevel(logging.DEBUG)

    # Console handler (human-readable)
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(_HumanFormatter())
    root.addHandler(console)

    # File handler (JSON lines)
    fh = _build_file_handler(Path(_LOG_DIR_STR))
    if fh is not None:
        fh.setLevel(logging.DEBUG)
        root.addHandler(fh)


_configure_root()


def get_logger(name: str) -> logging.Logger:
    """Return a named VGA logger.

    Usage:
        logger = get_logger(__name__)
        logger.info("Stage started", extra={"stage_id": "S-01", "scene_id": "sc_001"})
    """
    return logging.getLogger(name)


def trace_event(
    event_name: str,
    stage_id: str | None = None,
    scene_id: str | None = None,
    job_id: str | None = None,
    **kwargs: Any,
) -> None:
    """Log a pipeline trace event as INFO level with structured fields."""
    logger = get_logger("vga.trace")
    extra: dict[str, Any] = {}
    if stage_id:
        extra["stage_id"] = stage_id
    if scene_id:
        extra["scene_id"] = scene_id
    if job_id:
        extra["job_id"] = job_id
    extra.update(kwargs)
    logger.info("TRACE:%s", event_name, extra=extra)
