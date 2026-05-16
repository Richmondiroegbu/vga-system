"""
SessionHealthMonitor — monitors end-to-end pipeline session health.
Spec: VGA Runtime Spec v17.2 §resilience/session_health_monitor.py
"""
from __future__ import annotations

import logging
import time
from typing import Dict

logger = logging.getLogger(__name__)


class SessionHealthMonitor:
    """Tracks health signals across a full pipeline session."""

    def __init__(self) -> None:
        self._session_start = time.monotonic()
        self._stage_times: Dict[str, float] = {}
        self._warnings: list[str] = []

    def record_stage_time(self, stage_id: str, elapsed_s: float) -> None:
        self._stage_times[stage_id] = elapsed_s

    def add_warning(self, warning: str) -> None:
        self._warnings.append(warning)
        logger.warning("SessionHealth warning: %s", warning)

    @property
    def session_elapsed_s(self) -> float:
        return time.monotonic() - self._session_start

    @property
    def has_warnings(self) -> bool:
        return len(self._warnings) > 0

    def health_report(self) -> dict:
        return {
            "session_elapsed_s": round(self.session_elapsed_s, 2),
            "stage_times": self._stage_times,
            "warnings": self._warnings,
            "warning_count": len(self._warnings),
        }
