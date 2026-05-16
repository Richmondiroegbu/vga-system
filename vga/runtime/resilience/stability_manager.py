"""
StabilityManager — tracks and maintains pipeline stability signals.
Spec: VGA Runtime Spec v17.2 §resilience/stability_manager.py
"""
from __future__ import annotations

import logging
from collections import deque
from typing import Deque

logger = logging.getLogger(__name__)

_WINDOW = 10   # rolling window for stability assessment


class StabilityManager:
    """Tracks stage success/failure history to assess pipeline stability."""

    def __init__(self) -> None:
        self._outcomes: Deque[bool] = deque(maxlen=_WINDOW)
        self._consecutive_failures = 0

    def record_success(self) -> None:
        self._outcomes.append(True)
        self._consecutive_failures = 0

    def record_failure(self) -> None:
        self._outcomes.append(False)
        self._consecutive_failures += 1

    @property
    def success_rate(self) -> float:
        if not self._outcomes:
            return 1.0
        return sum(self._outcomes) / len(self._outcomes)

    @property
    def is_stable(self) -> bool:
        return self.success_rate >= 0.70 and self._consecutive_failures < 3

    @property
    def consecutive_failures(self) -> int:
        return self._consecutive_failures

    def stability_level(self) -> str:
        if self.success_rate >= 0.90:
            return "STABLE"
        elif self.success_rate >= 0.70:
            return "DEGRADED"
        return "UNSTABLE"
