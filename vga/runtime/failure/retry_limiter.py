"""
RetryLimiter — enforces per-stage retry count limits.
Spec: VGA Runtime Spec v17.2 §failure/retry_limiter.py
"""
from __future__ import annotations

import logging
from collections import defaultdict

from vga.config.settings import settings
from vga.core.exceptions import RetryExhaustedError

logger = logging.getLogger(__name__)


class RetryLimiter:
    """Tracks retry attempts per stage and enforces the maximum retry limit."""

    def __init__(self) -> None:
        self._counts: dict[str, int] = defaultdict(int)

    def increment(self, stage_id: str) -> int:
        """Increment retry count. Returns new count."""
        self._counts[stage_id] += 1
        count = self._counts[stage_id]
        logger.info("RetryLimiter: %s attempt %d/%d", stage_id, count, settings.MAX_RETRIES)
        return count

    def check_limit(self, stage_id: str) -> None:
        """Raise RetryExhaustedError if limit exceeded."""
        if self._counts[stage_id] >= settings.MAX_RETRIES:
            raise RetryExhaustedError(
                f"Stage {stage_id} exceeded max retries ({settings.MAX_RETRIES})",
                stage_id=stage_id,
            )

    def reset(self, stage_id: str) -> None:
        """Reset retry count after successful completion."""
        self._counts.pop(stage_id, None)

    def get_count(self, stage_id: str) -> int:
        return self._counts.get(stage_id, 0)
