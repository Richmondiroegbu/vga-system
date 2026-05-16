"""
RetryHelper — exponential backoff retry utility.
All pipeline retries use this module. Max 3 retries with backoff [5, 15, 45]s.
Spec: VGA Coding Standards RULE-09 (retry with backoff)
"""
from __future__ import annotations

import logging
import time
from typing import Callable, TypeVar

from vga.config.settings import settings
from vga.core.exceptions import RetryExhaustedError

logger = logging.getLogger(__name__)

T = TypeVar("T")


def with_retry(
    fn: Callable[[], T],
    max_retries: int | None = None,
    backoff: list[float] | None = None,
    stage_id: str = "",
    error_label: str = "",
) -> T:
    """Execute fn with exponential backoff retry.

    Args:
        fn:          callable to execute (no args — use lambda or partial)
        max_retries: max attempts (defaults to settings.MAX_RETRIES = 3)
        backoff:     wait times in seconds (defaults to settings.BACKOFF_SECONDS)
        stage_id:    for logging
        error_label: human-readable operation name for error messages

    Returns:
        Result of fn() on success

    Raises:
        RetryExhaustedError after all attempts fail
    """
    max_retries = max_retries or settings.MAX_RETRIES
    backoff = backoff or settings.BACKOFF_SECONDS
    last_exc: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            return fn()
        except Exception as exc:
            last_exc = exc
            if attempt == max_retries:
                break
            wait = backoff[min(attempt - 1, len(backoff) - 1)]
            logger.warning(
                "Retry %d/%d for %s [%s]: %s — waiting %.0fs",
                attempt, max_retries, error_label or "operation", stage_id, exc, wait,
            )
            time.sleep(wait)

    raise RetryExhaustedError(
        f"{error_label or 'operation'} failed after {max_retries} attempts: {last_exc}",
        stage_id=stage_id,
    )
