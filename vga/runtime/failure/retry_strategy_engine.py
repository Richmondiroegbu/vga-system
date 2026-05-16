"""
RetryStrategyEngine — selects the appropriate retry strategy per failure type.
Spec: VGA Runtime Spec v17.2 §failure/retry_strategy_engine.py
"""
from __future__ import annotations

from dataclasses import dataclass

from vga.config.settings import settings
from vga.models.enums import FailureSeverity


@dataclass
class RetryStrategy:
    should_retry: bool
    max_retries: int
    backoff_seconds: list[float]
    description: str


def get_strategy(severity: FailureSeverity, exc: BaseException) -> RetryStrategy:
    """Return the retry strategy for a given failure severity."""
    if severity == FailureSeverity.CRITICAL:
        return RetryStrategy(
            should_retry=False,
            max_retries=0,
            backoff_seconds=[],
            description="CRITICAL — no retry, pipeline halts",
        )
    if severity == FailureSeverity.DEGRADED:
        return RetryStrategy(
            should_retry=True,
            max_retries=settings.MAX_RETRIES,
            backoff_seconds=settings.BACKOFF_SECONDS,
            description=f"DEGRADED — retry up to {settings.MAX_RETRIES} times",
        )
    return RetryStrategy(
        should_retry=True,
        max_retries=1,
        backoff_seconds=[2.0],
        description="WARNING — single retry then continue",
    )
