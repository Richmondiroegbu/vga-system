"""
ExecutionScheduler — schedules pipeline stage execution with resource awareness.
Spec: VGA Runtime Spec v17.2 §resilience/execution_scheduler.py
"""
from __future__ import annotations

import logging
import time
from typing import Callable

from vga.runtime.resilience.resource_monitor import ResourceMonitor

logger = logging.getLogger(__name__)


class ExecutionScheduler:
    """Schedules stage execution, waiting for resources if needed."""

    def __init__(self, resource_monitor: ResourceMonitor | None = None) -> None:
        self._monitor = resource_monitor or ResourceMonitor()

    def wait_for_resources(self, max_wait_s: float = 60.0) -> bool:
        """Block until resources are healthy or timeout expires.

        Returns:
            True if resources became healthy, False if timed out
        """
        deadline = time.monotonic() + max_wait_s
        while time.monotonic() < deadline:
            if self._monitor.is_healthy():
                return True
            logger.info("ExecutionScheduler: waiting for resources...")
            time.sleep(5.0)
        return False

    def schedule(self, fn: Callable, stage_id: str) -> object:
        """Execute fn after confirming resources are available."""
        if not self.wait_for_resources():
            logger.warning(
                "ExecutionScheduler: resource wait timed out for %s — proceeding anyway",
                stage_id,
            )
        return fn()
