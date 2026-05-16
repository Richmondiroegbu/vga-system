"""
AsyncIOManager — manages async file I/O operations for large artifact writes.
Spec: VGA Runtime Spec v17.2 §resilience/async_io_manager.py
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)


class AsyncIOManager:
    """Performs artifact writes in background threads to avoid blocking the pipeline."""

    def __init__(self) -> None:
        self._threads: list[threading.Thread] = []

    def write_async(self, write_fn: Callable, label: str = "") -> None:
        """Execute write_fn in a background thread."""
        def _run():
            try:
                write_fn()
                logger.debug("AsyncIOManager: write complete — %s", label)
            except Exception as exc:
                logger.error("AsyncIOManager: write failed [%s]: %s", label, exc)

        thread = threading.Thread(target=_run, daemon=True, name=f"vga-io-{label}")
        thread.start()
        self._threads.append(thread)

    def wait_all(self, timeout: float = 30.0) -> None:
        """Wait for all pending writes to complete."""
        for t in self._threads:
            t.join(timeout=timeout)
        self._threads.clear()
