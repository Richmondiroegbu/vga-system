"""
PipelineWorker — background worker thread that pulls jobs from PipelineQueue.
Spec: VGA Codebase Structure Design v17.2 §core/worker.py
"""
from __future__ import annotations

import logging
import threading
from typing import Callable, Optional

from vga.core.queue import PipelineQueue, PipelineJob

logger = logging.getLogger(__name__)


class PipelineWorker(threading.Thread):
    """Background thread that processes pipeline jobs from the queue."""

    def __init__(
        self,
        job_queue: PipelineQueue,
        handler: Callable[[PipelineJob], None],
        name: str = "vga-worker",
    ) -> None:
        super().__init__(name=name, daemon=True)
        self._queue = job_queue
        self._handler = handler
        self._stop_event = threading.Event()

    def run(self) -> None:
        """Main worker loop — pulls and processes jobs until stopped."""
        logger.info("PipelineWorker %s started", self.name)
        while not self._stop_event.is_set():
            job = self._queue.dequeue(timeout=2.0)
            if job is None:
                continue
            try:
                logger.info("PipelineWorker: processing job %s", job.job_id)
                self._handler(job)
            except Exception as exc:
                logger.error(
                    "PipelineWorker: job %s failed: %s", job.job_id, exc, exc_info=True
                )
            finally:
                self._queue.complete()
        logger.info("PipelineWorker %s stopped", self.name)

    def stop(self) -> None:
        """Signal the worker to stop after the current job completes."""
        self._stop_event.set()
