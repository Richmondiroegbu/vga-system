"""
PipelineQueue — in-memory job queue with priority ordering.
Spec: VGA Codebase Structure Design v17.2 §core/queue.py
"""
from __future__ import annotations

import logging
import queue
import threading
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass(order=True)
class PipelineJob:
    """A queued pipeline job with priority ordering (lower = higher priority)."""
    priority: int
    job_id: str = field(compare=False)
    payload: Any = field(compare=False, default=None)


class PipelineQueue:
    """Thread-safe priority queue for VGA pipeline jobs."""

    def __init__(self) -> None:
        self._queue: queue.PriorityQueue[PipelineJob] = queue.PriorityQueue()
        self._lock = threading.Lock()
        self._active_count = 0

    def enqueue(self, job_id: str, payload: Any = None, priority: int = 10) -> None:
        """Add a job to the queue."""
        job = PipelineJob(priority=priority, job_id=job_id, payload=payload)
        self._queue.put(job)
        logger.info("PipelineQueue: enqueued job %s (priority=%d)", job_id, priority)

    def dequeue(self, timeout: float = 5.0) -> Optional[PipelineJob]:
        """Remove and return the highest-priority job. Returns None on timeout."""
        try:
            job = self._queue.get(timeout=timeout)
            with self._lock:
                self._active_count += 1
            return job
        except queue.Empty:
            return None

    def complete(self) -> None:
        """Signal that the current job is complete."""
        self._queue.task_done()
        with self._lock:
            self._active_count = max(0, self._active_count - 1)

    @property
    def size(self) -> int:
        return self._queue.qsize()

    @property
    def active(self) -> int:
        return self._active_count

    def is_empty(self) -> bool:
        return self._queue.empty()
