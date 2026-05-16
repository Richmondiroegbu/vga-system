"""
Metrics — lightweight pipeline performance metrics collection.
Spec: VGA Observability Spec v17.2
"""
from __future__ import annotations

import logging
import time
from collections import defaultdict
from typing import Dict, List

logger = logging.getLogger(__name__)


class Metrics:
    """Collects timing and count metrics for pipeline stages."""

    def __init__(self) -> None:
        self._timings: Dict[str, List[float]] = defaultdict(list)
        self._counts: Dict[str, int] = defaultdict(int)

    def record_timing(self, name: str, elapsed_s: float) -> None:
        self._timings[name].append(elapsed_s)

    def increment(self, name: str) -> None:
        self._counts[name] += 1

    def summary(self) -> dict:
        return {
            "timings": {
                k: {"mean": sum(v) / len(v), "n": len(v), "total": sum(v)}
                for k, v in self._timings.items()
            },
            "counts": dict(self._counts),
        }
