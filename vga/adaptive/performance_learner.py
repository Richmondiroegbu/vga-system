"""
PerformanceLearner — learns per-stage performance patterns over time.
Feeds into CalibrationEngine for SLA threshold adjustment.
Spec: VGA System Architecture v17.2 §10 (Adaptive Subsystem)
"""
from __future__ import annotations

import logging
from collections import defaultdict
from typing import Dict, List

logger = logging.getLogger(__name__)


class PerformanceLearner:
    """Tracks stage execution patterns and computes performance baselines."""

    def __init__(self) -> None:
        self._timings: Dict[str, List[float]] = defaultdict(list)
        self._clip_scores: Dict[str, List[float]] = defaultdict(list)

    def record(self, stage_id: str, elapsed_s: float, clip_score: float | None = None) -> None:
        self._timings[stage_id].append(elapsed_s)
        if clip_score is not None:
            self._clip_scores[stage_id].append(clip_score)

    def baseline_sla(self, stage_id: str) -> float | None:
        """Return the 90th percentile execution time for a stage (SLA baseline)."""
        times = self._timings.get(stage_id, [])
        if not times:
            return None
        sorted_times = sorted(times)
        p90_idx = int(len(sorted_times) * 0.9)
        return sorted_times[min(p90_idx, len(sorted_times) - 1)]

    def mean_clip_score(self, stage_id: str) -> float | None:
        scores = self._clip_scores.get(stage_id, [])
        return sum(scores) / len(scores) if scores else None

    def summary(self) -> dict:
        return {
            stage: {
                "n_runs": len(times),
                "mean_s": sum(times) / len(times),
                "p90_s": self.baseline_sla(stage),
                "mean_clip": self.mean_clip_score(stage),
            }
            for stage, times in self._timings.items()
        }
