"""
AdaptiveMemory — records stage success/failure history for calibration.
Spec: VGA System Architecture v17.2 §10 (Adaptive Subsystem); FR-450–FR-460
"""
from __future__ import annotations

import logging
from collections import defaultdict
from typing import Dict, List

logger = logging.getLogger(__name__)


class AdaptiveMemory:
    """Stores per-stage success/failure records for CalibrationEngine use."""

    def __init__(self) -> None:
        self._successes: Dict[str, int] = defaultdict(int)
        self._failures: Dict[str, int] = defaultdict(int)
        self._clip_scores: Dict[str, List[float]] = defaultdict(list)

    def record_success(self, stage_id: str, clip_score: float | None = None) -> None:
        self._successes[stage_id] += 1
        if clip_score is not None:
            self._clip_scores[stage_id].append(clip_score)

    def record_failure(self, stage_id: str) -> None:
        self._failures[stage_id] += 1

    def success_rate(self, stage_id: str) -> float:
        total = self._successes[stage_id] + self._failures[stage_id]
        return self._successes[stage_id] / total if total else 1.0

    def mean_clip_score(self, stage_id: str) -> float | None:
        scores = self._clip_scores[stage_id]
        return sum(scores) / len(scores) if scores else None
