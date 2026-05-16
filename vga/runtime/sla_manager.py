"""
SLAManager — tracks and enforces per-stage SLA time budgets.
Spec: VGA System Requirements v17.2 §7 (Non-Functional Requirements); NFR-001–NFR-020
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

from vga.config.settings import settings
from vga.core.exceptions import SLAViolationError

logger = logging.getLogger(__name__)

# SLA budget (seconds) per stage_id
_SLA_BUDGETS: Dict[str, float] = {
    "S-01": settings.SLA_SCRIPT_MAX_S,
    "S-02": settings.SLA_SCENE_PLAN_MAX_S,
    "S-03": settings.SLA_IDENTITY_DESIGN_MAX_S,
    "S-04": settings.SLA_COMPOSITION_MAX_S,
    "S-05": settings.SLA_BASE_IMAGE_MAX_S,
    "S-06": settings.SLA_BASE_IMAGE_MAX_S,
    "S-07": settings.SLA_BASE_IMAGE_MAX_S,
    "S-08": settings.SLA_SEGMENT_GEN_MAX_S,
    "S-09": settings.SLA_SEGMENT_GEN_CRITICAL_MAX_S,
    "S-10": settings.SLA_SCENE_PLAN_MAX_S,
    "S-11": settings.SLA_LIPSYNC_MAX_S,
    "S-12": settings.SLA_LIPSYNC_MAX_S,
    "S-13": settings.SLA_AUDIO_MIX_MAX_S,
    "S-14": settings.SLA_AUDIO_MIX_MAX_S,
    "S-15": settings.SLA_AUDIO_MIX_MAX_S,
    "S-16": settings.SLA_EXPORT_MAX_S,
}


class SLAManager:
    """Records actual stage durations and checks against SLA budgets."""

    def __init__(self) -> None:
        self._records: Dict[str, List[float]] = {}

    def record(self, stage_id: str, elapsed_s: float) -> None:
        """Record elapsed time for stage_id. Logs warning if SLA exceeded."""
        self._records.setdefault(stage_id, []).append(elapsed_s)
        budget = _SLA_BUDGETS.get(stage_id)
        if budget is not None and elapsed_s > budget:
            logger.warning(
                "SLA exceeded: stage=%s elapsed=%.1fs budget=%.1fs",
                stage_id, elapsed_s, budget,
            )

    def get_durations(self, stage_id: str) -> List[float]:
        """Return all recorded durations for a stage."""
        return self._records.get(stage_id, [])

    def summary(self) -> Dict[str, float]:
        """Return dict of stage_id → mean elapsed seconds."""
        return {
            sid: sum(times) / len(times)
            for sid, times in self._records.items()
            if times
        }
