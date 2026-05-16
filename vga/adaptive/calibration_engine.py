"""
CalibrationEngine — adjusts pipeline parameters based on AdaptiveMemory history.
Spec: VGA System Architecture v17.2 §10; FR-461–FR-470
"""
from __future__ import annotations

import logging

from vga.adaptive.adaptive_memory import AdaptiveMemory
from vga.config.settings import settings

logger = logging.getLogger(__name__)


class CalibrationEngine:
    """Reads AdaptiveMemory to suggest parameter adjustments for upcoming stages."""

    def __init__(self, memory: AdaptiveMemory) -> None:
        self._memory = memory

    def recommended_cfg(self, stage_id: str) -> float:
        """Return recommended SVI CFG based on historical clip scores."""
        mean_score = self._memory.mean_clip_score(stage_id)
        if mean_score is None:
            return settings.SVI_CFG_DEFAULT
        if mean_score < settings.CLIP_IDENTITY_THRESHOLD:
            # Lower clip score → increase CFG for stronger conditioning
            return min(settings.SVI_CFG_DEFAULT + 0.2, settings.SVI_CFG_MAX)
        return settings.SVI_CFG_DEFAULT

    def recommended_steps(self, stage_id: str, critical: bool = False) -> int:
        """Return recommended diffusion step count."""
        rate = self._memory.success_rate(stage_id)
        if rate < 0.7 or critical:
            return settings.STEPS_CRITICAL
        return settings.STEPS_STANDARD
