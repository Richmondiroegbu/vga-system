"""
StrategyOptimizer — selects optimal generation strategies based on historical patterns.
Spec: VGA System Architecture v17.2 §10 (Adaptive Subsystem)
"""
from __future__ import annotations

import logging

from vga.config.settings import settings

logger = logging.getLogger(__name__)


class StrategyOptimizer:
    """Recommends generation strategy adjustments based on historical performance."""

    def recommend_svi_steps(self, stage_id: str, history_success_rate: float) -> int:
        """Recommend step count for SVI based on success rate."""
        if history_success_rate < 0.70:
            return settings.STEPS_CRITICAL    # 50 — more steps for struggling stages
        return settings.STEPS_STANDARD        # 30

    def recommend_svi_cfg(self, mean_clip_score: float | None) -> float:
        """Recommend SVI CFG based on mean CLIP score from history."""
        if mean_clip_score is None:
            return settings.SVI_CFG_DEFAULT
        if mean_clip_score < settings.CLIP_IDENTITY_THRESHOLD:
            return min(settings.SVI_CFG_DEFAULT + 0.3, settings.SVI_CFG_MAX)
        return settings.SVI_CFG_DEFAULT

    def recommend_lora_weight(self, drift_score: float) -> float:
        """Recommend LoRA weight for FLUX based on current identity drift."""
        if drift_score > 0.10:
            return settings.FLUX_IDENTITY_LORA_WEIGHT_MAX   # stronger identity preservation
        return (settings.FLUX_IDENTITY_LORA_WEIGHT_MIN + settings.FLUX_IDENTITY_LORA_WEIGHT_MAX) / 2
