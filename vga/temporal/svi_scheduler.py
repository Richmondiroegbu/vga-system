"""
SVIScheduler — dynamic LoRA weight scheduling per SVI diffusion timestep.
STATIC LoRA weight is FORBIDDEN. Weight MUST vary per noise phase. RULE-86, FR-932–FR-934.
Spec: VGA Temporal Engine Spec v17.2 §SVI Scheduler; RULE-86, FR-932–FR-934
"""
from __future__ import annotations

import logging

from vga.config.settings import settings
from vga.core.exceptions import SVICFGViolationError, SVISchedulerViolationError
from vga.models.enums import TemporalPhase

logger = logging.getLogger(__name__)


class SVIScheduler:
    """Implements dynamic LoRA weight scheduling for SVI Pro 2.

    Phase mapping (RULE-86):
      HIGH_NOISE: t > 0.67 * T  → weight = 0.6 (structure + motion)
      MID_NOISE:  0.33 * T < t ≤ 0.67 * T → weight = 0.5
      LOW_NOISE:  t ≤ 0.33 * T → weight = 0.4 (detail preservation)

    FORBIDDEN: using the same weight for all timesteps (SVISchedulerViolationError).
    """

    def __init__(self, cfg: float, steps: int, critical: bool = False) -> None:
        """Initialize SVIScheduler.

        Args:
            cfg:      classifier-free guidance scale — MUST be in [5.0, 6.0]
            steps:    total diffusion steps (30 minimum for quality)
            critical: if True, uses STEPS_CRITICAL (50); else uses steps

        Raises:
            SVICFGViolationError if cfg is outside [5.0, 6.0]
        """
        self._cfg = self._validate_cfg(cfg)
        self._steps = max(steps, settings.STEPS_STANDARD)   # floor at 30
        self._critical = critical
        self._high_threshold = settings.HIGH_NOISE_FRACTION   # 0.67
        self._mid_threshold = settings.MID_NOISE_FRACTION     # 0.33
        logger.debug("SVIScheduler: cfg=%.2f steps=%d", self._cfg, self._steps)

    @property
    def cfg(self) -> float:
        return self._cfg

    def get_steps(self) -> int:
        """Return step count. Critical segments use STEPS_CRITICAL (50)."""
        if self._critical:
            return settings.STEPS_CRITICAL
        return self._steps

    def get_lora_weight(self, timestep_index: int) -> float:
        """Return dynamic LoRA weight for the given timestep index. RULE-86.

        MUST be called per-timestep — precomputing a static weight FORBIDDEN.

        Args:
            timestep_index: current step index (0 = first, T-1 = last)

        Returns:
            float LoRA weight (0.4, 0.5, or 0.6)
        """
        phase = self.get_noise_phase(timestep_index)
        match phase:
            case TemporalPhase.HIGH_NOISE:
                return settings.LORA_WEIGHT_HIGH_NOISE    # 0.6
            case TemporalPhase.MID_NOISE:
                return settings.LORA_WEIGHT_MID_NOISE     # 0.5
            case TemporalPhase.LOW_NOISE:
                return settings.LORA_WEIGHT_LOW_NOISE     # 0.4

    def get_noise_phase(self, timestep_index: int) -> TemporalPhase:
        """Classify timestep into HIGH_NOISE, MID_NOISE, or LOW_NOISE phase."""
        total = self.get_steps()
        fraction = timestep_index / max(total - 1, 1)

        if fraction > self._high_threshold:
            return TemporalPhase.HIGH_NOISE
        elif fraction > self._mid_threshold:
            return TemporalPhase.MID_NOISE
        else:
            return TemporalPhase.LOW_NOISE

    def build_lora_schedule(self) -> dict:
        """Build the complete lora_schedule dict for SVIGenerationRecord logging."""
        return {
            "high_noise_weight": settings.LORA_WEIGHT_HIGH_NOISE,
            "mid_noise_weight": settings.LORA_WEIGHT_MID_NOISE,
            "low_noise_weight": settings.LORA_WEIGHT_LOW_NOISE,
        }

    @staticmethod
    def _validate_cfg(cfg: float) -> float:
        """Assert CFG in [5.0, 6.0]. Raise SVICFGViolationError otherwise."""
        if not (settings.SVI_CFG_MIN <= cfg <= settings.SVI_CFG_MAX):
            raise SVICFGViolationError(cfg_value=cfg)
        return cfg
