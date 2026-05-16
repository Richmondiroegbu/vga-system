"""
TemporalRetryController — adjusts SVI parameters on segment retry.
Increases steps and adjusts CFG on each retry attempt.
Spec: VGA Temporal Engine Spec v17.2 §TemporalRetryController; FR-935
"""
from __future__ import annotations

import logging

from vga.config.settings import settings

logger = logging.getLogger(__name__)


class TemporalRetryController:
    """Adjusts SVI generation parameters after each failed segment attempt.

    On each retry:
    - Steps increase toward STEPS_CRITICAL (50)
    - CFG decreases slightly toward SVI_CFG_MIN (5.0) to allow more variation

    Retry count is bounded by settings.TEMPORAL_MAX_RETRIES_PER_SEGMENT (3).
    """

    def __init__(self) -> None:
        self._max_retries = settings.TEMPORAL_MAX_RETRIES_PER_SEGMENT
        logger.info("TemporalRetryController: max_retries=%d", self._max_retries)

    def adjust_params(
        self,
        attempt: int,
        current_cfg: float,
        current_steps: int,
    ) -> tuple[float, int]:
        """Return adjusted (cfg, steps) for the given retry attempt.

        Args:
            attempt:       current attempt number (0 = first try, 1 = first retry, ...)
            current_cfg:   current CFG value
            current_steps: current step count

        Returns:
            (adjusted_cfg, adjusted_steps) — both clamped to valid ranges
        """
        if attempt == 0:
            return current_cfg, current_steps

        # Increase steps by 10 per retry, up to STEPS_CRITICAL
        new_steps = min(current_steps + attempt * 10, settings.STEPS_CRITICAL)

        # Decrease CFG slightly per retry (more variation = more chance of passing CLIP)
        cfg_delta = attempt * 0.1
        new_cfg = max(current_cfg - cfg_delta, settings.SVI_CFG_MIN)
        new_cfg = min(new_cfg, settings.SVI_CFG_MAX)   # ensure in range

        logger.info(
            "TemporalRetryController: attempt=%d cfg %.2f→%.2f steps %d→%d",
            attempt, current_cfg, new_cfg, current_steps, new_steps,
        )
        return new_cfg, new_steps

    @property
    def max_retries(self) -> int:
        return self._max_retries
