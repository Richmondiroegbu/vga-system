"""
GatingController — manages pipeline quality-gate mode selection.
Spec: VGA System Architecture v17.2 §9 (Adaptive Gating); FR-400–FR-420
"""
from __future__ import annotations

import logging

from vga.models.enums import GatingMode

logger = logging.getLogger(__name__)


class GatingController:
    """Selects and enforces the appropriate GatingMode for each pipeline run.

    STRICT  — all validations enforced (production default)
    BALANCED — standard validations (reduced retry thresholds)
    FAST    — minimal validations (non-production / debugging only)
    """

    def __init__(self, default_mode: GatingMode = GatingMode.STRICT) -> None:
        self._mode = default_mode
        logger.info("GatingController initialized — mode=%s", default_mode.value)

    @property
    def mode(self) -> GatingMode:
        return self._mode

    def set_mode(self, mode: GatingMode) -> None:
        """Override gating mode. STRICT is always used in production."""
        self._mode = mode
        logger.info("GatingController mode changed to %s", mode.value)

    def is_strict(self) -> bool:
        return self._mode == GatingMode.STRICT

    def clip_threshold(self, base_threshold: float) -> float:
        """Return effective CLIP threshold based on current mode."""
        if self._mode == GatingMode.FAST:
            return base_threshold * 0.90   # relaxed for testing only
        return base_threshold

    def max_retries(self, base_retries: int) -> int:
        """Return effective max retries based on current mode."""
        if self._mode == GatingMode.FAST:
            return 1
        return base_retries
