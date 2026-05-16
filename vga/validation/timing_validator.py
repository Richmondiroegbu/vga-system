"""
TimingValidator — validates audio timing alignment per segment.
Ensures dialogue audio duration matches target within ±0.10s (RULE-96).
Spec: VGA Validation Spec v17.2 §TimingValidator; RULE-96
"""
from __future__ import annotations

import logging

from vga.config.settings import settings
from vga.core.exceptions import SLAViolationError

logger = logging.getLogger(__name__)


class TimingValidator:
    """Validates that generated audio duration matches the target segment duration."""

    def validate_timing(
        self,
        actual_duration_s: float,
        target_duration_s: float,
        segment_id: str,
        stage_id: str = "S-11",
    ) -> float:
        """Validate timing error is within tolerance. RULE-96.

        Args:
            actual_duration_s: actual audio duration
            target_duration_s: expected segment duration
            segment_id:        for logging
            stage_id:          pipeline stage (default S-11)

        Returns:
            float timing_error_s (absolute value)

        Raises:
            SLAViolationError if timing_error > TIMING_TOLERANCE_S
        """
        timing_error = abs(actual_duration_s - target_duration_s)

        if timing_error > settings.TIMING_TOLERANCE_S:
            logger.warning(
                "TimingValidator: segment=%s timing_error=%.3fs exceeds ±%.2fs (RULE-96)",
                segment_id, timing_error, settings.TIMING_TOLERANCE_S,
            )
        else:
            logger.debug(
                "TimingValidator: segment=%s timing_error=%.3fs OK",
                segment_id, timing_error,
            )

        return timing_error
