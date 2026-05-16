"""
SegmentRoleRouter — routes segments to the correct generation model based on role.
Segment_1 → Wan2.2 (S-08). Segments 2..N → SVI (S-09 via TemporalEngine).
Spec: VGA Codebase Structure Design v17.2 §temporal/segment_role_router.py
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class SegmentRoleRouter:
    """Determines which model generates each segment based on its role in the sequence."""

    @staticmethod
    def get_model(segment_number: int) -> str:
        """Return the model key for a segment number.

        Args:
            segment_number: 1-indexed segment position

        Returns:
            "wan22" for Segment_1, "svi" for all subsequent segments
        """
        if segment_number == 1:
            return "wan22"
        return "svi"

    @staticmethod
    def get_stage_id(segment_number: int) -> str:
        """Return the pipeline stage ID for a segment number."""
        if segment_number == 1:
            return "S-08"
        return "S-09"

    @staticmethod
    def is_anchor_segment(segment_number: int) -> bool:
        """True for Segment_1 — the anchor that initializes the TemporalBuffer."""
        return segment_number == 1

    @staticmethod
    def requires_buffer_conditioning(segment_number: int) -> bool:
        """True for all segments after Segment_1 — require 5-frame buffer input."""
        return segment_number > 1
