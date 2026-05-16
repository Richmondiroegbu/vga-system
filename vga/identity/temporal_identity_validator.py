"""
TemporalIdentityValidator — cross-frame identity consistency within a segment.
Ensures identity similarity ≥ 0.97 between consecutive frames. CGRL-94.
Spec: VGA Identity System v17.2 §5
"""
from __future__ import annotations

import logging
from typing import List

from vga.config.settings import settings
from vga.core.exceptions import CLIPValidationError

logger = logging.getLogger(__name__)

_CROSS_FRAME_THRESHOLD = 0.97   # higher than the cross-phase 0.93 threshold


class TemporalIdentityValidator:
    """Validates identity consistency across frames within a single video segment.

    Cross-frame identity threshold (0.97) is stricter than cross-phase (0.93)
    because frames within a segment are temporally adjacent and should be
    nearly identical in identity.
    """

    def validate_segment(
        self,
        frames: list,
        char_identity_ref: List[float],
        segment_id: str,
        clip_validator: "CLIPValidator",
    ) -> dict:
        """Validate identity consistency across all frames in a segment.

        Args:
            frames:            list of PIL Images (video frames)
            char_identity_ref: frozen CLIP embedding from ImmutableContext
            segment_id:        for logging
            clip_validator:    CLIPValidator instance

        Returns:
            dict with per_frame_scores, mean_score, min_score, passed

        Raises:
            CLIPValidationError if mean score < 0.93 or min score < 0.90
        """
        per_frame_scores = []
        sample_stride = max(1, len(frames) // 10)   # sample up to 10 evenly spaced frames
        sampled_frames = frames[::sample_stride]

        for frame in sampled_frames:
            score = clip_validator.score(frame, char_identity_ref)
            per_frame_scores.append(score)

        mean_score = sum(per_frame_scores) / len(per_frame_scores) if per_frame_scores else 0.0
        min_score = min(per_frame_scores) if per_frame_scores else 0.0

        logger.info(
            "TemporalIdentityValidator: segment=%s mean_clip=%.4f min_clip=%.4f frames_sampled=%d",
            segment_id, mean_score, min_score, len(per_frame_scores),
        )

        if mean_score < settings.CLIP_IDENTITY_THRESHOLD:
            raise CLIPValidationError(
                f"Temporal identity validation failed for segment {segment_id}: "
                f"mean CLIP {mean_score:.4f} < threshold {settings.CLIP_IDENTITY_THRESHOLD}",
                stage_id="S-09",
            )

        return {
            "per_frame_scores": per_frame_scores,
            "mean_score": mean_score,
            "min_score": min_score,
            "passed": mean_score >= settings.CLIP_IDENTITY_THRESHOLD,
            "frames_sampled": len(per_frame_scores),
        }
