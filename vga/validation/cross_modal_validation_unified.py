"""
CrossModalValidationUnified — unified Video ↔ Audio ↔ Identity ↔ Temporal contract.
Called per-segment; returns CrossModalValidationContract (does NOT raise on failure).
CGRL-104, v17.2.
Spec: VGA File Responsibility Spec v17.2 §11.7
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class CrossModalValidationContract:
    """Result of unified cross-modal validation for one segment."""

    segment_id: str
    scene_id: str
    video_audio_aligned: bool
    identity_consistent: bool
    temporal_consistent: bool
    overall_passed: bool
    clip_score: float = 0.0
    continuity_score: float = 0.0
    alignment_error_s: float = 0.0
    identity_delta: float = 0.0
    failures: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    schema_version: str = "v6.0"


class CrossModalValidationUnified:
    """Unified cross-modal validation for a single video segment.

    Runs 4 checks simultaneously and returns a contract — does NOT raise.
    The decision to halt or continue is made by the caller (TemporalEngine).

    Replaces individual fragmented checks (check_duration, check_phoneme, check_identity).
    CGRL-104: Use this class everywhere; deprecated individual checks are FORBIDDEN.
    """

    def validate(
        self,
        segment_id: str,
        scene_id: str,
        clip_score: float,
        continuity_score: float,
        alignment_error_s: float,
        identity_delta: float,
        clip_threshold: float = 0.93,
        continuity_threshold: float = 0.85,
        alignment_tolerance_s: float = 0.10,
        identity_delta_threshold: float = 0.03,
    ) -> CrossModalValidationContract:
        """Run all 4 cross-modal checks and return a contract.

        Args:
            segment_id:           ID of the video segment being validated
            scene_id:             scene ID
            clip_score:           CLIP identity similarity score [0,1]
            continuity_score:     temporal continuity score [0,1]
            alignment_error_s:    video-audio duration alignment error in seconds
            identity_delta:       per-segment identity drift value
            clip_threshold:       minimum CLIP score (default 0.93, RULE-92)
            continuity_threshold: minimum continuity (default 0.85)
            alignment_tolerance_s: max alignment error (default ±0.10s, FR-972)
            identity_delta_threshold: max identity delta (default 0.03, RULE-97)

        Returns:
            CrossModalValidationContract — never raises
        """
        failures: list[str] = []
        warnings: list[str] = []

        # Check 1: Identity consistency (CLIP score)
        identity_ok = clip_score >= clip_threshold
        if not identity_ok:
            failures.append(
                f"CLIP score {clip_score:.4f} < threshold {clip_threshold:.4f} "
                f"(segment {segment_id})"
            )

        # Check 2: Temporal continuity
        temporal_ok = continuity_score >= continuity_threshold
        if not temporal_ok:
            failures.append(
                f"continuity_score {continuity_score:.4f} < threshold {continuity_threshold:.4f} "
                f"(segment {segment_id})"
            )

        # Check 3: Video-audio alignment (FR-972)
        alignment_ok = abs(alignment_error_s) <= alignment_tolerance_s
        if not alignment_ok:
            failures.append(
                f"alignment_error {alignment_error_s:.3f}s exceeds ±{alignment_tolerance_s:.2f}s "
                f"(segment {segment_id}, FR-972)"
            )

        # Check 4: Identity delta (RULE-97)
        delta_ok = identity_delta <= identity_delta_threshold
        if not delta_ok:
            failures.append(
                f"identity_delta {identity_delta:.4f} > threshold {identity_delta_threshold:.4f} "
                f"(segment {segment_id}, RULE-97)"
            )
        elif identity_delta > identity_delta_threshold * 0.8:
            warnings.append(
                f"identity_delta {identity_delta:.4f} approaching threshold — monitor"
            )

        overall_passed = not failures
        if failures:
            logger.warning(
                "CrossModalValidationUnified: segment=%s FAILED — %s", segment_id, failures
            )
        else:
            logger.debug(
                "CrossModalValidationUnified: segment=%s PASSED clip=%.4f cont=%.4f",
                segment_id, clip_score, continuity_score,
            )

        return CrossModalValidationContract(
            segment_id=segment_id,
            scene_id=scene_id,
            video_audio_aligned=alignment_ok,
            identity_consistent=identity_ok and delta_ok,
            temporal_consistent=temporal_ok,
            overall_passed=overall_passed,
            clip_score=clip_score,
            continuity_score=continuity_score,
            alignment_error_s=alignment_error_s,
            identity_delta=identity_delta,
            failures=failures,
            warnings=warnings,
        )
