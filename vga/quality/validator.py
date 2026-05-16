"""
QualityValidator — aggregates all per-stage quality signals into a single pass/fail.
Called by QualityAgent (S-16c) before writing PipelineReport.
Spec: VGA Codebase Structure Design v17.2 §quality/validator.py
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List

from vga.config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class QualityGate:
    name: str
    passed: bool
    score: float
    threshold: float
    detail: str = ""


@dataclass
class QualityReport:
    overall_passed: bool
    gates: List[QualityGate] = field(default_factory=list)
    overall_score: float = 0.0
    schema_version: str = "v6.0"


class QualityValidator:
    """Validates overall pipeline output quality by checking all quality gates."""

    def validate(self, pipeline_report: dict) -> QualityReport:
        """Run all quality gates against the pipeline report.

        Args:
            pipeline_report: dict representation of PipelineReport

        Returns:
            QualityReport with all gate results
        """
        gates: List[QualityGate] = []

        # Gate 1: Identity drift within threshold
        identity = pipeline_report.get("identity_state_final", {})
        if identity:
            drift = identity.get("cumulative_drift", 0.0)
            threshold = settings.IDENTITY_CUMULATIVE_DRIFT_THRESHOLD
            gates.append(QualityGate(
                name="identity_drift",
                passed=drift <= threshold,
                score=max(0.0, 1.0 - drift / threshold),
                threshold=threshold,
                detail=f"cumulative_drift={drift:.4f}",
            ))

        # Gate 2: Audio quality
        audio = pipeline_report.get("audio_quality_summary", {})
        if audio:
            snr_ok = audio.get("snr_passed", False)
            clip_ok = audio.get("clipping_passed", True)
            gates.append(QualityGate(
                name="audio_quality",
                passed=snr_ok and clip_ok,
                score=1.0 if (snr_ok and clip_ok) else 0.0,
                threshold=1.0,
                detail=f"snr_passed={snr_ok} clipping_passed={clip_ok}",
            ))

        # Gate 3: Temporal identity per segment
        seg_scores = pipeline_report.get("identity_per_segment_video", [])
        if seg_scores:
            mean_score = sum(seg_scores) / len(seg_scores)
            gates.append(QualityGate(
                name="temporal_identity",
                passed=mean_score >= settings.CLIP_IDENTITY_THRESHOLD,
                score=mean_score,
                threshold=settings.CLIP_IDENTITY_THRESHOLD,
                detail=f"mean_clip={mean_score:.4f} n_segments={len(seg_scores)}",
            ))

        # Gate 4: HRG completeness (11 checkpoints)
        hrg_count = pipeline_report.get("hrg_checkpoint_count", 0)
        gates.append(QualityGate(
            name="hrg_completeness",
            passed=hrg_count >= 11,
            score=min(1.0, hrg_count / 11),
            threshold=1.0,
            detail=f"hrg_count={hrg_count}/11",
        ))

        overall_passed = all(g.passed for g in gates)
        overall_score = (
            sum(g.score for g in gates) / len(gates) if gates else 0.0
        )

        if overall_passed:
            logger.info("QualityValidator: PASSED — score=%.3f", overall_score)
        else:
            failed = [g.name for g in gates if not g.passed]
            logger.error("QualityValidator: FAILED gates=%s score=%.3f", failed, overall_score)

        return QualityReport(
            overall_passed=overall_passed,
            gates=gates,
            overall_score=round(overall_score, 4),
        )
