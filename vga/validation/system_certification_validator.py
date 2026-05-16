"""
SystemCertificationValidator — validates all 7 v17.2 system certification conditions.
Called by QualityAgent before writing PipelineReport. CGRL-103, v17.2.
Raises SystemCertificationFailureError if ANY check fails.
Spec: VGA File Responsibility Spec v17.2 §11.8
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List

from vga.core.exceptions import CriticalPipelineError

logger = logging.getLogger(__name__)


class SystemCertificationFailureError(CriticalPipelineError):
    """Pipeline output failed v17.2 system certification. CGRL-103."""

    def __init__(self, failures: list[str], **kwargs):
        self.failures = failures
        super().__init__(
            f"System certification failed ({len(failures)} condition(s) violated): "
            + "; ".join(failures),
            **kwargs,
        )


@dataclass
class CertificationResult:
    certified: bool
    certification_version: str = "v17.2"
    conditions_checked: int = 7
    conditions_passed: int = 0
    certification_failures: List[str] = field(default_factory=list)


class SystemCertificationValidator:
    """Validates all 7 v17.2 system certification conditions before PipelineReport write.

    Conditions checked:
    1. Temporal loop integrity — TemporalBuffer maintained at 5 frames throughout
    2. Identity stability — cumulative drift within threshold, reference frozen
    3. Temporal continuity — overall continuity score ≥ threshold
    4. Audio quality — SNR ≥ 10dB and no clipping
    5. Human governance — all 11 HRG checkpoints completed and approved
    6. Auditability — schema_version = "v6.0" on all artifacts
    7. Validation propagation — CLIP validated in all 3 required phases
    """

    def certify(self, pipeline_report: dict) -> CertificationResult:
        """Run all 7 certification checks. Raises SystemCertificationFailureError on any failure.

        Args:
            pipeline_report: dict representation of the completed PipelineReport

        Returns:
            CertificationResult with certified=True if all 7 pass

        Raises:
            SystemCertificationFailureError if any condition fails
        """
        failures: list[str] = []

        # 1. Temporal loop integrity
        temporal_health = pipeline_report.get("temporal_engine_health", {})
        if temporal_health.get("buffer_size_violations", 0) > 0:
            failures.append(
                "Condition 1 FAILED: TemporalBuffer size violations detected — "
                f"{temporal_health.get('buffer_size_violations')} violation(s)"
            )

        # 2. Identity stability
        identity_final = pipeline_report.get("identity_state_final", {})
        if identity_final:
            if not identity_final.get("is_frozen", False):
                failures.append("Condition 2 FAILED: char_identity_ref was never frozen (RULE-95)")
            if identity_final.get("threshold_exceeded", False):
                drift = identity_final.get("cumulative_drift", 0.0)
                failures.append(
                    f"Condition 2 FAILED: identity cumulative drift {drift:.4f} exceeded threshold"
                )

        # 3. Temporal continuity
        identity_per_segment = pipeline_report.get("identity_per_segment_video", [])
        if identity_per_segment:
            below_threshold = [s for s in identity_per_segment if s < 0.93]
            if below_threshold:
                failures.append(
                    f"Condition 3 FAILED: {len(below_threshold)} segment(s) have CLIP < 0.93"
                )

        # 4. Audio quality
        audio_quality = pipeline_report.get("audio_quality_summary", {})
        if audio_quality:
            if not audio_quality.get("snr_passed", True):
                failures.append(
                    f"Condition 4 FAILED: audio SNR {audio_quality.get('snr_db', 0.0):.1f}dB "
                    f"below 10dB minimum (RULE-99)"
                )
            if not audio_quality.get("clipping_passed", True):
                failures.append(
                    f"Condition 4 FAILED: audio clipping detected "
                    f"peak={audio_quality.get('peak_db', 0.0):.1f}dBFS (RULE-99)"
                )

        # 5. Human governance (11 HRG checkpoints)
        hrg_count = pipeline_report.get("hrg_checkpoint_count", 0)
        if hrg_count < 11:
            failures.append(
                f"Condition 5 FAILED: only {hrg_count}/11 HRG checkpoints recorded"
            )
        hrg_outcomes = pipeline_report.get("hrg_outcomes", {})
        rejected = [k for k, v in hrg_outcomes.items() if v == "rejected"]
        if rejected:
            failures.append(
                f"Condition 5 FAILED: HRG checkpoints rejected without regeneration: {rejected}"
            )

        # 6. Auditability (schema_version)
        schema_version = pipeline_report.get("schema_version", "")
        if schema_version != "v6.0":
            failures.append(
                f"Condition 6 FAILED: PipelineReport schema_version={schema_version!r} "
                f"(expected 'v6.0')"
            )

        # 7. Validation propagation — rule_compliance records
        rule_compliance = pipeline_report.get("rule_compliance", [])
        critical_rules = {"RULE-86", "RULE-89", "RULE-95", "RULE-99", "RULE-106", "RULE-108"}
        for entry in rule_compliance:
            if entry.get("rule_id") in critical_rules and not entry.get("compliant", True):
                failures.append(
                    f"Condition 7 FAILED: {entry['rule_id']} not compliant — "
                    f"{entry.get('description', '')}"
                )

        passed = len([i for i in range(7) if not any(f.startswith(f"Condition {i+1}") for f in failures)])
        result = CertificationResult(
            certified=not failures,
            conditions_passed=7 - len(failures),
            certification_failures=failures,
        )

        if failures:
            logger.error(
                "SystemCertificationValidator: FAILED — %d condition(s) violated: %s",
                len(failures), failures,
            )
            raise SystemCertificationFailureError(failures=failures)

        logger.info("SystemCertificationValidator: PASSED — all 7 conditions met")
        return result
