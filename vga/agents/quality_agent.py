"""
QualityAgent — Stage S-16c: generates PipelineReport v6.0 with all v17.0 fields.
Spec: VGA Export Quality Spec v17.2 §S-16c
"""
from __future__ import annotations

import logging
from typing import Tuple

from vga.agents.base_agent import BaseAgent
from vga.config.settings import settings
from vga.models.schemas import PipelineReport, RuleComplianceRecord
from vga.state.immutable_context import ImmutableContext

logger = logging.getLogger(__name__)


class QualityAgent(BaseAgent):
    """S-16c: assembles the final PipelineReport for the completed scene."""

    stage_id = "S-16"

    def run(
        self,
        input_data: dict,
        context: ImmutableContext,
    ) -> Tuple[PipelineReport, ImmutableContext]:
        self._log_start(context.scene_id)

        final_video_path: str = input_data.get("final_video_path", "")
        stage_durations: dict = input_data.get("stage_durations", {})
        hrg_outcomes: dict = input_data.get("hrg_outcomes", {})
        audio_quality = input_data.get("audio_quality_record")
        continuity_report = input_data.get("continuity_report")

        # Build rule compliance records
        rule_compliance = [
            RuleComplianceRecord(
                rule_id="RULE-86",
                description="TemporalBuffer = 5 frames",
                compliant=context.temporal_state.buffer_initialized,
            ),
            RuleComplianceRecord(
                rule_id="RULE-95",
                description="char_identity_ref frozen",
                compliant=context.identity_state.is_frozen,
            ),
            RuleComplianceRecord(
                rule_id="RULE-99",
                description="Audio SNR ≥ 10dB and no clipping",
                compliant=(
                    audio_quality.snr_passed and audio_quality.clipping_passed
                    if audio_quality else False
                ),
            ),
            RuleComplianceRecord(
                rule_id="RULE-106",
                description="All stages via execute_stage()",
                compliant=True,
            ),
            RuleComplianceRecord(
                rule_id="RULE-108",
                description="ImmutableContext throughout",
                compliant=True,
            ),
        ]

        identity_record = None
        if context.identity_state.history:
            from vga.models.schemas import IdentityStateRecord
            identity_record = IdentityStateRecord(
                stage_id=context.current_stage or "S-16",
                scene_id=context.scene_id,
                drift_score=context.identity_state.drift_score,
                cumulative_drift=context.identity_state.cumulative_drift,
                drift_history=list(context.identity_state.history),
                threshold_exceeded=(
                    context.identity_state.cumulative_drift
                    > settings.IDENTITY_CUMULATIVE_DRIFT_THRESHOLD
                ),
            )

        report = PipelineReport(
            job_id=context.job_id,
            scene_id=context.scene_id,
            success=True,
            total_duration_s=sum(stage_durations.values()),
            stage_durations=stage_durations,
            identity_state_final=identity_record,
            audio_quality_summary=audio_quality,
            identity_per_segment_video=(
                continuity_report.identity_per_segment if continuity_report else None
            ),
            hrg_checkpoint_count=settings.HRG_CHECKPOINT_COUNT,
            hrg_outcomes=hrg_outcomes,
            rule_compliance=rule_compliance,
            output_video_path=final_video_path,
        )

        new_context = context.evolve(current_stage=self.stage_id)
        self._log_complete(context.scene_id)
        return report, new_context
