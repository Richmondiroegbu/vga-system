"""
TemporalOrchestrator — coordinates Phase 3 (S-08, S-09, S-10) execution.
Wires VideoSegmentGenerator → TemporalEngine → ContinuityValidationAgent.
Spec: VGA Codebase Structure Design v17.2 §temporal/temporal_orchestrator.py
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple

from vga.config.settings import settings
from vga.state.immutable_context import ImmutableContext

logger = logging.getLogger(__name__)


class TemporalOrchestrator:
    """Orchestrates the full temporal phase (S-08 → S-09 → S-10) for a scene."""

    def __init__(
        self,
        video_segment_generator: "VideoSegmentGenerator",
        temporal_engine: "TemporalEngine",
        continuity_agent: "ContinuityValidationAgent",
        orchestrator: "MasterOrchestrator",
    ) -> None:
        self._s08 = video_segment_generator
        self._s09 = temporal_engine
        self._s10 = continuity_agent
        self._orchestrator = orchestrator

    def run_scene(
        self,
        refined_image: "PIL.Image.Image",
        scene_plan: "ScenePlanSchema",
        segment_plans: list,
        context: ImmutableContext,
        output_dir: Path,
    ) -> Tuple[list, "ContinuityReport", ImmutableContext]:
        """Execute the complete temporal pipeline for one scene.

        Args:
            refined_image:  from S-07
            scene_plan:     scene metadata
            segment_plans:  list of SegmentPlanSchema for all segments
            context:        must have frozen identity + CompositionPlan
            output_dir:     where to write video segments

        Returns:
            (all_segments, continuity_report, updated_context)
        """
        context.assert_composition_plan()
        context.assert_identity_frozen()

        logger.info("TemporalOrchestrator: starting Phase 3 for scene %s", context.scene_id)

        # S-08: Generate Segment_1 via Wan2.2
        input_s08 = {
            "refined_image": refined_image,
            "output_dir": str(output_dir),
            "prompt": self._build_segment_prompt(scene_plan, segment_plans[0] if segment_plans else None, context),
        }
        output_s08, context = self._orchestrator.execute_stage(self._s08, input_s08, context)
        segment_1 = output_s08["segment_1"]
        buffer = output_s08["buffer"]

        # S-09: Generate Segments 2..N via TemporalEngine (autoregressive)
        all_segments = [segment_1]
        if len(segment_plans) > 1:
            remaining_plans = segment_plans[1:]
            segments_2_n, context = self._s09.generate_segments(
                segment_1=segment_1,
                segment_plans=remaining_plans,
                context=context,
                output_dir=output_dir,
            )
            all_segments.extend(segments_2_n)

        # S-10: Continuity validation
        input_s10 = {"all_segments": all_segments}
        continuity_report, context = self._orchestrator.execute_stage(
            self._s10, input_s10, context
        )

        logger.info(
            "TemporalOrchestrator: Phase 3 complete — %d segments, continuity=%.4f",
            len(all_segments), continuity_report.overall_continuity_score,
        )
        return all_segments, continuity_report, context

    @staticmethod
    def _build_segment_prompt(scene_plan: object, segment: object, context: ImmutableContext) -> str:
        """Build generation prompt for Segment_1."""
        base = getattr(scene_plan, "setting", "cinematic scene")
        action = getattr(segment, "action_description", "character in action") if segment else "character in action"
        plan = context.composition_plan
        if plan:
            return f"{action}, {plan.camera_motion}, {plan.lighting_style} lighting, photorealistic, cinematic"
        return f"{action}, {base}, cinematic quality"
