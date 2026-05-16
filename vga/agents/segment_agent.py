"""
SegmentAgent — Stage S-02b: generates SegmentPlanSchema from ScenePlanSchema.
Separate from SceneAgent per file responsibility spec (one file = one responsibility).
Spec: VGA Narrative Agents Spec v17.2 §S-02; VGA File Responsibility Spec §SegmentAgent
"""
from __future__ import annotations

import logging
from typing import List, Tuple

from vga.agents.base_agent import BaseAgent
from vga.config.settings import settings
from vga.models.schemas import ScenePlanSchema, SegmentPlanSchema
from vga.state.immutable_context import ImmutableContext

logger = logging.getLogger(__name__)


class SegmentAgent(BaseAgent):
    """S-02b: Generates per-segment plans from a validated ScenePlanSchema.

    Creates segment-level production plans covering camera instructions,
    dialogue assignments, and timing for each 3–5s segment within a scene.
    """

    stage_id = "S-02"

    def run(
        self,
        input_data: dict,
        context: ImmutableContext,
    ) -> Tuple[List[SegmentPlanSchema], ImmutableContext]:
        """Generate segment plans from a scene plan.

        Args:
            input_data: dict with scene_plan (ScenePlanSchema)
            context:    current ImmutableContext

        Returns:
            (list of SegmentPlanSchema, new_context)
        """
        self._log_start(context.scene_id)

        scene_plan: ScenePlanSchema = input_data["scene_plan"]
        segments = scene_plan.segments

        # Enrich each segment with camera instructions if not already set
        enriched = []
        for i, seg in enumerate(segments):
            if not seg.camera_instruction or seg.camera_instruction == "eye level, static":
                # Default progressive camera variation across segments
                instructions = [
                    "eye level, static",
                    "low angle, slow push-in",
                    "medium shot, rack focus",
                    "close-up, steady",
                    "wide shot, slow pullback",
                ]
                camera = instructions[i % len(instructions)]
            else:
                camera = seg.camera_instruction

            enriched.append(
                SegmentPlanSchema(
                    segment_id=seg.segment_id,
                    scene_id=seg.scene_id,
                    segment_number=seg.segment_number,
                    duration_s=seg.duration_s,
                    dialogue=seg.dialogue,
                    action_description=seg.action_description,
                    camera_instruction=camera,
                )
            )

        logger.info(
            "SegmentAgent: generated %d segment plans for scene %s",
            len(enriched), context.scene_id,
        )
        new_context = context.evolve(current_stage=self.stage_id)
        self._log_complete(context.scene_id)
        return enriched, new_context
