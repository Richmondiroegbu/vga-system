"""
SceneAgent — Stage S-02: generates scene/segment plans from script.
Initializes ImmutableContext with 5-dimensional state. HRG-2 follows.
Spec: VGA Narrative Agents Spec v17.2 §S-02; FR-950
"""
from __future__ import annotations

import logging
from typing import List, Tuple

from vga.agents.base_agent import BaseAgent
from vga.config.settings import settings
from vga.models.schemas import ScenePlanSchema, ScriptSchema, SegmentPlanSchema
from vga.state.context_factory import ContextFactory
from vga.state.immutable_context import ImmutableContext, TemporalState

logger = logging.getLogger(__name__)


class SceneAgent(BaseAgent):
    """S-02: generates ScenePlanSchema with SegmentPlans from ScriptSchema.

    Also initializes the 5-dimensional ImmutableContext for the scene.
    """

    stage_id = "S-02"

    def run(
        self,
        input_data: ScriptSchema,
        context: ImmutableContext,
    ) -> Tuple[List[ScenePlanSchema], ImmutableContext]:
        """Parse script into scene and segment plans.

        Args:
            input_data: ScriptSchema from S-01
            context:    existing ImmutableContext (or create from ContextFactory)

        Returns:
            (list of ScenePlanSchema, new_context with temporal_state initialized)
        """
        self._log_start(context.scene_id)

        scene_plans = []
        for scene_desc in input_data.scenes:
            # Distribute scene duration across 3–5 segments of 3–5s each
            scene_duration = min(
                max(scene_desc.duration_hint_s or 20.0,
                    settings.SCENE_DURATION_MIN_S),
                settings.SCENE_DURATION_MAX_S,
            )
            n_segments = max(2, int(scene_duration / settings.SEGMENT_DURATION_MAX_S))
            seg_duration = scene_duration / n_segments

            segments = [
                SegmentPlanSchema(
                    segment_id=f"{scene_desc.scene_id}_seg{i+1}",
                    scene_id=scene_desc.scene_id,
                    segment_number=i + 1,
                    duration_s=round(
                        max(settings.SEGMENT_DURATION_MIN_S,
                            min(seg_duration, settings.SEGMENT_DURATION_MAX_S)),
                        2,
                    ),
                    action_description=scene_desc.description,
                    camera_instruction="eye level, static",
                )
                for i in range(n_segments)
            ]

            plan = ScenePlanSchema(
                job_id=input_data.job_id,
                scene_id=scene_desc.scene_id,
                scene_number=scene_desc.scene_number,
                duration_s=round(scene_duration, 2),
                segments=segments,
                setting=scene_desc.description[:100],
                characters_present=[c.character_id for c in input_data.characters],
                emotional_beat=scene_desc.emotional_tone,
            )
            scene_plans.append(plan)

        total_segments = sum(len(p.segments) for p in scene_plans)
        new_context = context.evolve(
            current_stage=self.stage_id,
            temporal_state=TemporalState(
                segment_index=0,
                total_segments=total_segments,
                buffer_initialized=False,
            ),
        )

        self._log_complete(context.scene_id)
        return scene_plans, new_context
