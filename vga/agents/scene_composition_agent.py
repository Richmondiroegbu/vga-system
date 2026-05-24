"""
SceneCompositionAgent — Stage S-04 (NEW v17.0): generates CompositionPlan.
Translates scene narrative into full 6-field CompositionPlanSchema.
CompositionPlan is REQUIRED for all image/video generation (RULE-88).
HRG-4 follows. Spec: VGA Narrative Agents Spec v17.2 §S-04; RULE-88, FR-980–FR-982
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Tuple

from vga.agents.base_agent import BaseAgent
from vga.config.settings import settings
from vga.core.exceptions import CompositionPlanValidationError
from vga.models.schemas import CompositionPlanSchema, ScenePlanSchema
from vga.models.wrappers.qwen_wrapper import QwenWrapper
from vga.state.immutable_context import CameraState, ImmutableContext, LightingState

logger = logging.getLogger(__name__)

_COMPOSITION_SYSTEM_PROMPT = """You are a professional cinematographer AI for the VGA cinematic system.
Your task is to create a detailed CompositionPlan for a video scene.
Return ONLY valid JSON with ALL 6 required fields:
- camera_angle: one of [extreme close-up, close-up, medium close-up, medium shot, medium wide shot, wide shot, extreme wide shot, overhead, low angle, high angle, dutch angle, eye level]
- camera_motion: description like "slow dolly forward", "static", "pan left", "crane up"
- character_positions: list of objects with character_id, position (center/left/right), facing (camera/away/left/right)
- focus_subject: who/what the camera focuses on (e.g., "main_character")
- lighting_style: description like "low-key dramatic", "soft natural", "golden hour", "rim lighting"
- motion_vector: camera/scene motion like "forward_slow", "stationary", "right_medium", "up_slow"
"""


class SceneCompositionAgent(BaseAgent):
    """S-04 (NEW v17.0): generates CompositionPlan from scene narrative.

    All 6 CompositionPlan fields are mandatory (RULE-88).
    Retries up to COMPOSITION_MAX_RETRIES on validation failure.
    Writes composition_plan_{scene_id}.json to disk.
    """

    stage_id = "S-04"

    def __init__(self, qwen: QwenWrapper | None = None) -> None:
        self._qwen = qwen or QwenWrapper()

    def run(
        self,
        input_data: dict,
        context: ImmutableContext,
    ) -> Tuple[CompositionPlanSchema, ImmutableContext]:
        """Generate CompositionPlan for the scene.

        Args:
            input_data: dict with scene_plan (ScenePlanSchema) and identity_design
            context:    current ImmutableContext

        Returns:
            (CompositionPlanSchema, new_context with composition_plan + camera + lighting)
        """
        self._log_start(context.scene_id)

        scene_plan: ScenePlanSchema = input_data["scene_plan"]
        identity_design = input_data.get("identity_design", {})
        scene_id = context.scene_id

        prompt = self._build_prompt(scene_plan, identity_design)

        plan: CompositionPlanSchema | None = None
        last_error: Exception | None = None

        for attempt in range(settings.COMPOSITION_MAX_RETRIES):
            try:
                raw = self._qwen.generate_structured(
                    prompt=prompt,
                    output_schema=CompositionPlanSchema,
                    system_prompt=_COMPOSITION_SYSTEM_PROMPT,
                )
                plan = raw
                break
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "SceneCompositionAgent attempt %d/%d failed: %s",
                    attempt + 1, settings.COMPOSITION_MAX_RETRIES, exc,
                )

        if plan is None:
            raise CompositionPlanValidationError(
                f"SceneCompositionAgent failed after {settings.COMPOSITION_MAX_RETRIES} "
                f"attempts for scene {scene_id}: {last_error}"
            )

        # Write to disk (audit trail)
        self._write_plan(plan, scene_id, context.job_id)

        # Evolve context with composition_plan, camera_state, lighting_state
        new_context = context.evolve(
            current_stage=self.stage_id,
            composition_plan=plan,
            camera_state=CameraState(
                angle=plan.camera_angle,
                motion=plan.camera_motion,
            ),
            lighting_state=LightingState(style=plan.lighting_style),
        )

        self._log_complete(scene_id)
        return plan, new_context

    @staticmethod
    def _build_prompt(plan: ScenePlanSchema, identity_design) -> str:
        if hasattr(identity_design, "character_identity"):
            char_desc = identity_design.character_identity
        elif isinstance(identity_design, dict):
            char_desc = identity_design.get("character_identity", "main character")
        else:
            char_desc = "main character"
        return (
            f"Create a CompositionPlan for this scene:\n"
            f"Scene ID: {plan.scene_id}\n"
            f"Setting: {plan.setting}\n"
            f"Emotional beat: {plan.emotional_beat}\n"
            f"Number of segments: {len(plan.segments)}\n"
            f"Main character: {char_desc}\n"
            f"Characters present: {', '.join(plan.characters_present)}\n\n"
            f"scene_id must be: {plan.scene_id}\n"
            f"schema_version must be: v6.0"
        )

    @staticmethod
    def _write_plan(plan: CompositionPlanSchema, scene_id: str, job_id: str) -> None:
        """Persist CompositionPlan to disk for audit and HRG-4 display."""
        try:
            output_dir = settings.HRG_DIR / job_id
            output_dir.mkdir(parents=True, exist_ok=True)
            out_path = output_dir / f"composition_plan_{scene_id}.json"
            out_path.write_text(plan.model_dump_json(indent=2), encoding="utf-8")
            logger.info("SceneCompositionAgent: plan written to %s", out_path)
        except Exception as exc:
            logger.warning("SceneCompositionAgent: failed to write plan: %s", exc)
