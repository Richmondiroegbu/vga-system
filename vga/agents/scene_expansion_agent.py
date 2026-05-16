"""
SceneExpansionAgent — Stage S-06C: expands identity into full scene with environment.
Fully binds CompositionPlan (all 6 fields) with LoRA weight 0.5–0.6. CLIP ≥ 0.93.
Spec: VGA Image Pipeline Agents Spec v17.2 §S-06C
"""
from __future__ import annotations

import logging
from typing import Tuple

from vga.agents.base_agent import BaseAgent
from vga.config.settings import settings
from vga.models.wrappers.flux_wrapper import FluxWrapper
from vga.state.immutable_context import ImmutableContext
from vga.validation.clip_validator import CLIPValidator

logger = logging.getLogger(__name__)


class SceneExpansionAgent(BaseAgent):
    """S-06C: Expands merged identity image into full scene with environment binding.

    The CompositionPlan is FULLY BOUND at this stage — all 6 fields must be present
    and are used to constrain the generation. RULE-88 enforced.
    """

    stage_id = "S-06"

    def __init__(
        self,
        flux: FluxWrapper | None = None,
        clip_validator: CLIPValidator | None = None,
    ) -> None:
        self._flux = flux or FluxWrapper()
        self._clip = clip_validator or CLIPValidator()

    def run(
        self,
        input_data: dict,
        context: ImmutableContext,
    ) -> Tuple[dict, ImmutableContext]:
        """Expand merged identity image into a fully environment-bound scene.

        Args:
            input_data: dict with merged_image, identity_design
            context:    must have CompositionPlan (FULLY BOUND — RULE-88) + identity_state

        Returns:
            (dict with scene_expanded_image + clip_score, new_context)
        """
        self._log_start(context.scene_id)
        context.assert_composition_plan()   # RULE-88 — all 6 fields required

        merged_image = input_data["merged_image"]
        design = input_data.get("identity_design", {})
        char_desc = design.get("character_identity", "") if isinstance(design, dict) else getattr(design, "character_identity", "")
        env_desc = design.get("environment_description", "") if isinstance(design, dict) else getattr(design, "environment_description", "")
        reference = context.identity_state.embedding_vector
        plan = context.composition_plan

        # Full scene prompt: character + environment + all CompositionPlan fields bound
        scene_prompt = (
            f"{char_desc}, "
            f"{env_desc}, "
            f"{plan.lighting_style} lighting, "
            f"{plan.camera_angle} shot, "
            f"focus on {plan.focus_subject}, "
            f"cinematic quality, photorealistic, full body in environment"
        )

        expanded = self._flux.generate_with_consistency_lora(
            reference_image=merged_image,
            prompt=scene_prompt,
            composition_plan=plan,
            lora_weight=0.55,    # balanced: identity preservation + environment binding
        )
        score = self._clip.score(expanded, reference)
        self._clip.assert_above_threshold(score, self.stage_id, context.scene_id)
        logger.info("SceneExpansionAgent: expanded CLIP=%.4f", score)

        output = {
            "scene_expanded_image": expanded,
            "clip_score": score,
            "scene_id": context.scene_id,
            "schema_version": settings.SCHEMA_VERSION,
        }
        new_context = context.evolve(current_stage=self.stage_id)
        self._log_complete(context.scene_id)
        return output, new_context
