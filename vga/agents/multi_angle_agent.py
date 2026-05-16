"""
MultiAngleAgent — Stage S-06A: generates multi-angle identity reinforcement variants.
5-8 angle variants with Consistance_Edit_LoRA (weight 0.5–0.6). CLIP ≥ 0.93 each.
Spec: VGA Image Pipeline Agents Spec v17.2 §S-06A
"""
from __future__ import annotations

import logging
from typing import List, Tuple

from vga.agents.base_agent import BaseAgent
from vga.config.settings import settings
from vga.models.schemas import CompositionPlanSchema
from vga.models.wrappers.flux_wrapper import FluxWrapper
from vga.state.immutable_context import ImmutableContext
from vga.validation.clip_validator import CLIPValidator

logger = logging.getLogger(__name__)

_ANGLES = ["close-up", "medium shot", "medium wide shot", "wide shot", "low angle", "high angle"]


class MultiAngleAgent(BaseAgent):
    """S-06A: Multi-angle identity reinforcement — 5-8 variants across different camera angles."""

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
        """Generate multi-angle variants of the best base image.

        Args:
            input_data: dict with best_image, identity_design, num_angles (default 5)
            context:    must contain CompositionPlan and identity_state with embedding

        Returns:
            (dict with angle_variants + clip_scores, new_context)
        """
        self._log_start(context.scene_id)
        context.assert_composition_plan()

        best_image = input_data["best_image"]
        design = input_data.get("identity_design", {})
        char_desc = design.get("character_identity", "") if isinstance(design, dict) else getattr(design, "character_identity", "")
        reference = context.identity_state.embedding_vector
        plan = context.composition_plan
        num_angles = input_data.get("num_angles", 5)
        angles = _ANGLES[:num_angles]

        variants = []
        clip_scores = []

        for angle in angles:
            modified = plan.model_copy(update={"camera_angle": angle})
            img = self._flux.generate_with_consistency_lora(
                reference_image=best_image,
                prompt=char_desc,
                composition_plan=modified,
                lora_weight=0.55,
            )
            score = self._clip.score(img, reference)
            self._clip.assert_above_threshold(score, self.stage_id, context.scene_id)
            variants.append(img)
            clip_scores.append(score)
            logger.info("MultiAngleAgent: angle=%s CLIP=%.4f", angle, score)

        output = {
            "angle_variants": variants,
            "clip_scores": clip_scores,
            "best_variant_index": clip_scores.index(max(clip_scores)),
            "scene_id": context.scene_id,
            "schema_version": settings.SCHEMA_VERSION,
        }
        new_context = context.evolve(current_stage=self.stage_id)
        self._log_complete(context.scene_id)
        return output, new_context
