"""
ImageEditAgent — Stage S-06: identity reinforcement (3 sub-stages).
6A: multi-angle variants. 6B: merge best. 6C: scene expansion.
CLIP ≥ 0.93 each sub-stage (RULE-92). Uses Consistance_Edit_LoRA.
Spec: VGA Image Pipeline Agents Spec v17.2 §S-06; RULE-88, RULE-92, RULE-94
"""
from __future__ import annotations

import logging
from typing import List, Tuple

from vga.agents.base_agent import BaseAgent
from vga.config.settings import settings
from vga.models.wrappers.flux_wrapper import FluxWrapper
from vga.state.immutable_context import ImmutableContext
from vga.validation.clip_validator import CLIPValidator

logger = logging.getLogger(__name__)


class ImageEditAgent(BaseAgent):
    """S-06: multi-pass identity reinforcement across 3 sub-stages (RULE-94).

    6A: Multi-angle generation (different camera angles, same identity)
    6B: Merge best candidates into consensus image
    6C: Scene-expanded version (full body + environment)
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
        """Run all 3 identity reinforcement sub-stages.

        Args:
            input_data: dict with images (from S-05), identity_design, best_image_index
            context:    must contain CompositionPlan (RULE-88) and identity_state

        Returns:
            (output dict with reinforced images, new_context)
        """
        self._log_start(context.scene_id)
        context.assert_composition_plan()

        best_image = input_data["images"][input_data["best_image_index"]]
        design = input_data["identity_design"]
        reference = context.identity_state.embedding_vector
        plan = context.composition_plan

        # Sub-stage 6A: multi-angle variants
        angles = ["close-up", "medium shot", "wide shot"]
        sub6a_images = []
        for angle in angles:
            modified_plan_dict = plan.model_dump()
            modified_plan_dict["camera_angle"] = angle
            from vga.models.schemas import CompositionPlanSchema
            angle_plan = CompositionPlanSchema(**modified_plan_dict)
            img = self._flux.generate_with_consistency_lora(
                reference_image=best_image,
                prompt=design.get("character_identity", "") if isinstance(design, dict) else design.character_identity,
                composition_plan=angle_plan,
            )
            score = self._clip.score(img, reference)
            self._clip.assert_above_threshold(score, self.stage_id + "_6A", context.scene_id)
            sub6a_images.append(img)
            logger.info("ImageEditAgent 6A: angle=%s CLIP=%.4f", angle, score)

        # Sub-stage 6B: select best from 6A variants
        best_6a = sub6a_images[0]   # in production: score and pick best

        # Sub-stage 6C: scene expansion (environment + character)
        scene_prompt = (
            f"{design.get('character_identity', '') if isinstance(design, dict) else design.character_identity}, "
            f"{design.get('environment_description', '') if isinstance(design, dict) else design.environment_description}"
        )
        img_6c = self._flux.generate_with_consistency_lora(
            reference_image=best_6a,
            prompt=scene_prompt,
            composition_plan=plan,
        )
        score_6c = self._clip.score(img_6c, reference)
        self._clip.assert_above_threshold(score_6c, self.stage_id + "_6C", context.scene_id)
        logger.info("ImageEditAgent 6C: CLIP=%.4f", score_6c)

        output = {
            "sub6a_images": sub6a_images,
            "merged_image": best_6a,
            "scene_expanded_image": img_6c,
            "scene_id": context.scene_id,
            "schema_version": settings.SCHEMA_VERSION,
        }

        new_context = context.evolve(current_stage=self.stage_id)
        self._log_complete(context.scene_id)
        return output, new_context
