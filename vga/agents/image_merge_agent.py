"""
ImageMergeAgent — Stage S-06B: merges best angle variants into consensus image.
Uses Consistance_Edit_LoRA at weight 0.6 (composition binding). CLIP ≥ 0.93.
Spec: VGA Image Pipeline Agents Spec v17.2 §S-06B
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


class ImageMergeAgent(BaseAgent):
    """S-06B: Merges best angle variants into a single consensus identity image."""

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
        """Merge the best angle variant into a consensus reference image.

        Args:
            input_data: dict with angle_variants, best_variant_index, identity_design
            context:    must have CompositionPlan and identity_state embedding

        Returns:
            (dict with merged_image + clip_score, new_context)
        """
        self._log_start(context.scene_id)
        context.assert_composition_plan()

        variants: List = input_data["angle_variants"]
        best_idx: int = input_data.get("best_variant_index", 0)
        design = input_data.get("identity_design", {})
        char_desc = design.get("character_identity", "") if isinstance(design, dict) else getattr(design, "character_identity", "")
        reference = context.identity_state.embedding_vector
        plan = context.composition_plan

        best_variant = variants[best_idx]

        # Merge: use best variant as reference, regenerate with composition binding
        merged = self._flux.generate_with_consistency_lora(
            reference_image=best_variant,
            prompt=f"{char_desc}, consistent identity, composition binding, cinematic",
            composition_plan=plan,
            lora_weight=0.6,    # higher weight for composition binding (S-06B)
        )
        score = self._clip.score(merged, reference)
        self._clip.assert_above_threshold(score, self.stage_id, context.scene_id)
        logger.info("ImageMergeAgent: merged CLIP=%.4f", score)

        output = {
            "merged_image": merged,
            "clip_score": score,
            "scene_id": context.scene_id,
            "schema_version": settings.SCHEMA_VERSION,
        }
        new_context = context.evolve(current_stage=self.stage_id)
        self._log_complete(context.scene_id)
        return output, new_context
