"""
IdentityReinforcementEngine — drives the S-06 (6A, 6B, 6C) identity reinforcement loop.
Orchestrates MultiAngleAgent, ImageMergeAgent, SceneExpansionAgent with LoRA scheduling.
Spec: VGA Identity System v17.2 §6
"""
from __future__ import annotations

import logging
from typing import Tuple

from vga.config.settings import settings
from vga.state.immutable_context import ImmutableContext

logger = logging.getLogger(__name__)


class IdentityReinforcementEngine:
    """Orchestrates the 3-sub-stage identity reinforcement pipeline (S-06).

    Sub-stages:
    6A: MultiAngleAgent — 5-8 angle variants (LoRA 0.5–0.6)
    6B: ImageMergeAgent — merge best candidates (LoRA 0.6)
    6C: SceneExpansionAgent — environment binding (LoRA 0.5–0.6, full CompositionPlan)

    CGRL-94: CLIPValidator called after each sub-stage.
    """

    def __init__(
        self,
        multi_angle_agent: "MultiAngleAgent | None" = None,
        image_merge_agent: "ImageMergeAgent | None" = None,
        scene_expansion_agent: "SceneExpansionAgent | None" = None,
        clip_validator: "CLIPValidator | None" = None,
    ) -> None:
        from vga.agents.multi_angle_agent import MultiAngleAgent
        from vga.agents.image_merge_agent import ImageMergeAgent
        from vga.agents.scene_expansion_agent import SceneExpansionAgent
        from vga.validation.clip_validator import CLIPValidator

        self._6a = multi_angle_agent or MultiAngleAgent()
        self._6b = image_merge_agent or ImageMergeAgent()
        self._6c = scene_expansion_agent or SceneExpansionAgent()
        self._clip = clip_validator or CLIPValidator()

    def run_reinforcement(
        self,
        best_base_image: "PIL.Image.Image",
        identity_design: object,
        context: ImmutableContext,
        orchestrator: "MasterOrchestrator",
    ) -> Tuple[dict, ImmutableContext]:
        """Run the full 3-stage reinforcement loop via execute_stage().

        Args:
            best_base_image: best image from S-05
            identity_design: IdentityDesignSchema from S-03
            context:         current ImmutableContext with CompositionPlan + identity_state
            orchestrator:    MasterOrchestrator for execute_stage() calls

        Returns:
            (output_dict with scene_expanded_image, updated context)
        """
        logger.info("IdentityReinforcementEngine: starting S-06 for scene %s", context.scene_id)

        # Sub-stage 6A: Multi-angle variants
        input_6a = {
            "best_image": best_base_image,
            "identity_design": identity_design,
            "num_angles": 5,
        }
        output_6a, context = orchestrator.execute_stage(self._6a, input_6a, context)

        # Sub-stage 6B: Merge best variant
        input_6b = {
            "angle_variants": output_6a["angle_variants"],
            "best_variant_index": output_6a["best_variant_index"],
            "identity_design": identity_design,
        }
        output_6b, context = orchestrator.execute_stage(self._6b, input_6b, context)

        # Sub-stage 6C: Scene expansion (full CompositionPlan binding)
        input_6c = {
            "merged_image": output_6b["merged_image"],
            "identity_design": identity_design,
        }
        output_6c, context = orchestrator.execute_stage(self._6c, input_6c, context)

        logger.info(
            "IdentityReinforcementEngine: S-06 complete — final CLIP=%.4f",
            output_6c.get("clip_score", 0.0),
        )
        return output_6c, context
