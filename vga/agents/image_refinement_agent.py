"""
ImageRefinementAgent — Stage S-07: Z-Image-Turbo refinement + identity freeze.
drift ≤ 0.02 (RULE-93). CLIP ≥ 0.93 (RULE-92). Freezes char_identity_ref (RULE-95).
Spec: VGA Image Pipeline Agents Spec v17.2 §S-07; RULE-92, RULE-93, RULE-95
"""
from __future__ import annotations

import logging
from typing import Tuple

from vga.agents.base_agent import BaseAgent
from vga.config.settings import settings
from vga.identity.identity_drift_controller import IdentityDriftController
from vga.identity.identity_manager import IdentityManager
from vga.models.wrappers.z_image_wrapper import ZImageWrapper
from vga.state.immutable_context import ImmutableContext
from vga.validation.clip_validator import CLIPValidator

logger = logging.getLogger(__name__)


class ImageRefinementAgent(BaseAgent):
    """S-07: refines the scene-expanded image and freezes char_identity_ref.

    RULE-93: drift ≤ 0.02 per step.
    RULE-95: char_identity_ref is frozen here — NEVER recomputed downstream.
    """

    stage_id = "S-07"

    def __init__(
        self,
        zimage: ZImageWrapper | None = None,
        clip_validator: CLIPValidator | None = None,
        identity_manager: IdentityManager | None = None,
        drift_controller: IdentityDriftController | None = None,
    ) -> None:
        self._zimage = zimage or ZImageWrapper()
        self._clip = clip_validator or CLIPValidator()
        self._identity_manager = identity_manager or IdentityManager()
        self._drift_controller = drift_controller or IdentityDriftController()

    def run(
        self,
        input_data: dict,
        context: ImmutableContext,
    ) -> Tuple[dict, ImmutableContext]:
        """Refine scene-expanded image and freeze identity reference.

        Args:
            input_data: dict with scene_expanded_image, identity_design
            context:    must have identity_state with embedding_vector (from S-05)

        Returns:
            (output dict with refined_image, new_context with FROZEN identity_state)
        """
        self._log_start(context.scene_id)

        source_image = input_data["scene_expanded_image"]
        design = input_data.get("identity_design", {})
        char_desc = (
            design.get("character_identity", "")
            if isinstance(design, dict)
            else getattr(design, "character_identity", "")
        )
        reference = context.identity_state.embedding_vector

        # Initial CLIP score before refinement
        initial_score = self._clip.score(source_image, reference)
        self._clip.assert_above_threshold(initial_score, self.stage_id, context.scene_id)

        # Refine with Z-Image-Turbo (light denoising to preserve identity)
        refined = self._zimage.refine(
            image=source_image,
            prompt=char_desc + ", photorealistic, high detail, cinematic",
            strength=settings.ZIMAGE_DENOISE_MIN,
        )

        # Validate drift (RULE-93: ≤ 0.02)
        refined_score = self._clip.score(refined, reference)
        self._drift_controller.check_drift(
            previous_clip=initial_score,
            current_clip=refined_score,
            step=1,
            scene_id=context.scene_id,
        )
        self._clip.assert_above_threshold(refined_score, self.stage_id, context.scene_id)

        logger.info(
            "ImageRefinementAgent: initial_clip=%.4f refined_clip=%.4f drift=%.4f",
            initial_score, refined_score, abs(initial_score - refined_score),
        )

        # Compute final embedding and FREEZE char_identity_ref (RULE-95)
        final_embedding = self._clip.encode_image(refined)
        new_context = self._identity_manager.freeze_identity(final_embedding, context)
        new_context = new_context.evolve(current_stage=self.stage_id)

        output = {
            "refined_image": refined,
            "initial_clip_score": initial_score,
            "refined_clip_score": refined_score,
            "drift_score": abs(initial_score - refined_score),
            "scene_id": context.scene_id,
            "schema_version": settings.SCHEMA_VERSION,
        }

        self._log_complete(context.scene_id)
        return output, new_context
