"""
IdentityDriftController — per-step drift enforcement for image refinement stages.
Enforces RULE-93: drift ≤ 0.02 per refinement step.
Spec: VGA Identity System v17.2 §3; RULE-93, FR-609–FR-612
"""
from __future__ import annotations

import logging

from vga.config.settings import settings
from vga.core.exceptions import CLIPValidationError

logger = logging.getLogger(__name__)


class IdentityDriftController:
    """Enforces per-step drift constraints during the image refinement loop (S-07).

    RULE-93: cumulative drift per refinement step ≤ 0.02.
    Used by ImageRefinementAgent to reject drift-violating candidates.
    """

    def __init__(self, max_drift_per_step: float | None = None) -> None:
        self._max_drift = max_drift_per_step or settings.CLIP_DRIFT_THRESHOLD

    def check_drift(
        self,
        previous_clip: float,
        current_clip: float,
        step: int,
        scene_id: str,
    ) -> float:
        """Compute drift between consecutive refinement steps and enforce limit.

        Args:
            previous_clip: CLIP score before this refinement step
            current_clip:  CLIP score after this refinement step
            step:          refinement step index (for logging)
            scene_id:      scene being processed

        Returns:
            float drift value (0.0–1.0)

        Raises:
            CLIPValidationError if drift exceeds per-step threshold (RULE-93)
        """
        drift = abs(previous_clip - current_clip)

        logger.debug(
            "IdentityDriftController: step=%d scene=%s prev=%.4f cur=%.4f drift=%.4f",
            step, scene_id, previous_clip, current_clip, drift,
        )

        if drift > self._max_drift:
            raise CLIPValidationError(
                f"Identity drift {drift:.4f} at refinement step {step} exceeds "
                f"per-step threshold {self._max_drift:.4f} for scene {scene_id}. RULE-93.",
                stage_id="S-07",
            )

        return drift
