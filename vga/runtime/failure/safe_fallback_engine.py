"""
SafeFallbackEngine — provides safe fallback actions for recoverable failures.
Spec: VGA Runtime Spec v17.2 §failure/safe_fallback_engine.py
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class SafeFallbackEngine:
    """Determines and executes safe fallback behaviour for DEGRADED failures."""

    def get_fallback_action(self, stage_id: str, exc: BaseException) -> str:
        """Return a string describing the fallback action for this failure.

        Args:
            stage_id: pipeline stage where failure occurred
            exc:      the exception

        Returns:
            str describing the action taken (for logging/audit)
        """
        exc_name = type(exc).__name__

        fallback_map = {
            "CLIPValidationError": "retry_with_adjusted_prompt",
            "AudioQualityError": "normalize_and_remix",
            "CompositionPlanValidationError": "retry_composition_generation",
            "IdentityCumulativeDriftError": "regenerate_from_last_stable",
            "CrossModalAlignmentError": "trim_to_shorter_duration",
        }

        action = fallback_map.get(exc_name, "retry_stage")
        logger.info("SafeFallbackEngine: stage=%s exc=%s → action=%s", stage_id, exc_name, action)
        return action

    def apply_fallback(self, action: str, stage_id: str, context: object) -> None:
        """Log the fallback action. Actual execution is handled by the calling agent."""
        logger.warning(
            "SafeFallbackEngine: applying '%s' for stage %s", action, stage_id
        )
