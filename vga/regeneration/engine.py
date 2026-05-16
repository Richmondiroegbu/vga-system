"""
RegenerationEngine — decides and executes phase regeneration when quality fails.
Triggered by IdentityCumulativeDriftError or quality gate failure.
Spec: VGA Codebase Structure Design v17.2 §regeneration/engine.py
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from vga.config.settings import settings
from vga.core.exceptions import CriticalPipelineError
from vga.state.immutable_context import ImmutableContext

logger = logging.getLogger(__name__)


@dataclass
class RegenerationDecision:
    should_regenerate: bool
    phase: str   # "image", "video", "audio", "full"
    reason: str
    max_allowed: int = 1   # IDENTITY_MAX_PHASE_REGENERATIONS


class RegenerationEngine:
    """Determines whether and how to regenerate a phase after quality failure.

    Maximum 1 phase regeneration allowed (settings.IDENTITY_MAX_PHASE_REGENERATIONS).
    Full pipeline regeneration is NEVER triggered — only targeted phase regeneration.
    """

    def __init__(self) -> None:
        self._regeneration_counts: dict[str, int] = {}

    def decide(
        self,
        failure_type: str,
        context: ImmutableContext,
        exc: Exception,
    ) -> RegenerationDecision:
        """Determine if regeneration should be triggered and which phase.

        Args:
            failure_type: exception class name
            context:      current ImmutableContext
            exc:          the exception that triggered the decision

        Returns:
            RegenerationDecision

        Raises:
            CriticalPipelineError if regeneration limit exceeded
        """
        scene_id = context.scene_id
        count = self._regeneration_counts.get(scene_id, 0)
        max_allowed = settings.IDENTITY_MAX_PHASE_REGENERATIONS

        if count >= max_allowed:
            raise CriticalPipelineError(
                f"RegenerationEngine: scene {scene_id} has already been regenerated "
                f"{count} time(s) — maximum {max_allowed} allowed. "
                f"Manual intervention required."
            )

        phase_map = {
            "IdentityCumulativeDriftError": "image",
            "CLIPValidationError": "image",
            "AudioQualityError": "audio",
            "CrossModalAlignmentError": "audio",
            "TemporalSegmentFailureError": "video",
        }
        phase = phase_map.get(failure_type, "image")

        logger.warning(
            "RegenerationEngine: triggering %s phase regeneration for scene=%s "
            "(attempt %d/%d) reason=%s",
            phase, scene_id, count + 1, max_allowed, failure_type,
        )

        return RegenerationDecision(
            should_regenerate=True,
            phase=phase,
            reason=f"{failure_type}: {exc}",
            max_allowed=max_allowed,
        )

    def record_regeneration(self, scene_id: str) -> None:
        """Record that a regeneration occurred for a scene."""
        self._regeneration_counts[scene_id] = self._regeneration_counts.get(scene_id, 0) + 1

    def reset(self, scene_id: str) -> None:
        """Reset regeneration count after successful completion."""
        self._regeneration_counts.pop(scene_id, None)
