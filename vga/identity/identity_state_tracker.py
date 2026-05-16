"""
IdentityStateTracker — sole owner of cumulative identity drift tracking.
Updates IdentityState after each stage; triggers regeneration when drift > 0.15.
Spec: VGA Identity System v17.2; RULE-89, RULE-95, RULE-97, FR-600–FR-615
"""
from __future__ import annotations

import logging

from vga.config.settings import settings
from vga.core.exceptions import IdentityCumulativeDriftError
from vga.models.schemas import IdentityStateRecord
from vga.state.immutable_context import IdentityState, ImmutableContext

logger = logging.getLogger(__name__)


class IdentityStateTracker:
    """Tracks cumulative identity drift across all pipeline phases.

    Called by MasterOrchestrator after each stage that produces a CLIP score.
    Raises IdentityCumulativeDriftError when cumulative drift exceeds threshold.
    """

    def __init__(self) -> None:
        self._records: list[IdentityStateRecord] = []
        logger.info(
            "IdentityStateTracker initialized — threshold=%.2f",
            settings.IDENTITY_CUMULATIVE_DRIFT_THRESHOLD,
        )

    def update(
        self,
        stage_id: str,
        scene_id: str,
        clip_score: float,
        context: ImmutableContext,
    ) -> IdentityState:
        """Update IdentityState with new CLIP score; compute and accumulate drift.

        Args:
            stage_id:   pipeline stage that produced the score
            scene_id:   scene being processed
            clip_score: CLIP similarity score [0, 1] for this stage
            context:    current ImmutableContext

        Returns:
            Updated IdentityState (new instance — does not mutate context)

        Raises:
            IdentityCumulativeDriftError when cumulative drift > THRESHOLD
        """
        identity = context.identity_state
        drift = self._compute_drift(clip_score)
        updated_identity = identity.update_drift(drift)

        threshold_exceeded = (
            updated_identity.cumulative_drift > settings.IDENTITY_CUMULATIVE_DRIFT_THRESHOLD
        )

        record = IdentityStateRecord(
            stage_id=stage_id,
            scene_id=scene_id,
            drift_score=drift,
            cumulative_drift=updated_identity.cumulative_drift,
            drift_history=list(updated_identity.history),
            threshold_exceeded=threshold_exceeded,
        )
        self._records.append(record)

        logger.info(
            "IdentityTracker: stage=%s scene=%s clip=%.4f drift=%.4f cumulative=%.4f",
            stage_id, scene_id, clip_score, drift, updated_identity.cumulative_drift,
        )

        if threshold_exceeded:
            raise IdentityCumulativeDriftError(
                drift_score=updated_identity.cumulative_drift,
                threshold=settings.IDENTITY_CUMULATIVE_DRIFT_THRESHOLD,
                stage_id=stage_id,
            )

        return updated_identity

    def get_records(self) -> list[IdentityStateRecord]:
        """Return all recorded IdentityStateRecords for audit."""
        return list(self._records)

    def _compute_drift(self, clip_score: float) -> float:
        """Convert CLIP score to drift value. Higher clip score = lower drift."""
        return max(0.0, 1.0 - clip_score)
