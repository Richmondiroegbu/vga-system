"""
IdentityManager — coordinates identity freeze and cross-phase consistency checks.
Works alongside IdentityStateTracker and CLIPValidator.
Spec: VGA Identity System v17.2 §2; FR-601–FR-608
"""
from __future__ import annotations

import logging
from typing import List, Optional

from vga.config.settings import settings
from vga.core.exceptions import IdentityReferenceCorruptionError
from vga.state.immutable_context import IdentityState, ImmutableContext

logger = logging.getLogger(__name__)


class IdentityManager:
    """Manages the lifecycle of char_identity_ref across pipeline phases.

    The reference embedding is computed ONCE at S-07 (ImageRefinementAgent)
    and frozen in ImmutableContext. All downstream stages use the frozen value.
    """

    def freeze_identity(
        self,
        embedding_vector: List[float],
        context: ImmutableContext,
    ) -> ImmutableContext:
        """Freeze char_identity_ref in context. Called exactly once at S-07.

        Raises IdentityReferenceCorruptionError if already frozen (RULE-95).
        Returns new ImmutableContext with frozen identity_state.
        """
        if context.identity_state.is_frozen:
            raise IdentityReferenceCorruptionError(
                "freeze_identity() called but char_identity_ref is already frozen. "
                "This indicates a pipeline logic error. RULE-95."
            )
        frozen_state = context.identity_state.freeze(embedding_vector)
        new_context = context.evolve(identity_state=frozen_state)
        logger.info(
            "IdentityManager: char_identity_ref FROZEN for scene=%s "
            "embedding_dim=%d",
            context.scene_id, len(embedding_vector),
        )
        return new_context

    def assert_consistency(
        self,
        clip_score: float,
        stage_id: str,
        context: ImmutableContext,
        threshold: Optional[float] = None,
    ) -> None:
        """Assert that clip_score meets threshold. Raises CLIPValidationError if not."""
        from vga.core.exceptions import CLIPValidationError
        effective = threshold or settings.CLIP_IDENTITY_THRESHOLD
        if clip_score < effective:
            raise CLIPValidationError(
                f"Identity consistency check failed at {stage_id}: "
                f"CLIP score {clip_score:.4f} < threshold {effective:.4f}. RULE-92.",
                stage_id=stage_id,
            )
