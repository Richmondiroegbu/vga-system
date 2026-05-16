"""Unit tests for identity system (IdentityState, IdentityStateTracker, IdentityDriftController)."""
from __future__ import annotations

import pytest

from vga.core.exceptions import CLIPValidationError, IdentityCumulativeDriftError, IdentityReferenceCorruptionError
from vga.identity.identity_drift_controller import IdentityDriftController
from vga.identity.identity_state_tracker import IdentityStateTracker
from vga.state.immutable_context import IdentityState


def test_identity_state_update_drift_appends():
    """update_drift() is append-only and accumulates."""
    identity = IdentityState()
    updated = identity.update_drift(0.05)
    assert updated.drift_score == 0.05
    assert updated.cumulative_drift == 0.05
    assert 0.05 in updated.history

    updated2 = updated.update_drift(0.03)
    assert updated2.cumulative_drift == pytest.approx(0.08)
    assert len(updated2.history) == 2


def test_identity_state_freeze_sets_flag():
    """freeze() marks identity as frozen with embedding."""
    identity = IdentityState()
    embedding = [0.5] * 768
    frozen = identity.freeze(embedding)
    assert frozen.is_frozen is True
    assert frozen.embedding_vector == embedding


def test_identity_state_double_freeze_raises():
    """freeze() on already-frozen identity raises IdentityReferenceCorruptionError."""
    identity = IdentityState()
    frozen = identity.freeze([0.5] * 768)
    with pytest.raises(IdentityReferenceCorruptionError):
        frozen.freeze([0.6] * 768)


def test_identity_tracker_raises_on_excess_drift(frozen_context):
    """IdentityStateTracker raises IdentityCumulativeDriftError when drift > 0.15."""
    tracker = IdentityStateTracker()
    # Simulate very low CLIP score to generate high drift
    with pytest.raises(IdentityCumulativeDriftError):
        # Loop to accumulate drift > 0.15
        ctx = frozen_context
        for _ in range(20):
            tracker.update(
                stage_id="S-09",
                scene_id=ctx.scene_id,
                clip_score=0.1,   # very low → drift ~0.9 per step
                context=ctx,
            )


def test_drift_controller_accepts_low_drift():
    """IdentityDriftController accepts drift <= 0.02 (RULE-93)."""
    controller = IdentityDriftController()
    drift = controller.check_drift(
        previous_clip=0.95,
        current_clip=0.94,
        step=1,
        scene_id="sc_001",
    )
    assert drift == pytest.approx(0.01)


def test_drift_controller_rejects_high_drift():
    """IdentityDriftController raises CLIPValidationError for drift > 0.02 (RULE-93)."""
    controller = IdentityDriftController()
    with pytest.raises(CLIPValidationError):
        controller.check_drift(
            previous_clip=0.95,
            current_clip=0.90,   # drift = 0.05 > 0.02
            step=1,
            scene_id="sc_001",
        )
