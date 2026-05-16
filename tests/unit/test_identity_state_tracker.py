"""Unit tests for IdentityStateTracker. RULE-95, RULE-96."""
from __future__ import annotations

import pytest

from vga.core.exceptions import IdentityCumulativeDriftError
from vga.identity.identity_state_tracker import IdentityStateTracker
from vga.state.context_factory import ContextFactory
from vga.state.immutable_context import IdentityState


@pytest.fixture
def tracker():
    return IdentityStateTracker()


@pytest.fixture
def frozen_context():
    ctx = ContextFactory.create_initial("job_test", "sc_001")
    embedding = [0.5] * 768
    frozen_identity = ctx.identity_state.freeze(embedding)
    return ctx.evolve(identity_state=frozen_identity)


def test_update_accumulates_drift(tracker, frozen_context):
    """update() accumulates drift across calls."""
    updated = tracker.update("S-09", "sc_001", clip_score=0.95, context=frozen_context)
    assert updated.drift_score > 0.0
    assert len(updated.history) == 1

    ctx2 = frozen_context.evolve(identity_state=updated)
    updated2 = tracker.update("S-09", "sc_001", clip_score=0.94, context=ctx2)
    assert updated2.cumulative_drift > updated.cumulative_drift
    assert len(updated2.history) == 2


def test_update_raises_on_threshold_exceeded(tracker, frozen_context):
    """update() raises IdentityCumulativeDriftError when cumulative drift > 0.15. RULE-95."""
    ctx = frozen_context
    with pytest.raises(IdentityCumulativeDriftError):
        # Very low CLIP score → high drift → threshold exceeded after several updates
        for _ in range(20):
            identity = tracker.update("S-09", "sc_001", clip_score=0.1, context=ctx)
            ctx = ctx.evolve(identity_state=identity)


def test_identity_state_tracker_records_are_stored(tracker, frozen_context):
    """update() adds a record to the tracker's internal records list."""
    tracker.update("S-09", "sc_001", clip_score=0.95, context=frozen_context)
    records = tracker.get_records()
    assert len(records) == 1
    assert records[0].stage_id == "S-09"
    assert records[0].scene_id == "sc_001"


def test_high_clip_score_produces_low_drift(tracker, frozen_context):
    """A CLIP score of 0.99 should produce very low drift."""
    updated = tracker.update("S-09", "sc_001", clip_score=0.99, context=frozen_context)
    assert updated.drift_score < 0.02   # very high clip → very low drift
