"""
Chaos test: Identity cumulative drift exceeding threshold.
RULE-95 — cumulative drift > 0.15 triggers IdentityCumulativeDriftError.
"""
from __future__ import annotations

import pytest

from vga.core.exceptions import IdentityCumulativeDriftError
from vga.identity.identity_state_tracker import IdentityStateTracker
from vga.state.context_factory import ContextFactory


@pytest.fixture
def frozen_context():
    ctx = ContextFactory.create_initial("job_test", "sc_001")
    frozen = ctx.identity_state.freeze([0.5] * 768)
    return ctx.evolve(identity_state=frozen)


def test_high_drift_accumulates_and_triggers_error(frozen_context):
    """Multiple low CLIP scores accumulate drift > 0.15 → IdentityCumulativeDriftError."""
    tracker = IdentityStateTracker()
    ctx = frozen_context

    with pytest.raises(IdentityCumulativeDriftError) as exc_info:
        for _ in range(25):   # many iterations with very low CLIP → drift accumulates
            identity = tracker.update("S-09", "sc_001", clip_score=0.05, context=ctx)
            ctx = ctx.evolve(identity_state=identity)

    assert exc_info.value.drift_score > 0.15
    assert exc_info.value.threshold == 0.15


def test_low_drift_does_not_trigger_error(frozen_context):
    """High CLIP scores → low drift → no error raised.
    drift = 1 - clip_score. With clip=0.99, drift=0.01/step.
    5 steps × 0.01 = 0.05 cumulative — well below 0.15 threshold.
    """
    tracker = IdentityStateTracker()
    ctx = frozen_context

    # 5 iterations at very high clip → cumulative drift stays well below threshold
    for _ in range(5):
        identity = tracker.update("S-09", "sc_001", clip_score=0.99, context=ctx)
        ctx = ctx.evolve(identity_state=identity)

    assert ctx.identity_state.cumulative_drift < 0.10


def test_identity_cumulative_drift_error_stores_values():
    """IdentityCumulativeDriftError stores drift_score and threshold."""
    exc = IdentityCumulativeDriftError(drift_score=0.20, threshold=0.15)
    assert exc.drift_score == 0.20
    assert exc.threshold == 0.15
    assert "0.2000" in str(exc) or "0.20" in str(exc)
