"""
Integration test: identity reference preservation across pipeline phases.
Verifies RULE-95 — same char_identity_ref across ALL phases.
"""
from __future__ import annotations

import pytest

from vga.core.exceptions import IdentityReferenceCorruptionError
from vga.state.context_factory import ContextFactory
from vga.state.immutable_context import IdentityState


@pytest.fixture
def base_context():
    return ContextFactory.create_initial("job_test", "sc_001")


def test_identity_reference_frozen_once(base_context):
    """char_identity_ref can only be frozen once. RULE-95."""
    embedding_v1 = [0.1] * 768
    frozen_identity = base_context.identity_state.freeze(embedding_v1)

    with pytest.raises(IdentityReferenceCorruptionError):
        frozen_identity.freeze([0.2] * 768)   # second freeze → FORBIDDEN


def test_frozen_identity_preserved_across_evolve(base_context):
    """evolve() preserves frozen identity reference unchanged."""
    embedding = [0.5] * 768
    frozen = base_context.identity_state.freeze(embedding)
    ctx_frozen = base_context.evolve(identity_state=frozen)

    # evolve on another dimension should not affect identity
    ctx_evolved = ctx_frozen.evolve(current_stage="S-09")
    assert ctx_evolved.identity_state.is_frozen is True
    assert ctx_evolved.identity_state.embedding_vector == embedding


def test_assert_identity_frozen_raises_if_not_frozen(base_context):
    """assert_identity_frozen() raises if identity is not yet frozen."""
    with pytest.raises(IdentityReferenceCorruptionError):
        base_context.assert_identity_frozen()


def test_assert_identity_frozen_passes_if_frozen(base_context):
    """assert_identity_frozen() does not raise when identity is frozen."""
    embedding = [0.5] * 768
    frozen_identity = base_context.identity_state.freeze(embedding)
    ctx = base_context.evolve(identity_state=frozen_identity)
    ctx.assert_identity_frozen()   # should not raise


def test_embedding_vector_unchanged_across_drift_updates(base_context):
    """IdentityState.update_drift() never changes embedding_vector."""
    embedding = [0.7] * 768
    frozen = base_context.identity_state.freeze(embedding)

    updated = frozen.update_drift(0.02)
    assert updated.embedding_vector == embedding   # unchanged
    assert updated.drift_score == 0.02
    assert updated.cumulative_drift == 0.02
