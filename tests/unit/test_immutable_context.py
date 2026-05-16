"""Unit tests for ImmutableContext. RULE-108, FR-950–FR-954."""
from __future__ import annotations

import pytest

from vga.core.exceptions import (
    CompositionPlanValidationError,
    IdentityReferenceCorruptionError,
    ImmutableContextViolationError,
)
from vga.state.context_factory import ContextFactory
from vga.state.immutable_context import (
    CameraState,
    IdentityState,
    ImmutableContext,
    LightingState,
    MotionState,
    TemporalState,
)


def test_context_is_frozen(valid_context):
    """ImmutableContext must be a frozen dataclass — direct attribute assignment fails."""
    with pytest.raises((AttributeError, TypeError)):
        valid_context.scene_id = "new_scene"


def test_evolve_creates_new_instance(valid_context):
    """evolve() returns a NEW ImmutableContext; original is unchanged."""
    new_ctx = valid_context.evolve(scene_id="new_scene")
    assert new_ctx is not valid_context
    assert new_ctx.scene_id == "new_scene"
    assert valid_context.scene_id == "test_scene"   # original unchanged


def test_evolve_preserves_unspecified_fields(valid_context):
    """evolve() carries all unspecified fields forward unchanged."""
    new_ctx = valid_context.evolve(current_stage="S-01")
    assert new_ctx.job_id == valid_context.job_id
    assert new_ctx.scene_id == valid_context.scene_id
    assert new_ctx.current_stage == "S-01"


def test_dict_context_raises_violation_error():
    """Creating an ImmutableContext with dict dimensions raises ImmutableContextViolationError."""
    with pytest.raises(ImmutableContextViolationError):
        ImmutableContext(
            identity_state={"embedding_vector": None},   # dict — FORBIDDEN
            motion_state=MotionState(),
            camera_state=CameraState(),
            lighting_state=LightingState(),
            temporal_state=TemporalState(),
            job_id="job_1",
            scene_id="scene_1",
        )


def test_context_factory_validate_rejects_dict():
    """ContextFactory.validate() raises on dict input. RULE-108."""
    with pytest.raises(ImmutableContextViolationError):
        ContextFactory.validate({"scene_id": "test"})


def test_context_factory_validate_accepts_immutable_context(valid_context):
    """ContextFactory.validate() returns context unchanged if valid."""
    result = ContextFactory.validate(valid_context)
    assert result is valid_context


def test_assert_composition_plan_raises_when_missing(valid_context):
    """assert_composition_plan() raises CompositionPlanValidationError when plan is None."""
    with pytest.raises(CompositionPlanValidationError):
        valid_context.assert_composition_plan()


def test_identity_freeze_double_freeze_raises():
    """Calling freeze() twice raises IdentityReferenceCorruptionError. RULE-95."""
    identity = IdentityState()
    embedding = [0.1] * 768
    frozen = identity.freeze(embedding)

    with pytest.raises(IdentityReferenceCorruptionError):
        frozen.freeze([0.2] * 768)   # second freeze attempt — forbidden


def test_identity_freeze_sets_frozen_flag():
    """freeze() sets is_frozen=True."""
    identity = IdentityState()
    frozen = identity.freeze([0.1] * 768)
    assert frozen.is_frozen is True
    assert frozen.embedding_vector == [0.1] * 768


def test_with_stage_completed_tracks_stages(valid_context):
    """with_stage_completed() accumulates completed stage IDs."""
    ctx1 = valid_context.with_stage_completed("S-01", output={"script": "hello"})
    ctx2 = ctx1.with_stage_completed("S-02")
    assert "S-01" in ctx2.completed_stages
    assert "S-02" in ctx2.completed_stages
    assert ctx2.has_output("S-01") is True
    assert ctx2.has_output("S-02") is False


def test_create_initial_returns_zeroed_state():
    """ContextFactory.create_initial() returns context with all zero/default dimensions."""
    ctx = ContextFactory.create_initial("job_abc", "scene_xyz")
    assert ctx.job_id == "job_abc"
    assert ctx.scene_id == "scene_xyz"
    assert ctx.identity_state.is_frozen is False
    assert ctx.identity_state.cumulative_drift == 0.0
    assert ctx.composition_plan is None
    assert ctx.completed_stages == ()
