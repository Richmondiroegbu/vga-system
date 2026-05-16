"""Chaos tests enforcing critical architecture rules."""
from __future__ import annotations

import numpy as np
import pytest

from vga.core.exceptions import (
    AutoregressiveViolationError,
    CompositionPlanValidationError,
    IdentityReferenceCorruptionError,
    ImmutableContextViolationError,
    SVICFGViolationError,
    TemporalBufferError,
)
from vga.state.context_factory import ContextFactory
from vga.state.immutable_context import (
    CameraState,
    ImmutableContext,
    IdentityState,
    LightingState,
    MotionState,
    TemporalState,
)
from vga.temporal.svi_scheduler import SVIScheduler
from vga.temporal.temporal_buffer_manager import TemporalBuffer, TemporalBufferManager


# RULE-86: TemporalBuffer size enforcement

def test_rule_86_buffer_rejects_4_frames():
    with pytest.raises(TemporalBufferError):
        TemporalBuffer(
            frames=np.zeros((4, 64, 64, 3), dtype=np.float32),
            timestamps=[0.0] * 4,
        )


def test_rule_86_buffer_rejects_6_frames():
    with pytest.raises(TemporalBufferError):
        TemporalBuffer(
            frames=np.zeros((6, 64, 64, 3), dtype=np.float32),
            timestamps=[0.0] * 6,
        )


def test_rule_86_svi_cfg_below_range_raises():
    with pytest.raises(SVICFGViolationError):
        SVIScheduler(cfg=4.99, steps=30)


def test_rule_86_svi_cfg_above_range_raises():
    with pytest.raises(SVICFGViolationError):
        SVIScheduler(cfg=6.01, steps=30)


# RULE-88: CompositionPlan enforcement

def test_rule_88_assert_raises_without_plan(valid_context):
    with pytest.raises(CompositionPlanValidationError):
        valid_context.assert_composition_plan()


# RULE-95: Identity reference corruption

def test_rule_95_double_freeze_raises():
    identity = IdentityState()
    frozen = identity.freeze([0.1] * 768)
    with pytest.raises(IdentityReferenceCorruptionError):
        frozen.freeze([0.2] * 768)


def test_rule_95_assert_identity_frozen_fails_when_unfrozen(valid_context):
    with pytest.raises(IdentityReferenceCorruptionError):
        valid_context.assert_identity_frozen()


# RULE-108: ImmutableContext enforcement

def test_rule_108_dict_context_rejected():
    with pytest.raises(ImmutableContextViolationError):
        ImmutableContext(
            identity_state={"key": "value"},   # dict — FORBIDDEN
            motion_state=MotionState(),
            camera_state=CameraState(),
            lighting_state=LightingState(),
            temporal_state=TemporalState(),
            job_id="j1",
            scene_id="s1",
        )


def test_rule_108_context_factory_validate_rejects_dict():
    with pytest.raises(ImmutableContextViolationError):
        ContextFactory.validate({"scene_id": "bad"})


# RULE-87: encode raises on wrong latent shape

def test_rule_87_encode_returns_5_frame_latents(valid_buffer):
    manager = TemporalBufferManager()
    latents = manager.encode(valid_buffer)
    assert latents.shape[0] == 5, "Encoded latents MUST have 5-frame shape (RULE-87)"
