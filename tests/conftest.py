"""
VGA Test Configuration — shared fixtures for all test modules.
Provides valid_buffer (5-frame TemporalBuffer) and valid_context (ImmutableContext).
"""
from __future__ import annotations

import numpy as np
import pytest

from vga.state.context_factory import ContextFactory
from vga.state.immutable_context import ImmutableContext
from vga.temporal.temporal_buffer_manager import TemporalBuffer


@pytest.fixture
def valid_buffer() -> TemporalBuffer:
    """5-frame TemporalBuffer with correct dimensions (RULE-86)."""
    frames = np.zeros((5, 64, 64, 3), dtype=np.float32)
    timestamps = [0.0, 0.1, 0.2, 0.3, 0.4]
    return TemporalBuffer(frames=frames, timestamps=timestamps, scene_id="test_scene")


@pytest.fixture
def valid_context() -> ImmutableContext:
    """ImmutableContext initialized via ContextFactory."""
    return ContextFactory.create_initial(job_id="test_job", scene_id="test_scene")


@pytest.fixture
def frozen_context(valid_context: ImmutableContext) -> ImmutableContext:
    """ImmutableContext with frozen identity_state (char_identity_ref set)."""
    embedding = [0.1] * 768
    frozen_identity = valid_context.identity_state.freeze(embedding)
    return valid_context.evolve(identity_state=frozen_identity)
