"""Unit tests for MotionStateTracker. NFR-168 (≤ 1.0s SLA)."""
from __future__ import annotations

import time

import numpy as np
import pytest

from vga.temporal.motion_state_tracker import MotionStateTracker
from vga.temporal.temporal_buffer_manager import TemporalBuffer


@pytest.fixture
def stationary_buffer():
    """TemporalBuffer with identical frames (stationary motion)."""
    frames = np.zeros((5, 64, 64, 3), dtype=np.float32)
    return TemporalBuffer(frames=frames, timestamps=[0.0, 0.1, 0.2, 0.3, 0.4])


@pytest.fixture
def moving_buffer():
    """TemporalBuffer with frames showing gradual brightness increase (simulates motion)."""
    frames = np.stack([
        np.ones((64, 64, 3), dtype=np.float32) * (i * 0.1)
        for i in range(5)
    ])
    return TemporalBuffer(frames=frames, timestamps=[0.0, 0.1, 0.2, 0.3, 0.4])


def test_estimate_returns_motion_state(stationary_buffer):
    """estimate() returns a MotionState with required fields."""
    tracker = MotionStateTracker()
    motion = tracker.estimate(stationary_buffer)
    assert hasattr(motion, "velocity_vector")
    assert hasattr(motion, "direction")
    assert hasattr(motion, "magnitude")


def test_estimate_stationary_buffer_returns_low_magnitude(stationary_buffer):
    """Identical frames → near-zero magnitude (stationary)."""
    tracker = MotionStateTracker()
    motion = tracker.estimate(stationary_buffer)
    assert motion.magnitude < 0.5   # stationary should have very low magnitude


def test_estimate_completes_within_sla(stationary_buffer):
    """estimate() must complete in ≤ 1.0 second (NFR-168)."""
    tracker = MotionStateTracker()
    start = time.monotonic()
    tracker.estimate(stationary_buffer)
    elapsed = time.monotonic() - start
    assert elapsed < 1.0, f"estimate() took {elapsed:.2f}s — exceeds 1.0s SLA (NFR-168)"


def test_estimate_velocity_vector_is_tuple(stationary_buffer):
    """velocity_vector is a 2-tuple (dx, dy)."""
    tracker = MotionStateTracker()
    motion = tracker.estimate(stationary_buffer)
    assert isinstance(motion.velocity_vector, tuple)
    assert len(motion.velocity_vector) == 2


def test_classify_direction_stationary():
    """_classify_direction returns 'stationary' for near-zero velocity."""
    from vga.temporal.motion_state_tracker import _STATIONARY_THRESHOLD
    direction = MotionStateTracker._classify_direction((0.0, 0.0), 0.0)
    assert direction == "stationary"
