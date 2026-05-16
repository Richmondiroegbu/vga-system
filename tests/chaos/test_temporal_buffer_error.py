"""
Chaos test: TemporalBuffer size violations at all entry/exit points.
RULE-86 — buffer MUST contain exactly 5 frames at ALL times.
"""
from __future__ import annotations

import numpy as np
import pytest

from vga.core.exceptions import TemporalBufferError
from vga.temporal.temporal_buffer_manager import BUFFER_SIZE, TemporalBuffer, TemporalBufferManager


@pytest.mark.parametrize("bad_count", [0, 1, 2, 3, 4, 6, 7, 10])
def test_buffer_rejects_wrong_frame_count(bad_count):
    """TemporalBuffer raises immediately for ANY count != 5. RULE-86."""
    with pytest.raises(TemporalBufferError) as exc_info:
        TemporalBuffer(
            frames=np.zeros((bad_count, 64, 64, 3), dtype=np.float32),
            timestamps=[float(i) for i in range(bad_count)],
        )
    assert exc_info.value.frame_count == bad_count
    assert exc_info.value.required == BUFFER_SIZE


def test_assert_buffer_size_raises_immediately():
    """_assert_buffer_size() raises TemporalBufferError immediately. RULE-86."""
    bad_buffer = object.__new__(TemporalBuffer)
    object.__setattr__(bad_buffer, "frames", np.zeros((3, 64, 64, 3), dtype=np.float32))
    object.__setattr__(bad_buffer, "timestamps", [0.0, 0.1, 0.2])
    object.__setattr__(bad_buffer, "scene_id", "")
    object.__setattr__(bad_buffer, "segment_id", "")

    with pytest.raises(TemporalBufferError):
        TemporalBufferManager._assert_buffer_size(bad_buffer)


def test_buffer_error_stores_frame_count():
    """TemporalBufferError stores the actual vs required frame count."""
    with pytest.raises(TemporalBufferError) as exc_info:
        TemporalBuffer(
            frames=np.zeros((3, 64, 64, 3), dtype=np.float32),
            timestamps=[0.0, 0.1, 0.2],
        )
    assert exc_info.value.frame_count == 3
    assert exc_info.value.required == 5


def test_encode_raises_for_wrong_frame_count():
    """encode() called on wrong-size buffer raises TemporalBufferError."""
    manager = TemporalBufferManager()
    bad_buffer = object.__new__(TemporalBuffer)
    object.__setattr__(bad_buffer, "frames", np.zeros((2, 64, 64, 3), dtype=np.float32))
    object.__setattr__(bad_buffer, "timestamps", [0.0, 0.1])
    object.__setattr__(bad_buffer, "scene_id", "")
    object.__setattr__(bad_buffer, "segment_id", "")

    with pytest.raises(TemporalBufferError):
        manager.encode(bad_buffer)
