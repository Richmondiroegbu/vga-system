"""Unit tests for TemporalBufferManager. RULE-86, RULE-87."""
from __future__ import annotations

import numpy as np
import pytest

from vga.core.exceptions import AutoregressiveViolationError, TemporalBufferError
from vga.temporal.temporal_buffer_manager import BUFFER_SIZE, TemporalBuffer, TemporalBufferManager


def test_buffer_must_have_5_frames_on_creation(valid_buffer):
    """TemporalBuffer with 5 frames creates successfully."""
    assert valid_buffer.frames.shape[0] == 5


def test_buffer_creation_with_wrong_size_raises():
    """TemporalBuffer with != 5 frames raises TemporalBufferError."""
    with pytest.raises(TemporalBufferError) as exc_info:
        TemporalBuffer(
            frames=np.zeros((3, 64, 64, 3), dtype=np.float32),
            timestamps=[0.0, 0.1, 0.2],
        )
    assert exc_info.value.frame_count == 3
    assert exc_info.value.required == 5


def test_assert_buffer_size_raises_on_wrong_count():
    """_assert_buffer_size raises immediately if frames.shape[0] != 5."""
    bad_frames = np.zeros((4, 64, 64, 3), dtype=np.float32)
    with pytest.raises(TemporalBufferError):
        bad_buffer = object.__new__(TemporalBuffer)
        object.__setattr__(bad_buffer, "frames", bad_frames)
        object.__setattr__(bad_buffer, "timestamps", [0.0] * 4)
        object.__setattr__(bad_buffer, "scene_id", "")
        object.__setattr__(bad_buffer, "segment_id", "")
        TemporalBufferManager._assert_buffer_size(bad_buffer)


def test_buffer_size_constant_is_5():
    """TEMPORAL_BUFFER_SIZE must always be 5. RULE-86."""
    assert BUFFER_SIZE == 5


def test_encode_returns_cpu_array(valid_buffer):
    """encode() must return a CPU numpy array, not a GPU tensor."""
    manager = TemporalBufferManager()
    latents = manager.encode(valid_buffer)

    assert isinstance(latents, np.ndarray), "encode() must return numpy array (CPU-resident)"
    assert latents.shape[0] == 5, "Encoded latents must have 5-frame dimension (RULE-87)"


def test_encode_rejects_wrong_buffer_size():
    """encode() raises if buffer has wrong frame count."""
    manager = TemporalBufferManager()
    with pytest.raises(TemporalBufferError):
        # Manually create invalid buffer bypassing __post_init__
        bad = object.__new__(TemporalBuffer)
        object.__setattr__(bad, "frames", np.zeros((3, 64, 64, 3), dtype=np.float32))
        object.__setattr__(bad, "timestamps", [0.0, 0.1, 0.2])
        object.__setattr__(bad, "scene_id", "")
        object.__setattr__(bad, "segment_id", "")
        manager.encode(bad)


def test_normalization_uses_fixed_constants(valid_buffer):
    """Normalization must use fixed constants, not compute per-call means."""
    original = valid_buffer.frames.copy()
    normalized = TemporalBufferManager._normalize_frames(original)
    # Normalized values should not equal the original (since we applied mean/std)
    # (original is all zeros → normalized is (0 - mean) / std)
    assert not np.allclose(normalized, original)
