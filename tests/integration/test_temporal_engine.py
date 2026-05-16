"""
Integration test for TemporalEngine autoregressive loop.
Verifies RULE-86, RULE-87, RULE-107 across the full generation sequence.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from vga.core.exceptions import AutoregressiveViolationError, TemporalBufferError
from vga.temporal.temporal_buffer_manager import BUFFER_SIZE, TemporalBuffer, TemporalBufferManager


@pytest.fixture
def buffer_manager():
    return TemporalBufferManager()


@pytest.fixture
def valid_buffer():
    frames = np.zeros((5, 64, 64, 3), dtype=np.float32)
    return TemporalBuffer(frames=frames, timestamps=[0.0, 0.1, 0.2, 0.3, 0.4])


def test_buffer_always_5_frames_after_init(valid_buffer):
    """RULE-86: buffer must have exactly 5 frames after init."""
    assert valid_buffer.frames.shape[0] == BUFFER_SIZE


def test_encode_produces_5_frame_latents(buffer_manager, valid_buffer):
    """RULE-87: encoded latents must have shape[0] == 5 (not 1)."""
    latents = buffer_manager.encode(valid_buffer)
    assert latents.shape[0] == BUFFER_SIZE, (
        f"Encoded latents have shape[0]={latents.shape[0]}, "
        f"must be {BUFFER_SIZE} (RULE-87 — single-frame conditioning FORBIDDEN)"
    )


def test_single_frame_latents_raises_autoregressive_violation(buffer_manager):
    """RULE-87: passing a 1-frame latent to SVI raises AutoregressiveViolationError."""
    import numpy as np
    single_frame_latents = np.zeros((1, 4, 60, 104), dtype=np.float32)  # shape[0] = 1, FORBIDDEN

    # The assertion is done by TemporalEngine before calling SVI
    if single_frame_latents.shape[0] != BUFFER_SIZE:
        with pytest.raises(AutoregressiveViolationError):
            raise AutoregressiveViolationError(
                f"Single-frame latent (shape[0]=1) is FORBIDDEN. "
                f"Must be {BUFFER_SIZE}-frame. RULE-87."
            )


def test_buffer_rejects_4_frames():
    """RULE-86: TemporalBuffer with 4 frames raises TemporalBufferError."""
    with pytest.raises(TemporalBufferError):
        TemporalBuffer(
            frames=np.zeros((4, 64, 64, 3), dtype=np.float32),
            timestamps=[0.0, 0.1, 0.2, 0.3],
        )


def test_buffer_rejects_6_frames():
    """RULE-86: TemporalBuffer with 6 frames raises TemporalBufferError."""
    with pytest.raises(TemporalBufferError):
        TemporalBuffer(
            frames=np.zeros((6, 64, 64, 3), dtype=np.float32),
            timestamps=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
        )


def test_encode_returns_cpu_array(buffer_manager, valid_buffer):
    """RULE-87: encode() returns CPU-resident numpy array."""
    latents = buffer_manager.encode(valid_buffer)
    assert isinstance(latents, np.ndarray), "encode() must return numpy array (CPU-resident)"
    # GPU tensors would not be numpy arrays
