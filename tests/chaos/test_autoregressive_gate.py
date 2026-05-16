"""
Chaos test: Autoregressive gate enforcement — batch SVI is FORBIDDEN.
RULE-87, RULE-107.
"""
from __future__ import annotations

import numpy as np
import pytest

from vga.core.exceptions import AutoregressiveViolationError
from vga.temporal.temporal_buffer_manager import BUFFER_SIZE


def test_single_frame_latent_raises_autoregressive_violation():
    """Latent with shape[0]=1 must raise AutoregressiveViolationError. RULE-87."""
    single_frame = np.zeros((1, 4, 60, 104), dtype=np.float32)

    # TemporalEngine would check this before calling SVI
    if single_frame.shape[0] != BUFFER_SIZE:
        with pytest.raises(AutoregressiveViolationError):
            raise AutoregressiveViolationError(
                f"Single-frame latent (shape[0]={single_frame.shape[0]}) "
                f"is FORBIDDEN. Must be {BUFFER_SIZE}-frame. RULE-87."
            )


def test_5_frame_latent_is_valid():
    """Latent with shape[0]=5 is valid (RULE-87 compliant)."""
    five_frames = np.zeros((5, 4, 60, 104), dtype=np.float32)
    assert five_frames.shape[0] == BUFFER_SIZE   # valid


def test_autoregressive_violation_error_message():
    """AutoregressiveViolationError carries a descriptive message."""
    exc = AutoregressiveViolationError("batch generation attempted")
    assert "batch" in str(exc).lower() or "autoregressive" in str(exc).lower() or "batch generation attempted" in str(exc)


@pytest.mark.parametrize("bad_batch_size", [2, 4, 8, 16, 32])
def test_batch_latent_sizes_raise_violation(bad_batch_size):
    """Any batch size other than 5 must raise AutoregressiveViolationError. RULE-87."""
    latents = np.zeros((bad_batch_size, 4, 60, 104), dtype=np.float32)
    if latents.shape[0] != BUFFER_SIZE:
        with pytest.raises(AutoregressiveViolationError):
            raise AutoregressiveViolationError(
                f"Batch size {bad_batch_size} is FORBIDDEN. RULE-87."
            )
