"""
Regression guard for the temporal engine autoregressive loop.
Ensures RULE-86, RULE-87, RULE-107 are never silently broken across refactors.
Phase 12.18 — mandatory temporal loop integration test.
"""
from __future__ import annotations

import numpy as np
import pytest

from vga.core.exceptions import AutoregressiveViolationError, TemporalBufferError
from vga.temporal.temporal_buffer_manager import BUFFER_SIZE, TemporalBuffer, TemporalBufferManager
from vga.temporal.svi_scheduler import SVIScheduler


class TestTemporalRegressionGuard:
    """9 mandatory regression tests per Phase 12.18 spec."""

    def test_r1_buffer_size_constant_is_5(self):
        """R1: BUFFER_SIZE must be exactly 5. Any change breaks RULE-86."""
        assert BUFFER_SIZE == 5

    def test_r2_buffer_creation_enforces_5_frames(self):
        """R2: TemporalBuffer() raises for any count != 5."""
        for bad_count in [1, 2, 3, 4, 6, 10]:
            with pytest.raises(TemporalBufferError):
                TemporalBuffer(
                    frames=np.zeros((bad_count, 64, 64, 3), dtype=np.float32),
                    timestamps=[float(i) * 0.1 for i in range(bad_count)],
                )

    def test_r3_valid_buffer_has_correct_shape(self):
        """R3: Valid buffer has shape (5, H, W, 3)."""
        buf = TemporalBuffer(
            frames=np.zeros((5, 64, 64, 3), dtype=np.float32),
            timestamps=[0.0, 0.1, 0.2, 0.3, 0.4],
        )
        assert buf.frames.shape == (5, 64, 64, 3)

    def test_r4_encode_returns_5_frame_latents(self):
        """R4: encode() output shape[0] must be 5. RULE-87."""
        manager = TemporalBufferManager()
        buf = TemporalBuffer(
            frames=np.zeros((5, 64, 64, 3), dtype=np.float32),
            timestamps=[0.0, 0.1, 0.2, 0.3, 0.4],
        )
        latents = manager.encode(buf)
        assert latents.shape[0] == 5

    def test_r5_encode_returns_numpy_array_not_tensor(self):
        """R5: encode() returns CPU numpy array — not a GPU tensor. RULE-87 device discipline."""
        manager = TemporalBufferManager()
        buf = TemporalBuffer(
            frames=np.zeros((5, 64, 64, 3), dtype=np.float32),
            timestamps=[0.0, 0.1, 0.2, 0.3, 0.4],
        )
        latents = manager.encode(buf)
        assert isinstance(latents, np.ndarray)

    def test_r6_svi_scheduler_rejects_cfg_below_5(self):
        """R6: SVIScheduler raises on CFG < 5.0. RULE-86."""
        from vga.core.exceptions import SVICFGViolationError
        with pytest.raises(SVICFGViolationError):
            SVIScheduler(cfg=4.99, steps=30)

    def test_r7_svi_scheduler_rejects_cfg_above_8(self):
        """R7: SVIScheduler raises on CFG > 8.0. RULE-86. (Max raised from 6.0 to 8.0.)"""
        from vga.core.exceptions import SVICFGViolationError
        with pytest.raises(SVICFGViolationError):
            SVIScheduler(cfg=8.01, steps=30)

    def test_r8_lora_weight_varies_by_phase(self):
        """R8: LoRA weight must differ between high/mid/low noise phases. RULE-86 dynamic scheduling."""
        scheduler = SVIScheduler(cfg=5.5, steps=50)
        high_noise_step = int(0.85 * 49)   # > 0.67 * 49
        low_noise_step = int(0.1 * 49)     # < 0.33 * 49

        weight_high = scheduler.get_lora_weight(high_noise_step)
        weight_low = scheduler.get_lora_weight(low_noise_step)

        assert weight_high != weight_low, "RULE-86: static LoRA weight is FORBIDDEN — must vary per phase"
        assert weight_high == 0.6
        assert weight_low == 0.4

    def test_r9_immutable_context_dict_rejection(self):
        """R9: Dict passed as context must raise ImmutableContextViolationError. RULE-108."""
        from vga.core.exceptions import ImmutableContextViolationError
        from vga.state.context_factory import ContextFactory
        with pytest.raises(ImmutableContextViolationError):
            ContextFactory.validate({"job_id": "test", "scene_id": "sc_001"})
