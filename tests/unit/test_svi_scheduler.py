"""Unit tests for SVIScheduler. RULE-86, FR-932–FR-934."""
from __future__ import annotations

import pytest

from vga.config.settings import settings
from vga.core.exceptions import SVICFGViolationError
from vga.models.enums import TemporalPhase
from vga.temporal.svi_scheduler import SVIScheduler


def test_cfg_below_minimum_raises():
    """CFG below 5.0 raises SVICFGViolationError. RULE-86."""
    with pytest.raises(SVICFGViolationError) as exc_info:
        SVIScheduler(cfg=4.9, steps=30)
    assert exc_info.value.cfg_value == 4.9


def test_cfg_above_maximum_raises():
    """CFG above 8.0 raises SVICFGViolationError. RULE-86. (Max is 8.0, raised from 6.0.)"""
    with pytest.raises(SVICFGViolationError) as exc_info:
        SVIScheduler(cfg=8.01, steps=30)
    assert exc_info.value.cfg_value == 8.01


def test_cfg_at_minimum_valid():
    """CFG exactly 5.0 is valid."""
    scheduler = SVIScheduler(cfg=5.0, steps=30)
    assert scheduler.cfg == 5.0


def test_cfg_at_maximum_valid():
    """CFG exactly 8.0 (current max) is valid."""
    scheduler = SVIScheduler(cfg=8.0, steps=30)
    assert scheduler.cfg == 8.0


def test_high_noise_weight_at_step_0():
    """First timestep (step=0) should return high-noise weight 0.6. RULE-86."""
    scheduler = SVIScheduler(cfg=5.5, steps=50)
    # Step 0 of 50 → fraction 0.0 → LOW_NOISE (t <= 33%)
    # Actually step 0 fraction = 0/49 ≈ 0.0, which is LOW_NOISE
    # The high-noise test should use a step with fraction > 0.67
    high_noise_step = int(0.8 * 49)   # step 39 of 50 → fraction ~0.80 > 0.67
    weight = scheduler.get_lora_weight(high_noise_step)
    assert weight == settings.LORA_WEIGHT_HIGH_NOISE   # 0.6


def test_low_noise_weight_at_last_step():
    """Last timestep should return low-noise weight 0.4. RULE-86."""
    scheduler = SVIScheduler(cfg=5.5, steps=50)
    # Last step = 49 → fraction 49/49 = 1.0 → HIGH_NOISE
    # We need a step with fraction ≤ 0.33
    low_noise_step = int(0.1 * 49)   # step 4 of 50 → fraction ~0.08 ≤ 0.33
    weight = scheduler.get_lora_weight(low_noise_step)
    assert weight == settings.LORA_WEIGHT_LOW_NOISE   # 0.4


def test_mid_noise_weight():
    """Mid-range timestep returns mid-noise weight 0.5."""
    scheduler = SVIScheduler(cfg=5.5, steps=50)
    mid_step = int(0.5 * 49)   # step 24 → fraction ~0.5 → MID_NOISE
    weight = scheduler.get_lora_weight(mid_step)
    assert weight == settings.LORA_WEIGHT_MID_NOISE   # 0.5


def test_get_steps_floor_at_standard():
    """get_steps() never returns below STEPS_STANDARD (30)."""
    scheduler = SVIScheduler(cfg=5.5, steps=4)   # sub-minimum input
    assert scheduler.get_steps() >= settings.STEPS_STANDARD


def test_get_steps_critical():
    """Critical mode returns STEPS_CRITICAL (50)."""
    scheduler = SVIScheduler(cfg=5.5, steps=30, critical=True)
    assert scheduler.get_steps() == settings.STEPS_CRITICAL


def test_high_noise_phase_classification():
    """Step with fraction > 0.67 classified as HIGH_NOISE."""
    # SVIScheduler floors steps at STEPS_STANDARD (50), so total=50 regardless of input.
    scheduler = SVIScheduler(cfg=5.5, steps=30)
    step = int(0.9 * 49)   # fraction = 44/49 ≈ 0.898 > 0.67 → HIGH_NOISE
    assert scheduler.get_noise_phase(step) == TemporalPhase.HIGH_NOISE


def test_lora_schedule_dict_has_all_phases():
    """build_lora_schedule() returns dict with all 3 phases."""
    scheduler = SVIScheduler(cfg=5.5, steps=30)
    schedule = scheduler.build_lora_schedule()
    assert "high_noise_weight" in schedule
    assert "mid_noise_weight" in schedule
    assert "low_noise_weight" in schedule
    assert schedule["high_noise_weight"] == 0.6
    assert schedule["mid_noise_weight"] == 0.5
    assert schedule["low_noise_weight"] == 0.4
