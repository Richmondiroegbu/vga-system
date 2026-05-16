"""
Chaos test: SVI CFG violations — all CFG values outside [5.0, 6.0].
RULE-86, FR-936.
"""
from __future__ import annotations

import pytest

from vga.core.exceptions import SVICFGViolationError
from vga.temporal.svi_scheduler import SVIScheduler


@pytest.mark.parametrize("bad_cfg", [0.0, 1.0, 4.99, 6.01, 7.0, 10.0, -1.0, 100.0])
def test_svi_scheduler_raises_on_out_of_range_cfg(bad_cfg):
    """SVIScheduler raises SVICFGViolationError for CFG outside [5.0, 6.0]. RULE-86."""
    with pytest.raises(SVICFGViolationError) as exc_info:
        SVIScheduler(cfg=bad_cfg, steps=30)
    assert exc_info.value.cfg_value == bad_cfg


@pytest.mark.parametrize("valid_cfg", [5.0, 5.5, 6.0, 5.1, 5.9])
def test_svi_scheduler_accepts_valid_cfg(valid_cfg):
    """SVIScheduler accepts all CFG values within [5.0, 6.0]."""
    scheduler = SVIScheduler(cfg=valid_cfg, steps=30)
    assert scheduler.cfg == valid_cfg


def test_svi_cfg_violation_error_stores_value():
    """SVICFGViolationError stores the invalid CFG value."""
    with pytest.raises(SVICFGViolationError) as exc_info:
        SVIScheduler(cfg=9.9, steps=30)
    assert exc_info.value.cfg_value == 9.9
    assert "9.90" in str(exc_info.value) or "9.9" in str(exc_info.value)


def test_svi_scheduler_does_not_silently_clamp():
    """SVIScheduler MUST NOT silently clamp CFG — must raise instead. RULE-86."""
    # If clamping occurred, no exception would be raised
    # Spec requires raising, never clamping silently
    with pytest.raises(SVICFGViolationError):
        SVIScheduler(cfg=4.0, steps=30)   # below minimum — must raise, not clamp to 5.0
