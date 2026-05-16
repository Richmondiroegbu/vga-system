"""Unit tests for VGA exception hierarchy."""
from __future__ import annotations

import pytest

from vga.core.exceptions import (
    AudioQualityError,
    IdentityCumulativeDriftError,
    SVICFGViolationError,
    TemporalBufferError,
    TemporalSegmentFailureError,
    VGABaseError,
)


def test_vga_base_error():
    exc = VGABaseError("test message", stage_id="S-01")
    assert exc.stage_id == "S-01"
    assert str(exc) == "test message"


def test_temporal_buffer_error_stores_frame_count():
    exc = TemporalBufferError("bad buffer", frame_count=3, required=5)
    assert exc.frame_count == 3
    assert exc.required == 5


def test_svi_cfg_violation_error():
    exc = SVICFGViolationError(cfg_value=7.0)
    assert exc.cfg_value == 7.0
    assert "7.0" in str(exc)
    assert "color banding" in str(exc)


def test_identity_cumulative_drift_error():
    exc = IdentityCumulativeDriftError(drift_score=0.20, threshold=0.15)
    assert exc.drift_score == 0.20
    assert exc.threshold == 0.15
    assert "0.2000" in str(exc)


def test_temporal_segment_failure_error():
    exc = TemporalSegmentFailureError(scene_id="sc_001", segment_id=3)
    assert exc.scene_id == "sc_001"
    assert exc.segment_id == 3
    assert "3" in str(exc)


def test_audio_quality_error():
    exc = AudioQualityError(snr_db=5.0, peak_db=2.0)
    assert exc.snr_db == 5.0
    assert exc.peak_db == 2.0
    assert "5.0" in str(exc)


def test_all_exceptions_inherit_vga_base():
    """Every VGA exception must inherit from VGABaseError."""
    from vga.core.exceptions import (
        ArchitectureGuardViolationError,
        AutoregressiveViolationError,
        CLIPValidationError,
        CompositionPlanValidationError,
        CrossModalAlignmentError,
        IdentityReferenceCorruptionError,
        ImmutableContextViolationError,
        MissingPredecessorOutputError,
    )
    for exc_class in [
        ArchitectureGuardViolationError,
        AutoregressiveViolationError,
        CLIPValidationError,
        CompositionPlanValidationError,
        CrossModalAlignmentError,
        IdentityReferenceCorruptionError,
        ImmutableContextViolationError,
    ]:
        assert issubclass(exc_class, VGABaseError), (
            f"{exc_class.__name__} must inherit from VGABaseError"
        )
