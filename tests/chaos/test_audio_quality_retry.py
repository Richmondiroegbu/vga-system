"""
Chaos test: Audio quality failure detection and retry logic.
RULE-99 — SNR ≥ 10dB + peaks ≤ 0 dBFS enforced.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from vga.core.exceptions import AudioQualityError
from vga.models.schemas import AudioQualityRecord
from vga.validation.audio_quality_validator import AudioQualityValidator


def test_audio_quality_error_on_snr_below_10db(tmp_path):
    """AudioQualityError raised when SNR < 10dB. RULE-99."""
    audio_file = tmp_path / "bad_audio.wav"
    audio_file.write_bytes(b"\x00" * 100)

    validator = AudioQualityValidator()
    with patch.object(validator, "_measure", return_value=(5.0, -3.0)):
        with pytest.raises(AudioQualityError) as exc_info:
            validator.validate(str(audio_file), "sc_001")
    assert exc_info.value.snr_db == 5.0


def test_audio_quality_error_on_clipping(tmp_path):
    """AudioQualityError raised when peaks > 0 dBFS. RULE-99."""
    audio_file = tmp_path / "clipped.wav"
    audio_file.write_bytes(b"\x00" * 100)

    validator = AudioQualityValidator()
    with patch.object(validator, "_measure", return_value=(15.0, 2.0)):  # clipping
        with pytest.raises(AudioQualityError) as exc_info:
            validator.validate(str(audio_file), "sc_001")
    assert exc_info.value.peak_db == 2.0


def test_audio_quality_record_snr_failed_flag():
    """AudioQualityRecord correctly records snr_passed=False."""
    record = AudioQualityRecord(
        scene_id="sc_001",
        snr_db=7.5,
        peak_db=-2.0,
        clipping_detected=False,
        snr_passed=False,
        clipping_passed=True,
    )
    assert record.snr_passed is False
    assert record.clipping_passed is True


def test_audio_quality_record_clipping_flag():
    """AudioQualityRecord correctly records clipping_detected=True."""
    record = AudioQualityRecord(
        scene_id="sc_001",
        snr_db=15.0,
        peak_db=1.5,
        clipping_detected=True,
        snr_passed=True,
        clipping_passed=False,
    )
    assert record.clipping_detected is True
    assert record.clipping_passed is False
