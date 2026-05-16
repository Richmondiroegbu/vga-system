"""Unit tests for AudioQualityValidator. RULE-99."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from vga.core.exceptions import AudioQualityError
from vga.validation.audio_quality_validator import AudioQualityValidator


def test_validate_returns_record_on_pass(tmp_path):
    """validate() returns AudioQualityRecord when SNR and peak pass."""
    audio_file = tmp_path / "test.wav"
    audio_file.write_bytes(b"\x00" * 44100)   # dummy WAV

    validator = AudioQualityValidator()
    with patch.object(validator, "_measure", return_value=(15.0, -3.0)):
        record = validator.validate(str(audio_file), "sc_001")

    assert record.snr_passed is True
    assert record.clipping_passed is True
    assert record.snr_db == 15.0
    assert record.peak_db == -3.0
    assert record.schema_version == "v6.0"


def test_validate_raises_on_low_snr(tmp_path):
    """validate() raises AudioQualityError when SNR < 10dB. RULE-99."""
    audio_file = tmp_path / "test.wav"
    audio_file.write_bytes(b"\x00" * 100)

    validator = AudioQualityValidator()
    with patch.object(validator, "_measure", return_value=(5.0, -3.0)):
        with pytest.raises(AudioQualityError) as exc_info:
            validator.validate(str(audio_file), "sc_001")
    assert exc_info.value.snr_db == 5.0


def test_validate_raises_on_clipping(tmp_path):
    """validate() raises AudioQualityError when peak > 0 dBFS. RULE-99."""
    audio_file = tmp_path / "test.wav"
    audio_file.write_bytes(b"\x00" * 100)

    validator = AudioQualityValidator()
    with patch.object(validator, "_measure", return_value=(12.0, 1.5)):  # peak > 0
        with pytest.raises(AudioQualityError) as exc_info:
            validator.validate(str(audio_file), "sc_001")
    assert exc_info.value.peak_db == 1.5


def test_validate_snr_passed_false_but_clipping_ok_raises(tmp_path):
    """Both conditions checked independently."""
    audio_file = tmp_path / "test.wav"
    audio_file.write_bytes(b"\x00" * 100)

    validator = AudioQualityValidator()
    with patch.object(validator, "_measure", return_value=(8.0, -5.0)):
        with pytest.raises(AudioQualityError):
            validator.validate(str(audio_file), "sc_001")


def test_validate_missing_file_uses_defaults():
    """validate() on missing file uses fallback defaults (above threshold)."""
    validator = AudioQualityValidator()
    # Should not raise for missing file (returns defaults)
    record = validator.validate("/nonexistent/path.wav", "sc_001")
    assert record.snr_db > 0   # fallback returns reasonable defaults
