"""Unit tests for CrossModalAlignmentValidator. FR-972."""
from __future__ import annotations

from unittest.mock import patch

import pytest

from vga.core.exceptions import CrossModalAlignmentError
from vga.validation.cross_modal_alignment_validator import CrossModalAlignmentValidator


def test_validate_passes_within_tolerance(tmp_path):
    """validate() passes when alignment error ≤ 0.10s."""
    video = tmp_path / "test.mp4"
    audio = tmp_path / "test.wav"
    video.write_bytes(b"\x00" * 100)
    audio.write_bytes(b"\x00" * 100)

    validator = CrossModalAlignmentValidator()
    with patch.object(validator, "_get_video_duration", return_value=5.0):
        with patch.object(validator, "_get_audio_duration", return_value=5.05):  # 50ms error
            record = validator.validate(str(video), str(audio), "sc_001")

    assert record.within_tolerance is True
    assert abs(record.alignment_error_s) <= 0.10


def test_validate_raises_outside_tolerance(tmp_path):
    """validate() raises CrossModalAlignmentError when error > 0.10s. FR-972."""
    video = tmp_path / "test.mp4"
    audio = tmp_path / "test.wav"
    video.write_bytes(b"\x00" * 100)
    audio.write_bytes(b"\x00" * 100)

    validator = CrossModalAlignmentValidator()
    with patch.object(validator, "_get_video_duration", return_value=5.0):
        with patch.object(validator, "_get_audio_duration", return_value=5.5):  # 500ms error
            with pytest.raises(CrossModalAlignmentError):
                validator.validate(str(video), str(audio), "sc_001")


def test_validate_record_has_correct_fields(tmp_path):
    """Returned CrossModalAlignmentRecord has all required fields."""
    video = tmp_path / "test.mp4"
    audio = tmp_path / "test.wav"
    video.write_bytes(b"\x00" * 100)
    audio.write_bytes(b"\x00" * 100)

    validator = CrossModalAlignmentValidator()
    with patch.object(validator, "_get_video_duration", return_value=4.0):
        with patch.object(validator, "_get_audio_duration", return_value=4.0):
            record = validator.validate(str(video), str(audio), "sc_001", "seg_001")

    assert record.scene_id == "sc_001"
    assert record.segment_id == "seg_001"
    assert record.schema_version == "v6.0"
    assert record.within_tolerance is True
