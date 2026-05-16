"""Unit tests for VGA Pydantic schemas."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from vga.models.schemas import (
    AudioQualityRecord,
    CompositionPlanSchema,
    TemporalBufferRecord,
)


def test_composition_plan_all_required_fields():
    """CompositionPlanSchema requires all 6 fields."""
    plan = CompositionPlanSchema(
        scene_id="sc_001",
        camera_angle="medium shot",
        camera_motion="slow dolly forward",
        character_positions=[{"character_id": "hero", "position": "center"}],
        focus_subject="main_character",
        lighting_style="soft natural",
        motion_vector="forward_slow",
    )
    assert plan.schema_version == "v6.0"


def test_composition_plan_invalid_camera_angle():
    """CompositionPlanSchema raises ValidationError for invalid camera_angle."""
    with pytest.raises(ValidationError):
        CompositionPlanSchema(
            scene_id="sc_001",
            camera_angle="invalid_angle",
            camera_motion="static",
            character_positions=[{"character_id": "hero"}],
            focus_subject="main_character",
            lighting_style="natural",
            motion_vector="stationary",
        )


def test_composition_plan_empty_character_positions():
    """Empty character_positions raises ValidationError."""
    with pytest.raises(ValidationError):
        CompositionPlanSchema(
            scene_id="sc_001",
            camera_angle="medium shot",
            camera_motion="static",
            character_positions=[],   # empty — forbidden
            focus_subject="main_character",
            lighting_style="natural",
            motion_vector="stationary",
        )


def test_temporal_buffer_record_frame_count_5():
    """TemporalBufferRecord requires frame_count exactly 5."""
    record = TemporalBufferRecord(
        segment_id="seg_001",
        scene_id="sc_001",
        frame_count=5,
        timestamps=[0.0, 0.1, 0.2, 0.3, 0.4],
        scene_id_ref="sc_001",
    )
    assert record.frame_count == 5


def test_temporal_buffer_record_wrong_frame_count():
    """TemporalBufferRecord rejects frame_count != 5."""
    with pytest.raises(ValidationError):
        TemporalBufferRecord(
            segment_id="seg_001",
            scene_id="sc_001",
            frame_count=3,   # wrong!
            timestamps=[0.0, 0.1, 0.2],
            scene_id_ref="sc_001",
        )


def test_audio_quality_record_passes():
    """AudioQualityRecord validates correctly."""
    rec = AudioQualityRecord(
        scene_id="sc_001",
        snr_db=15.0,
        peak_db=-3.0,
        clipping_detected=False,
        snr_passed=True,
        clipping_passed=True,
    )
    assert rec.snr_passed is True


def test_audio_quality_record_failed_snr():
    """AudioQualityRecord with snr_passed=False is valid (records failure)."""
    rec = AudioQualityRecord(
        scene_id="sc_001",
        snr_db=5.0,
        peak_db=-3.0,
        clipping_detected=False,
        snr_passed=False,
        clipping_passed=True,
    )
    assert rec.snr_passed is False
