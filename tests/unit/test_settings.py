"""Unit tests for VGASettings. Verifies all critical constants."""
from __future__ import annotations

import pytest

from vga.config.settings import settings


def test_schema_version():
    assert settings.SCHEMA_VERSION == "v6.0"


def test_temporal_buffer_size_is_5():
    """TEMPORAL_BUFFER_SIZE MUST be 5. RULE-86."""
    assert settings.TEMPORAL_BUFFER_SIZE == 5


def test_clip_identity_threshold():
    """CLIP_IDENTITY_THRESHOLD must be 0.93. RULE-92."""
    assert settings.CLIP_IDENTITY_THRESHOLD == 0.93


def test_svi_cfg_range():
    """SVI CFG range must be [5.0, 8.0]. RULE-86. (Max raised to 8.0 — vita-epfl recommends 7.0 for non-distill SVI.)"""
    assert settings.SVI_CFG_MIN == 5.0
    assert settings.SVI_CFG_MAX == 8.0


def test_lora_weights():
    """LoRA weights must be 0.6/0.5/0.4. FR-932–FR-934."""
    assert settings.LORA_WEIGHT_HIGH_NOISE == 0.6
    assert settings.LORA_WEIGHT_MID_NOISE == 0.5
    assert settings.LORA_WEIGHT_LOW_NOISE == 0.4


def test_audio_thresholds():
    """Audio SNR and peak limits. RULE-99."""
    assert settings.MIN_SNR_DB == 10.0
    assert settings.MAX_PEAK_DBFS == 0.0


def test_hrg_checkpoint_count():
    """Must have exactly 11 HRG checkpoints. v17.0."""
    assert settings.HRG_CHECKPOINT_COUNT == 11


def test_identity_cumulative_drift_threshold():
    """Cumulative identity drift threshold must be 0.15."""
    assert settings.IDENTITY_CUMULATIVE_DRIFT_THRESHOLD == 0.15


def test_composition_max_retries():
    assert settings.COMPOSITION_MAX_RETRIES == 3
