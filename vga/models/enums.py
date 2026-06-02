"""
VGA Enumerations.
Spec: VGA Data Contracts Interface Specification v17.2 §2
"""
from __future__ import annotations

from enum import Enum


class PipelineStageID(str, Enum):
    """All 16 pipeline stage identifiers."""

    S01_SCRIPT = "S-01"
    S02_SCENE_PLAN = "S-02"
    S03_IDENTITY_DESIGN = "S-03"
    S04_SCENE_COMPOSITION = "S-04"    # NEW v17.0
    S05_BASE_IMAGE = "S-05"
    S06_IDENTITY_REINFORCEMENT = "S-06"
    S07_IMAGE_REFINEMENT = "S-07"
    S08_VIDEO_SEGMENT_1 = "S-08"     # NEW v17.0 — Wan2.2 generates Segment_1
    S09_TEMPORAL_ENGINE = "S-09"     # SVI autoregressive loop (Segments 2..N)
    S10_CONTINUITY_VALIDATION = "S-10"
    S11_DIALOGUE = "S-11"
    S12_LIP_SYNC = "S-12"
    S13_AMBIENT_AUDIO = "S-13"
    S14_MUSIC = "S-14"
    S15_AUDIO_MIXING = "S-15"
    S16_EXPORT = "S-16"


class HRGCheckpoint(str, Enum):
    """All 11 Human Review Gate checkpoints (v17.0 adds HRG-2 and HRG-4)."""

    HRG_1_SCRIPT = "HRG-1"
    HRG_2_SCENE_PLAN = "HRG-2"              # NEW v17.0
    HRG_3_IDENTITY = "HRG-3"
    HRG_4_COMPOSITION = "HRG-4"             # NEW v17.0
    HRG_5_BASE_IMAGES = "HRG-5"
    HRG_6_IDENTITY_REINFORCEMENT = "HRG-6"
    HRG_7_REFINED_IMAGE = "HRG-7"
    HRG_8_MOTION_QA = "HRG-8"
    HRG_9_VOICE_QA = "HRG-9"
    HRG_10_LIPSYNC_QA = "HRG-10"
    HRG_11_FINAL_AUDIO_QA = "HRG-11"


class TemporalPhase(str, Enum):
    """SVI diffusion noise phases for dynamic LoRA scheduling. RULE-86."""

    HIGH_NOISE = "high_noise"    # t > 0.67*T → LoRA weight 0.6
    MID_NOISE = "mid_noise"      # 0.33*T < t ≤ 0.67*T → LoRA weight 0.5
    LOW_NOISE = "low_noise"      # t ≤ 0.33*T → LoRA weight 0.4


class GatingMode(str, Enum):
    """Adaptive gating mode. STRICT is default and required for production."""

    STRICT = "STRICT"       # all validations enforced
    BALANCED = "BALANCED"   # standard validations
    FAST = "FAST"           # minimal validations (non-production only)


class CompositionState(str, Enum):
    """Lifecycle state of a CompositionPlan for a scene."""

    PENDING = "pending"
    GENERATING = "generating"
    VALIDATED = "validated"
    APPROVED = "approved"    # after HRG-4 human approval


class FailureSeverity(str, Enum):
    """Pipeline failure classification used by SystemGuard."""

    CRITICAL = "CRITICAL"    # pipeline halts immediately
    DEGRADED = "DEGRADED"    # retry up to 3 times then escalate
    WARNING = "WARNING"      # log and continue


class MotionDirection(str, Enum):
    """Dominant optical-flow direction estimated by MotionStateTracker."""

    FORWARD = "forward"
    BACKWARD = "backward"
    LEFT = "left"
    RIGHT = "right"
    STATIONARY = "stationary"


class AudioChannel(str, Enum):
    """Named audio channels with mixing priority."""

    DIALOGUE = "dialogue"    # 0 dB — highest priority
    AMBIENT = "ambient"      # −12 dB
    MUSIC = "music"          # −18 dB


class TransitionMode(str, Enum):
    """Camera angle transition strategy at segment group boundaries (multi-reference I2V).

    NONE      — normal continuation, no angle change.
    HARD_CUT  — Strategy A: switch input_image + anchor to new angle reference;
                raise denoising_strength to 0.90 so the model ignores old motion
                vectors and re-generates mostly from the new reference. Sharp editorial
                cut style — 10-20 frame pose-reconciliation, then stable new angle.
    BLEND     — Strategy C: pixel-space cosine-ramp blend of the last 4 conditioning
                frames toward the new angle reference (α 0→0.25 over 4 frames);
                hard-switch input_image to new angle; raise denoising_strength to
                0.80. Produces a gradual 10-20 frame cross-dissolve transition.
                Suitable for motivated scene reveals or camera pans.
    """

    NONE = "none"
    HARD_CUT = "hard_cut"
    BLEND = "blend"
