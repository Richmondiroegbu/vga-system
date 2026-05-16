# Prompt 02: Core Foundation — Settings, Schemas, Exceptions, Logger
**Category:** Core Foundation  
**Files to implement:**
- `vga/config/settings.py`
- `vga/models/schemas.py`
- `vga/core/exceptions.py`
- `vga/core/logger.py`
- `vga/models/enums.py`
**Spec References:** 
- `01_VGA_System_Requirements_Document_v17.2.md` §9 (Data Requirements, schemas)
- `10_VGA_Coding_Standards_and_Rules_v17.2.md` (RULE-01 through RULE-20)
- `05_VGA_Data_Contracts_Interface_Specification_v17.2.md`  
**Dependencies:** Phase 1 complete (project skeleton exists)  
**Build Order:** Step 12.0.1 → 12.0.2 → 12.0.3 (must be first files written)

---

## Context

These files form the **lowest-level foundation** of the entire VGA system. Every other file imports from them. Get these right first — all higher layers depend on them.

**Critical principle:** Constants live ONLY in `settings.py`. Never hardcode threshold values inline in any other file.

---

## vga/config/settings.py

Implement a complete settings module using Pydantic `BaseSettings` (or plain constants) with ALL VGA constants organized by domain:

```python
"""
VGA v17.2 Configuration Settings
Single source of truth for ALL system constants.
Spec: VGA SRD v17.2 §5, §16
"""
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings  # or pydantic v2 BaseSettings
import os

class VGASettings(BaseSettings):
    # === Core Identity ===
    SCHEMA_VERSION: str = "v6.0"
    SYSTEM_VERSION: str = "17.2.0"
    MISSION: str = "Inspire audiences by telling stories of people who overcame adversity"
    
    # === Workspace Paths ===
    WORKSPACE_ROOT: Path = Path("/workspace")
    MODELS_DIR: Path = Path("/workspace/models")
    LORAS_DIR: Path = Path("/workspace/loras")
    ASSETS_DIR: Path = Path("/workspace/assets")
    OUTPUT_DIR: Path = Path("/workspace/output")
    LOGS_DIR: Path = Path("/workspace/logs")
    STATE_DIR: Path = Path("/workspace/state")
    HRG_DIR: Path = Path("/workspace/hrg")
    
    # === Model Paths ===
    QWEN_MODEL_PATH: Path = Path("/workspace/models/qwen")
    FLUX2_MODEL_PATH: Path = Path("/workspace/models/flux2")
    ZIMAGE_MODEL_PATH: Path = Path("/workspace/models/zimage")
    WAN22_MODEL_PATH: Path = Path("/workspace/models/wan22")
    SVI_MODEL_PATH: Path = Path("/workspace/models/svi")
    LATENTSYNC_PATH: Path = Path("/workspace/LatentSync")
    COSYVOICE_PATH: Path = Path("/workspace/CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B")
    MUSICGEN_MODEL_PATH: Path = Path("/workspace/models/musicgen")
    MMAUDIO_PATH: Path = Path("/workspace/MMAudio")
    CLIP_MODEL_PATH: Path = Path("/workspace/auxiliary/clip")
    SVI_REPO_PATH: Path = Path("/workspace/Stable-Video-Infinity")
    
    # === LoRA Paths ===
    SVI_HIGH_NOISE_PATH: Path = Path("/workspace/loras/svi/SVI_Wan2.2-I2V-A14B_high_noise_lora_v2.0_pro.safetensors")
    SVI_LOW_NOISE_PATH: Path = Path("/workspace/loras/svi/SVI_Wan2.2-I2V-A14B_low_noise_lora_v2.0_pro.safetensors")
    CONSISTENCY_LORA_PATH: Path = Path("/workspace/loras/consistency")
    
    # === SVI Environment ===
    SVI_REPO_BRANCH: str = "svi_wan22"
    SVI_WAN22_PYTHON: str = "/opt/conda/envs/svi_wan22/bin/python"
    
    # === Identity Thresholds ===
    CLIP_IDENTITY_THRESHOLD: float = 0.93     # RULE-92: minimum CLIP score everywhere
    CLIP_DRIFT_THRESHOLD: float = 0.02        # RULE-93: max drift in image refinement
    IDENTITY_CUMULATIVE_DRIFT_THRESHOLD: float = 0.15  # Full regeneration trigger
    LIPSYNC_IDENTITY_DELTA_THRESHOLD: float = 0.03     # RULE-97
    IDENTITY_MAX_PHASE_REGENERATIONS: int = 1
    
    # === Temporal Engine (RULE-86, RULE-87) ===
    TEMPORAL_BUFFER_SIZE: int = 5             # MUST be 5, always
    TEMPORAL_MAX_RETRIES_PER_SEGMENT: int = 3
    SEGMENT_CONTINUITY_MIN: float = 0.85
    
    # === SVI CFG + Steps (FR-936, FR-937) ===
    SVI_CFG_MIN: float = 5.0
    SVI_CFG_MAX: float = 6.0
    SVI_CFG_DEFAULT: float = 5.5
    STEPS_CRITICAL: int = 50   # 30–50 for critical segments
    STEPS_STANDARD: int = 30   # 4–8 for standard segments (use 30 as minimum quality)
    
    # === SVI LoRA Scheduling (FR-932–FR-934) ===
    LORA_WEIGHT_HIGH_NOISE: float = 0.6       # t > HIGH_NOISE_FRACTION * T
    LORA_WEIGHT_MID_NOISE: float = 0.5        # MID_NOISE_FRACTION < t <= HIGH_NOISE_FRACTION
    LORA_WEIGHT_LOW_NOISE: float = 0.4        # t <= MID_NOISE_FRACTION * T
    HIGH_NOISE_FRACTION: float = 0.67         # fraction of total diffusion steps
    MID_NOISE_FRACTION: float = 0.33
    
    # === Motion ===
    MOTION_STATIONARY_THRESHOLD: float = 0.02
    
    # === Scene Composition ===
    COMPOSITION_MAX_RETRIES: int = 3
    SLA_COMPOSITION_MAX_S: float = 15.0
    
    # === Audio Quality (RULE-99) ===
    MIN_SNR_DB: float = 10.0
    MAX_PEAK_DBFS: float = 0.0
    HEADROOM_DB: float = 1.0
    AUDIO_QUALITY_MAX_RETRIES: int = 3
    
    # === Audio Mixing Levels ===
    DIALOGUE_LEVEL_DB: float = 0.0
    AMBIENT_LEVEL_DB: float = -12.0
    MUSIC_LEVEL_DB: float = -18.0
    DUCKING_DB: float = -6.0           # ambient/music reduction during dialogue
    
    # === Cross-Modal Alignment (FR-972) ===
    CROSS_MODAL_SYNC_THRESHOLD: float = 0.9
    TIMING_TOLERANCE_S: float = 0.10
    
    # === HRG ===
    HRG_CHECKPOINT_COUNT: int = 11
    HRG_REVIEW_ENABLED: bool = True
    HRG_APPROVAL_TIMEOUT_SECONDS: int = 300
    
    # === Continuity Scoring Weights ===
    CONTINUITY_MOTION_WEIGHT: float = 0.40
    CONTINUITY_LIGHTING_WEIGHT: float = 0.30
    CONTINUITY_IDENTITY_WEIGHT: float = 0.30
    
    # === Video / Image Quality ===
    BASE_IMAGE_COUNT: int = 6
    SCENE_DURATION_MIN_S: float = 10.0
    SCENE_DURATION_MAX_S: float = 30.0
    SEGMENT_DURATION_MIN_S: float = 3.0
    SEGMENT_DURATION_MAX_S: float = 5.0
    IMAGE_CLIP_SCORE_MIN: float = 0.93
    
    # === FLUX.2-klein Settings ===
    FLUX_IDENTITY_LORA_WEIGHT_MIN: float = 0.4
    FLUX_IDENTITY_LORA_WEIGHT_MAX: float = 0.7
    FLUX_CFG: float = 1.0
    FLUX_STEPS: int = 4
    
    # === Z-Image-Turbo Settings ===
    ZIMAGE_DENOISE_MIN: float = 0.05
    ZIMAGE_DENOISE_MAX: float = 0.15
    ZIMAGE_CFG: float = 5.0
    
    # === VRAM ===
    VRAM_ENFORCE_HARD_LIMIT: bool = True
    VRAM_FREE_RATIO_MIN: float = 0.90   # After GPU cleanup, assert ≥ this
    
    # === Retry / Backoff ===
    MAX_RETRIES: int = 3
    BACKOFF_SECONDS: list = [5, 15, 45]
    
    # === Immutable Context ===
    IMMUTABLE_CONTEXT_ENFORCE: bool = True
    
    # === SLA KPIs (v15.0) ===
    SLA_SCRIPT_MAX_S: float = 30.0
    SLA_SCENE_PLAN_MAX_S: float = 10.0
    SLA_IDENTITY_DESIGN_MAX_S: float = 20.0
    SLA_BASE_IMAGE_MAX_S: float = 60.0
    SLA_SEGMENT_GEN_MAX_S: float = 120.0    # standard
    SLA_SEGMENT_GEN_CRITICAL_MAX_S: float = 300.0  # critical
    SLA_LIPSYNC_MAX_S: float = 30.0
    SLA_AUDIO_MIX_MAX_S: float = 10.0
    SLA_EXPORT_MAX_S: float = 30.0
    
    class Config:
        env_file = "/workspace/.env_vga"
        env_file_encoding = "utf-8"
        case_sensitive = True

# Singleton instance
settings = VGASettings()
```

---

## vga/core/exceptions.py

Implement the complete exception hierarchy:

```python
"""
VGA Exception Hierarchy.
All exceptions derive from VGABaseError.
Import exceptions from HERE — never define inline.
Spec: VGA File Responsibility Spec v17.2 §core/exceptions.py
"""

class VGABaseError(Exception):
    """Base class for all VGA exceptions."""
    def __init__(self, message: str, stage_id: str | None = None, **kwargs):
        self.stage_id = stage_id
        self.details = kwargs
        super().__init__(message)

# === v15.0 and earlier exceptions (retained) ===
class CriticalPipelineError(VGABaseError): ...
class PipelineAbortError(VGABaseError): ...
class SchemaValidationError(VGABaseError): ...
class ModelLoadError(VGABaseError): ...
class ModelUnloadError(VGABaseError): ...
class VRAMViolationError(VGABaseError): ...
class CLIPValidationError(VGABaseError): ...
class SLAViolationError(VGABaseError): ...
class HRGTimeoutError(VGABaseError): ...
class HRGRejectionError(VGABaseError): ...
class RetryExhaustedError(VGABaseError): ...
class ArchitectureGuardViolationError(VGABaseError): ...
class LoRALoadError(VGABaseError): ...
class OutputValidationError(VGABaseError): ...
class ContextEvolutionError(VGABaseError): ...

# === v17.0 NEW exceptions ===
class CompositionPlanValidationError(VGABaseError):
    """CompositionPlan schema validation failed. RULE-88."""
    ...

class TemporalBufferError(VGABaseError):
    """TemporalBuffer constraint violated (size ≠ 5, resolution mismatch). RULE-86."""
    def __init__(self, message: str, frame_count: int | None = None, required: int = 5, **kwargs):
        self.frame_count = frame_count
        self.required = required
        super().__init__(message, **kwargs)

class SVICFGViolationError(VGABaseError):
    """SVI CFG outside [5.0, 6.0]. RULE-86."""
    def __init__(self, cfg_value: float, **kwargs):
        self.cfg_value = cfg_value
        super().__init__(f"SVI CFG {cfg_value} outside [5.0, 6.0] — color banding risk", **kwargs)

class AutoregressiveViolationError(VGABaseError):
    """Batch SVI generation or single-frame conditioning attempted. RULE-87."""
    ...

class TemporalSegmentFailureError(VGABaseError):
    """All retries exhausted for a temporal segment. RULE-87."""
    def __init__(self, scene_id: str, segment_id: int, **kwargs):
        self.scene_id = scene_id
        self.segment_id = segment_id
        super().__init__(f"Temporal segment {segment_id} in scene {scene_id} failed after all retries", **kwargs)

class IdentityCumulativeDriftError(VGABaseError):
    """Cumulative identity drift exceeded threshold. RULE-95."""
    def __init__(self, drift_score: float, threshold: float, **kwargs):
        self.drift_score = drift_score
        self.threshold = threshold
        super().__init__(f"Identity cumulative drift {drift_score:.4f} exceeds threshold {threshold}", **kwargs)

class IdentityReferenceCorruptionError(VGABaseError):
    """char_identity_ref was recomputed or mutated mid-pipeline. RULE-95."""
    ...

class AudioQualityError(VGABaseError):
    """SNR < 10dB or peaks > 0 dBFS. RULE-99."""
    def __init__(self, snr_db: float | None = None, peak_db: float | None = None, **kwargs):
        self.snr_db = snr_db
        self.peak_db = peak_db
        msg = f"Audio quality failure: SNR={snr_db:.1f}dB (min 10), peak={peak_db:.1f}dBFS (max 0)"
        super().__init__(msg, **kwargs)

class MissingPredecessorOutputError(VGABaseError):
    """Stage tried to run without required predecessor output. RULE-90."""
    def __init__(self, stage_id: str, required_output: str, **kwargs):
        super().__init__(f"Stage {stage_id} requires {required_output} from predecessor", **kwargs)

class SVISchedulerViolationError(VGABaseError):
    """Static LoRA weight used in SVI generation. RULE-86."""
    ...

class CrossModalAlignmentError(VGABaseError):
    """Video-audio alignment error exceeds ±0.10s tolerance. FR-972."""
    ...

class ImmutableContextViolationError(VGABaseError):
    """Dict-based context used instead of ImmutableContext. RULE-108."""
    ...
```

---

## vga/models/schemas.py

Implement ALL Pydantic v2 schemas. Include ALL fields with validation:

### Key schemas to implement (partial list — implement all):

```python
"""
VGA Data Contracts — All Pydantic Schemas
schema_version: v6.0 for all v17.2 artifacts
Spec: VGA Data Contracts Interface Specification v17.2
"""
from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Dict, Any
from enum import Enum

# v17.0 NEW schemas (critical — implement these completely):

class CompositionPlanSchema(BaseModel):
    """Mandatory output of SceneCompositionAgent (S-04). RULE-88. schema_version v6.0."""
    scene_id: str
    camera_angle: str       # "medium shot", "close-up", "wide shot", etc.
    camera_motion: str      # "slow dolly forward", "static", "pan left", etc.
    character_positions: List[Dict[str, Any]]  # [{character_id, position, facing}]
    focus_subject: str      # "main_character"
    lighting_style: str     # "low-key dramatic", "soft natural", etc.
    motion_vector: str      # "forward_slow", "stationary", "right_medium", etc.
    schema_version: str = "v6.0"
    
    @field_validator("camera_angle")
    @classmethod
    def validate_camera_angle(cls, v: str) -> str:
        valid = {"extreme close-up", "close-up", "medium close-up", "medium shot",
                 "medium wide shot", "wide shot", "extreme wide shot", "overhead",
                 "low angle", "high angle", "dutch angle", "eye level"}
        if v.lower() not in valid:
            raise ValueError(f"Invalid camera_angle: {v!r}. Must be one of {valid}")
        return v

class TemporalBufferRecord(BaseModel):
    """Logged after each TemporalBuffer update. OR-034. schema_version v6.0."""
    segment_id: str
    scene_id: str
    frame_count: int = Field(5, ge=5, le=5)  # MUST be exactly 5
    timestamps: List[float] = Field(min_length=5, max_length=5)
    scene_id_ref: str
    schema_version: str = "v6.0"

class MotionStateRecord(BaseModel):
    """Logged per segment by MotionStateTracker. OR-035."""
    segment_id: str
    scene_id: str
    velocity_magnitude: float
    direction: str
    velocity_vector: Optional[List[float]] = None
    schema_version: str = "v6.0"

class IdentityStateRecord(BaseModel):
    """Logged per stage transition. OR-036."""
    stage_id: str
    scene_id: str
    drift_score: float
    cumulative_drift: float
    drift_history: List[float]
    threshold_exceeded: bool
    schema_version: str = "v6.0"

class AudioQualityRecord(BaseModel):
    """Logged after AudioMixingAgent. OR-037."""
    scene_id: str
    snr_db: float
    peak_db: float
    clipping_detected: bool
    snr_passed: bool      # snr_db >= MIN_SNR_DB (10.0)
    clipping_passed: bool  # peak_db <= 0.0
    schema_version: str = "v6.0"

class CrossModalAlignmentRecord(BaseModel):
    """Logged after cross-modal validation. FR-972."""
    scene_id: str
    segment_id: str
    video_duration_s: float
    audio_duration_s: float
    alignment_error_s: float
    within_tolerance: bool   # abs(error) <= 0.10s
    schema_version: str = "v6.0"

# Also implement: ScriptSchema, ScenePlanSchema, SegmentPlanSchema, 
# IdentityDesignSchema, SVIGenerationRecord, HRG2DisplayData,
# HRG4DisplayData, HRG8DisplayData, HRG10DisplayData, HRG11DisplayData,
# TemporalBufferStatusResponse, IdentityStateResponse, AudioValidationResponse,
# CompositionPlanUpdateRequest, ContinuityReport (with identity_per_segment field),
# PipelineReport (with v17.0 fields), TemporalState, CameraState, LightingState
```

---

## vga/models/enums.py

```python
"""VGA Enumerations. Spec: VGA v17.2."""
from enum import Enum, auto

class PipelineStageID(str, Enum):
    S01_SCRIPT = "S-01"
    S02_SCENE_PLAN = "S-02"
    S03_IDENTITY_DESIGN = "S-03"
    S04_SCENE_COMPOSITION = "S-04"   # NEW v17.0
    S05_BASE_IMAGE = "S-05"
    S06_IDENTITY_REINFORCEMENT = "S-06"
    S07_IMAGE_REFINEMENT = "S-07"
    S08_VIDEO_SEGMENT_1 = "S-08"     # NEW v17.0 (Wan2.2)
    S09_TEMPORAL_ENGINE = "S-09"     # SVI autoregressive
    S10_CONTINUITY_VALIDATION = "S-10"
    S11_DIALOGUE = "S-11"
    S12_LIP_SYNC = "S-12"
    S13_AMBIENT_AUDIO = "S-13"
    S14_MUSIC = "S-14"
    S15_AUDIO_MIXING = "S-15"
    S16_EXPORT = "S-16"

class HRGCheckpoint(str, Enum):
    HRG_1_SCRIPT = "HRG-1"
    HRG_2_SCENE_PLAN = "HRG-2"      # NEW v17.0
    HRG_3_IDENTITY = "HRG-3"
    HRG_4_COMPOSITION = "HRG-4"     # NEW v17.0
    HRG_5_BASE_IMAGES = "HRG-5"
    HRG_6_IDENTITY_REINFORCEMENT = "HRG-6"
    HRG_7_REFINED_IMAGE = "HRG-7"
    HRG_8_MOTION_QA = "HRG-8"
    HRG_9_VOICE_QA = "HRG-9"
    HRG_10_LIPSYNC_QA = "HRG-10"
    HRG_11_FINAL_AUDIO_QA = "HRG-11"

class TemporalPhase(str, Enum):
    """SVI diffusion noise phases for LoRA scheduling."""
    HIGH_NOISE = "high_noise"    # t > 0.67*T → LoRA weight 0.6
    MID_NOISE = "mid_noise"      # 0.33*T < t <= 0.67*T → LoRA weight 0.5
    LOW_NOISE = "low_noise"      # t <= 0.33*T → LoRA weight 0.4

class GatingMode(str, Enum):
    STRICT = "STRICT"     # all validations enforced
    BALANCED = "BALANCED" # standard validations
    FAST = "FAST"         # minimal validations (non-production only)

class CompositionState(str, Enum):
    PENDING = "pending"
    GENERATING = "generating"
    VALIDATED = "validated"
    APPROVED = "approved"  # after HRG-4

class FailureSeverity(str, Enum):
    CRITICAL = "CRITICAL"   # pipeline halts
    DEGRADED = "DEGRADED"   # retry then escalate
    WARNING = "WARNING"     # log and continue
```

---

## vga/core/logger.py

```python
"""
Structured logging for VGA v17.2.
Uses Python structlog (or stdlib logging with JSON formatter).
All events are machine-parseable JSON lines.
"""
# Implement:
# - get_logger(name) → returns structured logger with VGA context
# - trace_event(event_name, **kwargs) — for pipeline stage events
# - JSON output to /workspace/logs/vga_{date}.jsonl
# - Console output with human-readable formatting
# - Log levels: DEBUG, INFO, WARNING, ERROR, CRITICAL
# - Always include: timestamp, level, stage_id, scene_id, job_id (when available)
```

---

## Acceptance Criteria

- [ ] `from vga.config.settings import settings; assert settings.TEMPORAL_BUFFER_SIZE == 5`
- [ ] `from vga.config.settings import settings; assert settings.CLIP_IDENTITY_THRESHOLD == 0.93`
- [ ] `from vga.core.exceptions import TemporalBufferError, SVICFGViolationError` imports successfully
- [ ] `from vga.models.schemas import CompositionPlanSchema` — validation works with all required fields
- [ ] `CompositionPlanSchema(scene_id="test", camera_angle="invalid")` raises `ValidationError`
- [ ] `from vga.models.enums import PipelineStageID, HRGCheckpoint, TemporalPhase` — all 11 HRG checkpoints exist
- [ ] `AudioQualityRecord(scene_id="s1", snr_db=8.0, peak_db=-1.0, clipping_detected=False, snr_passed=False, clipping_passed=True)` validates correctly
