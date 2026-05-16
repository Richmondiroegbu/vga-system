# VGA Data Contracts & Interface Specification
**Project:** Video Generation Automation (VGA)
**Version:** 17.2.0
**Status:** Authoritative Reference
**Audience:** Pipeline Engineers, Agent Implementors, Claude Code Agent

---

## Table of Contents

1. [Overview](#1-overview)
2. [Schema Version](#2-schema-version)
3–19. [All v15.0 Schemas Retained](#3-19-all-v150-schemas-retained)
20. [Full Pipeline Stage Schemas (v16.0 — retained)](#20-full-pipeline-stage-schemas)
21. [Image Pipeline Schemas (v16.0 — retained)](#21-image-pipeline-schemas)
22. [Audio Pipeline Schemas (v16.0 — retained)](#22-audio-pipeline-schemas)
23. [HRG Decision Schemas (updated for 11 checkpoints)](#23-hrg-decision-schemas)
24. [API Request/Response Schemas (v16.0 — retained)](#24-api-requestresponse-schemas)
25. [Regeneration Request Schema (unchanged)](#25-regeneration-request-schema)
26. [Pipeline Report Schema (v17.0 additions)](#26-pipeline-report-schema)
27. [File System Artifact Map (v17.0 additions)](#27-file-system-artifact-map)
28. [Schema Migration Rules (v17.0 additions)](#28-schema-migration-rules)
29. [Interface Contracts Between Components (v17.0 additions)](#29-interface-contracts-between-components)
30. [v17.0 New Schemas](#30-v170-new-schemas)

---

## 1. Overview

This document is the authoritative definition of every data schema, interface contract, and file-system artifact used in the VGA v17.0 pipeline. **Every field defined here is mandatory unless explicitly marked `Optional`. Every schema carries `schema_version = "v6.0"`.**

**Retained from v16.0 (unchanged):** All §1–§24 from v16.0. All v16.0 schemas — ScriptSchema, SegmentPlanSchema, IdentityDesignSchema, CLIPValidationRecord, LoRAUsageRecord, ImageRefinementRecord, ContinuityReport, AudioTimingRecord, LipSyncValidationRecord, AudioMixReport, HRGDecisionRecord, HRGDecisionRequest, HRGDecisionResponse, JobSubmissionRequest — ALL retained.

**New in v17.0:**
- §30 v17.0 New Schemas:
  - `CompositionPlanSchema` — SceneCompositionAgent output; all 6 fields mandatory
  - `TemporalBufferRecord` — logged after each buffer update; frame_count always 5
  - `MotionStateRecord` — logged per segment; velocity/direction/magnitude
  - `IdentityStateRecord` — logged per stage transition; cumulative drift tracking
  - `AudioQualityRecord` — SNR + peak_db + clipping status per scene
  - `CrossModalAlignmentRecord` — video ↔ audio duration per segment
  - `SVIGenerationRecord` — cfg, steps, lora_schedule per SVI segment
  - `ContinuityReport` gains: `identity_per_segment: List[float]` (NEW v17.0 field)
  - `PipelineReport` gains: `composition_plan_summary`, `temporal_engine_health`, `identity_state_final`, `audio_quality_summary`
- HRG schemas updated to cover 11 checkpoints (HRG-1 through HRG-11)
- Schema version advances to `"v6.0"` for all new v17.0 artifacts
- **v17.2 additions:** SVIGenerationRecord gains SVI generation lineage fields (`previous_segment_id`, `source_buffer_frame_ids`, `buffer_timestamp_range`, `generation_mode`); TemporalBufferRecord gains `resolution` and `device_at_log_time` enforcement; Cross-modal validation unified contract added

---

## 2. Schema Version

```python
SCHEMA_VERSION: str = "v6.0"    # ALL artifacts written by v17.0 agents carry this string
```

---

## 3–19: All v15.0 Schemas Retained

Sections §3 through §19 are retained without modification from v15.0.

---

## 20. Full Pipeline Stage Schemas (retained from v16.0 with v17.0 updates)

All v16.0 §20 schemas retained. v17.0 updates:

```python
# ScriptSchema: schema_version updated to v6.0 (field default)
class ScriptSchema(BaseModel):
    scene_id: str
    duration: int                      # MUST be 10–30 seconds
    dialogue: str
    emotion: str
    motion_intent: str
    beats: List[str]                   # MUST be non-empty
    schema_version: str = "v6.0"      # updated from v5.2

    @validator("duration")
    def validate_duration(cls, v):
        assert 10 <= v <= 30, f"duration {v} must be 10–30"
        return v

    @validator("beats")
    def validate_beats(cls, v):
        assert len(v) > 0, "beats must be non-empty"
        return v

# SegmentPlanSchema: schema_version updated to v6.0
class SegmentPlanSchema(BaseModel):
    scene_id: str
    segments: List[SegmentEntry]
    total_duration: int
    schema_version: str = "v6.0"

# IdentityDesignSchema: schema_version updated to v6.0
class IdentityDesignSchema(BaseModel):
    character_identity: CharacterIdentity
    environment_description: str
    reference_strategy: str
    lighting_setup: Optional[str] = None
    color_palette: Optional[str] = None
    schema_version: str = "v6.0"
```

---

## 21. Image Pipeline Schemas (retained from v16.0, schema_version v6.0)

All v16.0 §21 schemas retained. schema_version updated to `"v6.0"` on all.

---

## 22. Audio Pipeline Schemas (retained from v16.0 with v17.0 additions)

All v16.0 §22 schemas retained. v17.0 updates to `ContinuityReport`:

```python
class ContinuityReport(BaseModel):
    """Output of ContinuityValidationAgent (Stage S-10). v17.0 adds identity_per_segment."""
    scene_id: str
    continuity_score: float
    threshold: float = 0.90
    passed: bool
    motion_smoothness: float
    lighting_consistency: float
    identity_consistency: float
    identity_per_segment: List[float]  # NEW v17.0: CLIP score per segment keyframe
    action_taken: str
    timestamp: str
    schema_version: str = "v6.0"
```

---

## 23. HRG Decision Schemas (updated for 11 checkpoints)

All v16.0 HRG schemas retained. v17.0 updates:

```python
class HRGCheckpointState(BaseModel):
    """State of a single HRG checkpoint. v17.0: updated to cover HRG-1 through HRG-11."""
    checkpoint: str                    # "HRG-1" | "HRG-2" | ... | "HRG-11"
    status: str                        # "pending" | "awaiting_human" | "approved" | "timed_out"
    display_data: dict                 # checkpoint-specific display data
    schema_version: str = "v6.0"

    @validator("checkpoint")
    def validate_checkpoint(cls, v):
        valid = {f"HRG-{i}" for i in range(1, 12)}  # HRG-1 through HRG-11
        assert v in valid, f"Invalid checkpoint {v}; must be HRG-1 through HRG-11"
        return v


class HRGDecisionRecord(BaseModel):
    """Logged for every human decision at HRG checkpoints. v17.0: 11 checkpoints."""
    checkpoint: str                    # "HRG-1" through "HRG-11"
    user: str
    action: str                        # "approved" | "modified" | "upload_replacement" | "trigger_regeneration"
    timestamp: str
    payload: Optional[dict] = None
    schema_version: str = "v6.0"

# HRG-2 specific display data (NEW v17.0)
class HRG2DisplayData(BaseModel):
    """Display data for HRG-2: Scene/Segment Plan Review."""
    scenes: List[dict]                 # scene_id, duration, beats_count
    segments_per_scene: dict           # scene_id → [{start, end, duration}]
    total_scenes: int
    total_segments: int
    total_duration_s: float
    schema_version: str = "v6.0"

# HRG-4 specific display data (NEW v17.0)
class HRG4DisplayData(BaseModel):
    """Display data for HRG-4: Scene Composition Review."""
    scene_id: str
    camera_angle: str
    camera_motion: str
    character_positions: List[dict]
    focus_subject: str
    lighting_style: str
    motion_vector: str
    validation_passed: bool
    schema_version: str = "v6.0"

# HRG-8 specific display data (updated from v16.0)
class HRG8DisplayData(BaseModel):
    """Display data for HRG-8: Motion QA Review. v17.0 adds identity_per_segment."""
    scene_id: str
    video_segment_paths: List[str]
    continuity_score: float
    motion_smoothness: float
    lighting_consistency: float
    identity_consistency: float
    identity_per_segment: List[float]  # NEW v17.0
    schema_version: str = "v6.0"

# HRG-10 specific display data (updated from v16.0)
class HRG10DisplayData(BaseModel):
    """Display data for HRG-10: Lip Sync QA Review. v17.0 adds identity_delta."""
    scene_id: str
    synced_video_paths: List[str]
    phoneme_alignment_scores: List[float]
    identity_delta_per_segment: List[float]  # NEW v17.0 explicit
    all_passed: bool
    schema_version: str = "v6.0"

# HRG-11 specific display data (updated from v16.0 HRG-9)
class HRG11DisplayData(BaseModel):
    """Display data for HRG-11: Final Audio QA Review. v17.0 adds SNR/clipping."""
    scene_id: str
    final_video_path: str
    mixed_audio_path: str
    dialogue_db: float
    ambient_db: float
    music_db: float
    snr_db: float                      # NEW v17.0 (RULE-99)
    peak_db: float                     # NEW v17.0 (RULE-99)
    clipping_detected: bool            # NEW v17.0
    snr_passed: bool                   # NEW v17.0
    schema_version: str = "v6.0"
```

---

## 24. API Request/Response Schemas (v17.0 additions)

All v16.0 §24 schemas retained. v17.0 additions:

```python
class CompositionPlanUpdateRequest(BaseModel):
    """Request to update CompositionPlan at HRG-4."""
    scene_id: str
    camera_angle: Optional[str] = None
    camera_motion: Optional[str] = None
    character_positions: Optional[List[dict]] = None
    focus_subject: Optional[str] = None
    lighting_style: Optional[str] = None
    motion_vector: Optional[str] = None
    schema_version: str = "v6.0"

class TemporalBufferStatusResponse(BaseModel):
    """Response for GET /jobs/{id}/temporal/buffer."""
    job_id: str
    scene_id: str
    buffer_ready: bool
    frame_count: int                   # should be 5 when ready
    last_updated_at: str
    schema_version: str = "v6.0"

class IdentityStateResponse(BaseModel):
    """Response for GET /jobs/{id}/identity/state."""
    job_id: str
    scene_id: str
    drift_score: float
    cumulative_drift: float
    drift_history: List[float]
    threshold: float = 0.15
    threshold_exceeded: bool
    schema_version: str = "v6.0"

class AudioValidationResponse(BaseModel):
    """Response for GET /jobs/{id}/audio/validation."""
    job_id: str
    scene_id: str
    snr_db: float
    peak_db: float
    clipping_detected: bool
    snr_passed: bool
    clipping_passed: bool
    schema_version: str = "v6.0"
```

---

## 25. Regeneration Request Schema (unchanged from v16.0, schema_version v6.0)

---

## 26. Pipeline Report Schema (v17.0 additions)

All v16.0 PipelineReport fields retained. v17.0 additions:

```python
class PipelineReport(BaseModel):
    """Comprehensive pipeline report generated by QualityAgent (S-16c). v17.0 additions."""

    # ── All v16.0 fields retained ─────────────────────────────────────────
    job_id: str
    scene_id: str
    total_duration_s: float
    stage_timings: dict
    clip_score_timeline: List[float]
    continuity_scores: List[float]
    timing_errors: List[float]
    hrg_decisions_summary: List[dict]
    sla_compliance_summary: dict
    adaptive_state_summary: dict
    schema_version: str = "v6.0"

    # ── NEW v17.0 fields ──────────────────────────────────────────────────
    composition_plan_summary: dict           # camera_angle, motion_vector, lighting_style per scene
    temporal_engine_health: dict             # buffer_updates, segments_generated, retries_per_segment
    motion_state_summary: List[dict]         # direction + magnitude per segment
    identity_state_final: dict               # final drift_score, cumulative_drift, history
    audio_quality_summary: dict              # snr_db, peak_db, clipping_detected per scene
    cross_modal_alignment_summary: List[dict] # alignment_error per segment
    identity_per_segment_video: List[float]  # CLIP scores per video segment (RULE-89)
    identity_delta_per_segment_sync: List[float]  # delta per lip-synced segment (RULE-97)
    hrg_checkpoint_count: int = 11           # updated from 9 to 11
    rule_compliance: dict                    # RULE-86 through RULE-99 compliance status
```

---

## 27. File System Artifact Map (v17.0 additions)

All v16.0 file system artifacts retained. v17.0 additions:

```
/workspace/
├── composition/                              ← NEW v17.0
│   └── {job_id}/
│       └── {scene_id}/
│           └── composition_plan_{scene_id}.json   ← SceneCompositionAgent output
│
├── temporal/                                 ← NEW v17.0
│   └── {job_id}/
│       └── {scene_id}/
│           ├── temporal_buffer_log.json      ← TemporalBufferRecord per segment
│           ├── motion_state_log.json         ← MotionStateRecord per segment
│           └── svi_generation_log.json       ← SVIGenerationRecord per segment
│
├── identity/                                 ← NEW v17.0
│   └── {job_id}/
│       └── identity_state_log.json           ← IdentityStateRecord per stage transition
│
├── validation/
│   └── {job_id}/
│       ├── identity_validation_log.json      ← CLIPValidationRecord (all phases, v17.0 extended)
│       ├── continuity_report_{scene_id}.json ← ContinuityReport (now includes identity_per_segment)
│       ├── lora_usage_log.json               ← LoRAUsageRecord
│       ├── audio_quality_log.json            ← AudioQualityRecord (NEW v17.0)
│       └── cross_modal_alignment_log.json    ← CrossModalAlignmentRecord (NEW v17.0)
│
├── images/                                   ← unchanged from v16.0
│   └── {job_id}/{scene_id}/
│       ├── base/
│       ├── angles/
│       ├── composed/
│       ├── expanded/
│       └── refined/
│
├── video/                                    ← NEW v17.0 (separate from synced)
│   └── {job_id}/
│       └── {scene_id}/
│           ├── segment_001.mp4               ← Wan2.2 Segment_1
│           ├── segment_002.mp4               ← SVI Segment_2
│           └── segment_N.mp4                 ← SVI Segment_N
│
├── audio/                                    ← unchanged from v16.0
│   └── {job_id}/{scene_id}/
│       ├── dialogue/
│       ├── ambient/
│       ├── music/
│       └── mixed/
│
├── synced/                                   ← lip-synced video
│   └── {job_id}/{scene_id}/
│       └── synced_{seg_id}.mp4
│
├── hrg_logs/
│   └── {job_id}/
│       └── hrg_log.json                      ← all 11 HRG decisions (v17.0: 11 checkpoints)
│
└── output/
    └── {job_id}/{scene_id}/
        ├── final_video.mp4
        ├── pipeline_report.json              ← includes v17.0 fields
        ├── sla_summary.json
        └── adaptive_state.json
```

---

## 28. Schema Migration Rules (v17.0 additions)

All v16.0 migration rules retained. v17.0 additions:

```python
def _migrate_v5_2_to_v6_0(artifact: dict) -> dict:
    """
    Migrate v5.2 artifacts to v6.0.
    Called by schema_migrations.py for any artifact with schema_version < v6.0.
    """
    # Update schema version
    artifact["schema_version"] = "v6.0"

    # ContinuityReport: add identity_per_segment if missing
    if artifact.get("_type") == "ContinuityReport":
        if "identity_per_segment" not in artifact:
            artifact["identity_per_segment"] = []

    # HRGDecisionRecord: validate checkpoint is HRG-1 through HRG-11
    if artifact.get("_type") == "HRGDecisionRecord":
        checkpoint = artifact.get("checkpoint", "")
        old_to_new = {
            # v16.0 had 9 checkpoints; v17.0 re-numbers HRG-2 through HRG-9
            # due to insertion of HRG-2 (Scene/Segment Plan) and HRG-4 (Composition)
            "HRG-2": "HRG-3",  # old Identity Review → new HRG-3
            "HRG-3": "HRG-5",  # old Base Image → new HRG-5
            "HRG-4": "HRG-6",  # old Composed Images → new HRG-6
            "HRG-5": "HRG-7",  # old Refined Images → new HRG-7
            "HRG-6": "HRG-8",  # old Motion QA → new HRG-8
            "HRG-7": "HRG-9",  # old Voice QA → new HRG-9
            "HRG-8": "HRG-10", # old Lip Sync QA → new HRG-10
            "HRG-9": "HRG-11", # old Final QA → new HRG-11
        }
        if checkpoint in old_to_new:
            artifact["checkpoint"] = old_to_new[checkpoint]

    return artifact
```

---

## 29. Interface Contracts Between Components (v17.0 additions)

All v16.0 interface contracts retained. v17.0 additions:

```
SceneCompositionAgent → ImagePipeline:
  Contract: SceneCompositionAgent writes composition_plan_{scene_id}.json to
            /workspace/composition/{job_id}/{scene_id}/
  Consumer: BaseImageAgent reads it; stores in context.camera_state + context.lighting_state
  Violation: BaseImageAgent must raise CompositionPlanMissingError if not present (RULE-88)

TemporalBufferManager → SVIWrapper:
  Contract: TemporalBufferManager.encode() returns Tensor of shape (5, C', H', W')
  Consumer: SVIWrapper receives as init_latents; must have shape[0] == 5
  Violation: SVIWrapper must raise AutoregressiveViolationError if shape[0] != 5

SVIScheduler → SVIWrapper:
  Contract: SVIScheduler.apply_lora(timestep) returns float ∈ {0.4, 0.5, 0.6}
  Consumer: SVIWrapper applies returned weight at each denoising step
  Violation: Static weight assignment bypassing SVIScheduler is FORBIDDEN

MotionStateTracker → TemporalEngine:
  Contract: MotionStateTracker.estimate(buffer.frames) returns MotionState
  Consumer: TemporalEngine uses motion_state.direction + magnitude in prompt/params
  Timing: MUST be called before each SVI generation; result NOT cached across segments

IdentityStateTracker → All Agents (cross-phase):
  Contract: IdentityStateTracker.update(IS, embedding) returns new IdentityState or raises
  Consumer: Every agent that calls CLIPValidator also calls IdentityStateTracker.update
  Shared state: IdentityState is passed via ImmutableContext; update via context.evolve()

AudioQualityValidator → AudioMixingAgent:
  Contract: AudioQualityValidator.validate(mixed) returns AudioQualityRecord
  Consumer: AudioMixingAgent MUST call before writing to storage
  Blocking: If snr < MIN_SNR_DB or peak > 0.0, agent must re-mix before storage write

CLIPValidator (cross-phase):
  Contract: ALL calls use char_identity_ref from ImmutableContext (frozen at S-07)
  Verification: ref_hash checked at every call to detect mutation
  Violation: IdentityReferenceCorruptionError if hash mismatch
```

---

## 30. v17.0 New Schemas

```python
from pydantic import BaseModel, validator
from typing import List, Optional, Dict
import torch

# ── Scene Composition Schema ──────────────────────────────────────────────

class CompositionPlanSchema(BaseModel):
    """
    Mandatory output of SceneCompositionAgent (S-04). RULE-88.
    All 6 fields are required; none may be None or empty.
    """
    scene_id: str
    camera_angle: str              # "close-up" | "medium shot" | "wide shot" | ...
    camera_motion: str             # "static" | "slow dolly forward" | "pan left" | ...
    character_positions: List[Dict]  # [{character_id: str, position: str, facing: str}]
    focus_subject: str             # primary visual focus element
    lighting_style: str            # "soft natural" | "low-key dramatic" | ...
    motion_vector: str             # "stationary" | "forward_slow" | "right_medium" | ...
    schema_version: str = "v6.0"

    @validator("camera_angle", "camera_motion", "focus_subject", "lighting_style", "motion_vector")
    def validate_nonempty(cls, v):
        assert v and len(v.strip()) > 0, f"Field must not be empty: {v!r}"
        return v

    @validator("character_positions")
    def validate_positions(cls, v):
        assert len(v) >= 1, "At least one character position required"
        return v


# ── Temporal Engine Schemas ───────────────────────────────────────────────

class TemporalBufferRecord(BaseModel):
    """
    Logged after each TemporalBuffer update. OR-034.
    frame_count MUST always equal TEMPORAL_BUFFER_SIZE (5).

    v17.1 HARD CONTRACT:
      - frame_count is STRICTLY enforced via validator; any value != 5 raises ValidationError
      - resolution MUST be provided; all 5 frames share the same (H, W)
      - timestamp_sequence MUST have exactly 5 entries (one per frame)
      - device_at_log_time MUST be "cpu" — buffer is always CPU-resident between steps
    """
    record_id: str
    segment_id: str
    scene_id: str
    frame_count: int               # MUST be 5 — enforced by validator below
    resolution: Tuple[int, int]    # (H, W) — identical across all 5 frames
    timestamp_sequence: List[float]  # MUST have exactly 5 entries
    buffer_initialized: bool       # True if this is the init record (from Segment_1)
    device_at_log_time: str        # MUST be "cpu" (buffer is CPU-resident)
    timestamp: str
    schema_version: str = "v6.0"

    @validator("frame_count")
    def validate_frame_count(cls, v):
        if v != 5:
            raise ValueError(
                f"TemporalBufferRecord.frame_count MUST be 5 (TEMPORAL_BUFFER_SIZE). "
                f"Got {v}. This indicates a TemporalBuffer invariant violation. "
                f"Pipeline MUST halt. [RULE-86, v17.1 Constraint C-1]"
            )
        return v

    @validator("timestamp_sequence")
    def validate_timestamp_sequence(cls, v):
        if len(v) != 5:
            raise ValueError(
                f"timestamp_sequence must have exactly 5 entries; got {len(v)}. "
                f"Must match frame_count=5. [RULE-86]"
            )
        return v

    @validator("device_at_log_time")
    def validate_device(cls, v):
        if v != "cpu":
            raise ValueError(
                f"TemporalBuffer must be CPU-resident between segments. "
                f"Got device='{v}'. Buffer must be transferred to CPU before logging. "
                f"[v17.1 Buffer Device Rule]"
            )
        return v


class MotionStateRecord(BaseModel):
    """
    Logged per segment by MotionStateTracker. OR-035.
    """
    record_id: str
    segment_id: str
    scene_id: str
    velocity_x: float
    velocity_y: float
    velocity_magnitude: float
    direction: str                 # "stationary" | "forward" | "backward" | "left" | "right" | "diagonal"
    is_stationary: bool
    timestamp: str
    schema_version: str = "v6.0"


class SVIGenerationRecord(BaseModel):
    """
    Logged per SVI segment generation by TemporalEngine.
    v17.2: Adds generation lineage traceability (previous_segment_id,
           source_buffer_frame_ids, buffer_timestamp_range, generation_mode).
    SVI generation MUST be traceable to a TemporalBuffer.
    If source_buffer_frame_ids is missing → INVALID GENERATION.
    """
    record_id: str
    segment_id: str
    scene_id: str
    previous_segment_id: str      # NEW v17.2: ID of Segment_n that produced the buffer
    source_buffer_frame_ids: List[str]  # NEW v17.2: MUST have exactly 5 frame IDs
    buffer_timestamp_range: Tuple[float, float]  # NEW v17.2: (oldest_ts, newest_ts) from buffer
    generation_mode: str          # NEW v17.2: MUST be "autoregressive"
    cfg: float                     # MUST be ∈ [5.0, 6.0]
    steps: int                     # STEPS_STANDARD or STEPS_CRITICAL
    is_critical: bool
    lora_schedule: List[float]     # [0.6, 0.5, 0.4] per phase
    attempt: int                   # retry attempt (0 = first)
    clip_score: float              # identity score for this segment
    clip_passed: bool
    per_segment_continuity: float  # boundary continuity with previous segment
    timestamp: str
    schema_version: str = "v6.0"

    @validator("cfg")
    def validate_cfg(cls, v):
        assert 5.0 <= v <= 6.0, f"SVI CFG {v} outside [5.0, 6.0]; RULE-86"
        return v

    @validator("source_buffer_frame_ids")
    def validate_frame_ids(cls, v):
        if len(v) != 5:
            raise ValueError(
                f"source_buffer_frame_ids MUST contain exactly 5 frame IDs; "
                f"got {len(v)}. SVI generation must be traceable to a 5-frame buffer. "
                f"[v17.2 SVI Lineage Requirement]"
            )
        return v

    @validator("generation_mode")
    def validate_generation_mode(cls, v):
        if v != "autoregressive":
            raise ValueError(
                f"generation_mode MUST be 'autoregressive'; got '{v}'. "
                f"Non-autoregressive SVI generation is FORBIDDEN. [v17.2]"
            )
        return v


# ── Identity State Schemas ────────────────────────────────────────────────

class IdentityStateRecord(BaseModel):
    """
    Logged per stage transition by IdentityStateTracker. OR-036.
    """
    record_id: str
    stage_id: str                  # "S-05" | "S-06A" | "S-06B" | "S-06C" | "S-07" | "S-09_seg_N" | "S-12"
    scene_id: str
    delta: float                   # per-stage identity drift
    drift_score: float             # cumulative drift after this update
    drift_history: List[float]     # all per-stage deltas so far
    threshold_exceeded: bool       # True if drift_score > IDENTITY_CUMULATIVE_DRIFT_THRESHOLD
    timestamp: str
    schema_version: str = "v6.0"


# ── Audio Quality Schemas ─────────────────────────────────────────────────

class AudioQualityRecord(BaseModel):
    """
    Logged after AudioMixingAgent completion. OR-037. RULE-99.
    """
    record_id: str
    scene_id: str
    snr_db: float                  # Signal-to-Noise Ratio in dB; MUST be ≥ MIN_SNR_DB (10.0)
    peak_db: float                 # Peak level in dBFS; MUST be ≤ 0.0
    clipping_detected: bool        # True if peak_db > 0.0
    snr_passed: bool               # True if snr_db >= MIN_SNR_DB
    clipping_passed: bool          # True if peak_db <= 0.0
    remix_count: int               # number of re-mix attempts before passing (0 = passed on first)
    normalization_applied: bool    # True if peak was normalized
    timestamp: str
    schema_version: str = "v6.0"


# ── Cross-Modal Alignment Schemas ─────────────────────────────────────────

class CrossModalAlignmentRecord(BaseModel):
    """
    Logged per segment by CrossModalAlignmentValidator. RULE-96.
    """
    record_id: str
    scene_id: str
    segment_id: str
    video_duration_s: float
    audio_duration_s: float
    alignment_error_s: float       # = abs(audio_duration_s - video_duration_s)
    within_tolerance: bool         # True if alignment_error_s <= TIMING_TOLERANCE_S
    tolerance_s: float = 0.10
    timestamp: str
    schema_version: str = "v6.0"


class CrossModalAlignmentReport(BaseModel):
    """
    Aggregated cross-modal alignment report for a full scene.
    """
    scene_id: str
    records: List[CrossModalAlignmentRecord]
    all_passed: bool
    max_error_s: float
    mean_error_s: float
    total_video_duration_s: float
    total_audio_duration_s: float
    total_error_s: float
    schema_version: str = "v6.0"


# ── StageResult v17.0 additions ───────────────────────────────────────────

# StageResult gains these new optional fields (from v16.0 base):
#   composition_plan: Optional[CompositionPlanSchema] = None
#   temporal_buffer_record: Optional[TemporalBufferRecord] = None
#   motion_state_record: Optional[MotionStateRecord] = None
#   svi_generation_record: Optional[SVIGenerationRecord] = None
#   identity_state_record: Optional[IdentityStateRecord] = None
#   audio_quality_record: Optional[AudioQualityRecord] = None
#   cross_modal_alignment_record: Optional[CrossModalAlignmentRecord] = None

# ── QualityReport v17.0 additions ─────────────────────────────────────────

# QualityReport gains these new fields (from v16.0 base):
#   composition_plan_summary: Optional[dict] = None
#   temporal_engine_health: Optional[dict] = None
#   motion_state_summary: Optional[List[dict]] = None
#   identity_state_final: Optional[dict] = None
#   audio_quality_summary: Optional[dict] = None
#   cross_modal_alignment_summary: Optional[dict] = None
#   identity_per_segment_video: Optional[List[float]] = None
#   identity_delta_per_segment_sync: Optional[List[float]] = None
#   rule_compliance: Optional[dict] = None   # RULE-86 through RULE-99
```
