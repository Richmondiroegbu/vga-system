# VGA File Responsibility Specification
**Project:** Video Generation Automation (VGA) — Motivation System
**Version:** 17.2.0
**Status:** Authoritative Reference
**Audience:** Claude Code Agent, All Engineers

---

## Overview

This document assigns a **single, authoritative responsibility** to every source file in the VGA codebase. **One file = one responsibility.** No file may exceed its responsibility. No responsibility may be split across files except where explicitly documented as a coordinated pair.

**Retained from v16.0 (unchanged):** All §1 through §9 ownership rules and file responsibilities. Every file listed in v16.0 retains its exact responsibility in v17.0.

**New in v17.0:** §10 through §19 — new files introduced by the v17.0 architecture pillars (SceneCompositionAgent, TemporalEngine subsystem, IdentityStateTracker, AudioQualityValidator, CrossModalAlignmentValidator, updated HRG components, new API routes, new UI panels).

**File additions summary (v17.0):**
- 1 new agent: `scene_composition_agent.py`
- 1 new agent: `video_segment_generator.py`
- 5 new temporal files: `temporal_engine.py`, `temporal_buffer_manager.py`, `svi_scheduler.py`, `motion_state_tracker.py`, `temporal_retry_controller.py`
- 1 new identity file: `identity_state_tracker.py`
- 2 new validation files: `audio_quality_validator.py`, `cross_modal_alignment_validator.py`
- 1 new validation file: `composition_validator.py`
- 4 new API route files: `temporal.py`, `identity.py`, `audio.py`, `composition.py`
- 2 new UI component files: `temporal_engine_panel.py`; new HRG panels `hrg_2` and `hrg_4`; updated `hrg_8`, `hrg_10`, `hrg_11`
- 1 new prompt file: `composition_prompts.py`
- 1 updated: `settings.py` (v17.0 constants)
- 1 updated: `schemas.py` (v17.0 schemas)
- 1 updated: `exceptions.py` (v17.0 exception types)
- 1 updated: `hrg_controller.py` (11 checkpoints)
- 1 updated: `master_orchestrator.py` (execute_stage contract)
- 1 updated: `immutable_context.py` (5-dimensional context)
- 1 new snapshot dir: `snapshots/v17_candidate/`

---

## 1–9: All v16.0 File Responsibilities Retained

All sections §1–§9 from v16.0 are retained without modification. Every file listed there has the same single responsibility in v17.0.

The only changes to v16.0 files are:
- `config/settings.py` — **extended** (v17.0 constants appended; all v16.0 constants unchanged)
- `models/schemas.py` — **extended** (v17.0 schemas added; all v16.0 schemas unchanged)
- `core/exceptions.py` — **extended** (v17.0 exceptions added; all v16.0 exceptions unchanged)
- `core/hrg_controller.py` — **updated** (11 checkpoints; v16.0 9-checkpoint logic retained and extended)
- `core/master_orchestrator.py` — **updated** (execute_stage() contract added as mandatory wrapper)
- `state/immutable_context.py` — **updated** (5-dimensional context: identity, motion, camera, lighting, temporal)
- `state/context_factory.py` — **updated** (creates 5-dim initial context at S-02)
- `agents/base_image_agent.py` — **updated** (accepts CompositionPlan; gates on RULE-88)
- `agents/scene_expansion_agent.py` — **updated** (CompositionPlan fully bound; RULE-88 enforced)
- `agents/image_refinement_agent.py` — **updated** (char_identity_ref FREEZE logic)
- `agents/continuity_validation_agent.py` — **updated** (`identity_per_segment` field in output)
- `agents/lip_sync_agent.py` — **updated** (IdentityStateTracker.update per segment)
- `agents/audio_mixing_agent.py` — **updated** (AudioQualityValidator + CrossModal calls)
- `agents/quality_agent.py` — **updated** (v17.0 fields in PipelineReport)
- `models/wrappers/svi_wrapper.py` — **updated** (multi-frame latents; SVIScheduler integration)
- `models/wrappers/wan_wrapper.py` — **updated** (CompositionPlan motion_params accepted)
- `bootstrap.py` — **updated** (v17.0 singletons 6Z-p through 6Z-z)
- All 11 `ui/components/hrg_panels/` — **updated** (renumbered; new panels added)

---

## 10. New Agent Files (v17.0)

### 10.1 `vga/agents/scene_composition_agent.py`

**Single responsibility:** Translate scene narrative fields (dialogue, emotion, motion_intent, characters, environment) into a validated `CompositionPlanSchema` with all 6 mandatory fields.

**Owns:**
- Calling `qwen_wrapper.generate_structured(scene_data, CompositionPlanSchema)`
- Retrying up to `COMPOSITION_MAX_RETRIES` times on schema validation failure
- Calling `CompositionPlanValidator.validate(plan)` before returning
- Writing `composition_plan_{scene_id}.json` to `/workspace/composition/{job_id}/{scene_id}/`
- Tracing `composition_plan_created` event

**Does NOT own:**
- HRG display (owned by `hrg_4_composition.py`)
- Image generation (owned by `base_image_agent.py`)
- Video generation (owned by `temporal_engine.py`)
- Storing CompositionPlan in context (owned by `execute_stage()` → `context.evolve()`)

**Raises:** `CompositionPlanValidationError` after all retries exhausted

---

### 10.2 `vga/agents/video_segment_generator.py`

**Single responsibility:** Generate Segment_1 using Wan2.2 with a refined image and CompositionPlan, then initialize the TemporalBuffer from the output.

**Owns:**
- Loading refined image from `images/{job_id}/{scene_id}/refined/`
- Calling `wan_wrapper.generate()` with CompositionPlan motion params
- Running CLIPValidator on Segment_1 keyframe (RULE-89)
- Calling `TemporalBufferManager.init(segment_1)` to create initial buffer
- Asserting `buffer.frames.shape[0] == TEMPORAL_BUFFER_SIZE` (RULE-86)
- Writing `video/{job_id}/{scene_id}/segment_001.mp4`
- Tracing `segment_1_generated`, `temporal_buffer_initialized` events

**Does NOT own:**
- Segments 2..N (owned by `temporal_engine.py`)
- CompositionPlan creation (owned by `scene_composition_agent.py`)
- Buffer update after Segment_1 (owned by `temporal_engine.py`)

**Raises:** `TemporalBufferError` if buffer initialization fails

---

## 11. New Temporal Engine Files (v17.0)

### 11.0 TEMPORAL ENGINE AUTHORITY (NEW v17.2 — GLOBAL RULE)

```
TEMPORAL ENGINE AUTHORITY v17.2:

TemporalEngine (`vga/temporal/temporal_engine.py`) is the ONLY component
authorized to perform the following operations:

  EXCLUSIVE OPERATIONS (TemporalEngine only):
    1. Control segment iteration (the autoregressive for loop)
    2. Update TemporalBuffer after each segment generation
    3. Invoke SVI Pro 2 for temporal segment generation
    4. Manage the complete autoregressive generation flow

  FORBIDDEN for ALL OTHER COMPONENTS:
    - Generate Segment_{n+1} outside TemporalEngine
    - Modify TemporalBuffer state directly (outside temporal_buffer_manager.py called by TemporalEngine)
    - Bypass TemporalEngine to call SVI directly
    - Inject externally generated segments into the autoregressive loop
    - Control the segment iteration order or count

  ENFORCEMENT MECHANISM:
    TemporalAuthorityGuard (vga/temporal/temporal_authority_guard.py) — NEW v17.2
    ArchitectureGuard — existing runtime guard
    Any violation raises ArchitectureGuardViolationError → pipeline halts
```

### 11.1 `vga/temporal/temporal_engine.py`

**Single responsibility:** Execute the autoregressive SVI loop for Segments 2 through N, managing buffer updates, motion state, identity validation, and context evolution per segment.

**Owns:**
- The outer autoregressive loop (`for n in range(2, N+1)`)
- Calling `TemporalBufferManager.encode()` → multi-frame latents (RULE-87)
- Calling `SVIScheduler` for noise-aware LoRA (RULE-86)
- Calling `MotionStateTracker.estimate()` per segment
- Calling `CLIPValidator.score()` per segment keyframe (RULE-89)
- Calling `ContinuityValidator.score_segment()` per segment boundary
- Calling `IdentityStateTracker.update()` per segment
- Calling `TemporalBufferManager.update()` after each successful segment
- Calling `context.evolve()` after each segment
- Logging `TemporalBufferRecord`, `MotionStateRecord`, `SVIGenerationRecord`
- Delegating to `TemporalRetryController.adjust()` on retry

**Does NOT own:**
- Segment_1 generation (owned by `video_segment_generator.py`)
- Buffer initialization (owned by `video_segment_generator.py`)
- SVI model loading/unloading (owned by `model_manager.py`)
- HRG display (owned by `hrg_8_motion_qa.py`)

**Raises:** `TemporalSegmentFailureError`, `TemporalBufferError`, `AutoregressiveViolationError`, `SVICFGViolationError`

---

### 11.2 `vga/temporal/temporal_buffer_manager.py`

**Single responsibility:** Manage the `TemporalBuffer` lifecycle — initialization, rolling update, multi-frame latent encoding, normalization, and strict size enforcement with CPU↔GPU device management. This is the SOLE owner of all temporal buffer state operations.

**Owns (v17.1 — expanded and authoritative):**

Frame Extraction:
- `_extract_last_5_frames(video_tensor: Tensor) → Tensor` — extracts exactly 5 frames from end of segment; raises `TemporalBufferError` if segment has fewer than 5 frames
- Frame ordering: frames[0]=oldest (t-4), frames[4]=newest (t); ordered from oldest to most recent

Normalization:
- `_normalize_frames(frames: Tensor) → Tensor` — normalizes all 5 frames to [0.0, 1.0] range using the same pipeline (mean/std or min-max; consistent across ALL frames and ALL calls)
- Normalization parameters MUST be fixed constants (not computed per-call)

Ordering Guarantee:
- `_assert_temporal_order(timestamps: List[float]) → None` — validates timestamps are strictly increasing; raises `TemporalBufferError` on out-of-order frames

Strict Size Enforcement:
- EVERY method that creates or returns a `TemporalBuffer` MUST assert `frames.shape[0] == TEMPORAL_BUFFER_SIZE` before returning
- `_assert_buffer_size(buffer: TemporalBuffer) → None` — shared internal validator; raises `TemporalBufferError(frame_count, required=5)` immediately on violation

Device Management (CPU ↔ GPU):
- Buffer frames are ALWAYS stored as CPU tensors
- `encode(buffer)` is the ONLY method that transfers frames to GPU: `frames.to(device)` inside encode; immediately returned as CPU tensor after encoding
- `_to_cpu(tensor: Tensor) → Tensor` — utility; called at end of encode before returning latents
- FORBIDDEN: any other method moving frames to GPU

Lifecycle Methods:
- `init(segment_1: VideoSegment) → TemporalBuffer` — extract last 5 frames → normalize → create TemporalBuffer (CPU)
- `update(buffer: TemporalBuffer, new_segment: VideoSegment) → TemporalBuffer` — rolling update from new_segment last 5 frames → normalize → new TemporalBuffer (CPU); old buffer discarded
- `encode(buffer: TemporalBuffer) → Tensor` — transfer frames to GPU → encode as latents → transfer latents to CPU → assert latents.shape[0] == 5 → return CPU latent tensor of shape (5, C', H', W')

Logging:
- `temporal_buffer_initialized` — logged after `init()` with frame_count=5
- `temporal_buffer_updated` — logged after `update()` with segment_id and frame_count=5
- `temporal_buffer_encoded` — logged after `encode()` with latent shape

**Does NOT own:**
- Calling SVI (owned by `svi_wrapper.py`)
- LoRA scheduling (owned by `svi_scheduler.py`)
- Motion estimation (owned by `motion_state_tracker.py`)
- Context evolution (owned by `temporal_engine.py`)
- Identity validation (owned by `clip_validator.py`)

**Raises:** `TemporalBufferError(scene_id, frame_count, required=5)` if frame count != 5 at ANY point — init, update, encode, or validation

---

### 11.3 `vga/temporal/svi_scheduler.py`

**Single responsibility:** Compute and return the noise-aware LoRA weight for each SVI diffusion timestep, and gate SVI CFG values to the valid range [5.0, 6.0].

**Owns:**
- `apply_lora(timestep: int) → float` — returns 0.6 / 0.5 / 0.4 based on noise level
- `assert_cfg_valid(cfg: float) → float` — raises `SVICFGViolationError` if out of range
- `get_lora_schedule() → list` — returns `[0.6, 0.5, 0.4]` for logging
- Threshold computation: `threshold_high = int(total_steps * HIGH_NOISE_FRACTION)`

**Does NOT own:**
- Applying the LoRA weight to the model (owned by `svi_wrapper.py`)
- SVI generation (owned by `svi_wrapper.py`)
- Buffer management (owned by `temporal_buffer_manager.py`)

**Raises:** `SVICFGViolationError` on invalid CFG (does NOT silently clamp)

---

### 11.6 `vga/temporal/temporal_authority_guard.py` (NEW v17.2)

**Single responsibility:** Runtime enforcement of TemporalEngine's exclusive authority over segment iteration, buffer updates, and SVI invocation.

**Owns:**
- `assert_authorized(caller_qualname, operation)` — validates caller is TemporalEngine
- `guard_svi_invoke(caller_qualname)` — guards SVI invocation
- `guard_buffer_update(caller_qualname)` — guards TemporalBuffer.update() calls
- `guard_segment_iteration(caller_qualname)` — guards segment loop control

**Does NOT own:**
- Actual SVI invocation (owned by `svi_wrapper.py`)
- Buffer state (owned by `temporal_buffer_manager.py`)
- Loop logic (owned by `temporal_engine.py`)

**Raises:** `ArchitectureGuardViolationError` on unauthorized access

---

### 11.7 `vga/validation/cross_modal_validation_unified.py` (NEW v17.2)

**Single responsibility:** Provide unified cross-modal validation for a single segment, checking Video ↔ Audio ↔ Identity ↔ Temporal consistency in a single call.

**Owns:**
- `validate(...)` — runs all 4 cross-modal checks; returns `CrossModalValidationContract`
- Duration alignment check (RULE-96)
- Phoneme alignment check
- Cross-frame identity similarity check (CLIP cosine sim ≥ 0.97)
- Temporal continuity check

**Does NOT own:**
- Individual validator implementations (owned by respective validator files)
- Storage of validation records (owned by `storage.py`)
- Retry logic (owned by calling agent)

**Returns:** `CrossModalValidationContract` (does NOT raise on failure; caller decides action)

---

### 11.8 `vga/validation/system_certification_validator.py` (NEW v17.2)

**Single responsibility:** Validate that a completed pipeline run meets all 7 v17.2 system certification conditions before the output is considered deployable.

**Owns:**
- `certify(pipeline_report)` — runs all 7 certification checks
- Temporal loop integrity check
- Identity stability check
- Temporal continuity check
- Audio quality check
- Human governance check
- Auditability check
- Validation propagation check

**Does NOT own:**
- Individual metric computation (owned by respective validators)
- PipelineReport creation (owned by `quality_agent.py`)

**Raises:** `SystemCertificationFailureError` if ANY check fails

---

### 11.4 `vga/temporal/motion_state_tracker.py`

**Single responsibility:** Estimate per-segment motion state (velocity, direction, magnitude) from TemporalBuffer frames via optical flow. Log `MotionStateRecord` per segment.

**Owns:**
- `estimate(frames: Tensor) → MotionState` — optical flow across 5 frames
- `_compute_optical_flow(frame_a, frame_b)` — pairwise flow computation
- `_classify_direction(vx, vy, magnitude, stationary)` — directional label
- `log_state(segment_id, scene_id, state)` — write `MotionStateRecord` to storage

**Does NOT own:**
- Buffer management (owned by `temporal_buffer_manager.py`)
- Propagating motion state into context (owned by `temporal_engine.py`)
- Using motion state for prompt building (owned by `temporal_engine._build_temporal_prompt()`)

---

### 11.5 `vga/temporal/temporal_retry_controller.py`

**Single responsibility:** Adjust SVI generation parameters between retry attempts when a segment fails identity or continuity checks.

**Owns:**
- `adjust(attempt: int, clip_score: float, cont_score: float)` — parameter adjustment logic
- `_adjust_steps_up()` — increase steps on attempt 1
- `_strengthen_identity_prompt()` — stronger identity anchoring on attempt 2

**Does NOT own:**
- Retry loop itself (owned by `temporal_engine._generate_segment_with_retry()`)
- Raising failure exception (owned by `temporal_engine.py`)
- Logging retry events (owned by `temporal_engine.py`)

---

## 12. New Identity File (v17.0)

### 12.1 `vga/identity/identity_state_tracker.py`

**Single responsibility:** Track cumulative identity drift (IdentityState) across all pipeline phases — image stages, video segments, and lip-synced segments. Raise `IdentityCumulativeDriftError` when cumulative drift exceeds threshold.

**Owns:**
- `update(char_identity_ref, new_frame, stage_id)` — compute delta; accumulate drift; log record
- `reset()` — clear drift accumulator after successful phase regeneration
- `IdentityState` accumulation: `_drift_score`, `_history`
- Writing `IdentityStateRecord` to storage
- Tracing `identity_state_update` events

**Does NOT own:**
- CLIP encoding (owned by `clip_validator.py`)
- Deciding what to do when drift exceeds threshold (caller decides; this tracker raises)
- Phase regeneration (owned by `regeneration/engine.py`)
- Freezing `char_identity_ref` (owned by `image_refinement_agent.py`)

**Raises:** `IdentityCumulativeDriftError` when `drift_score > IDENTITY_CUMULATIVE_DRIFT_THRESHOLD`

**Called by:**
- `agents/base_image_agent.py` (after S-05)
- `agents/image_edit_agent.py` (after S-06A, S-06B, S-06C)
- `agents/image_refinement_agent.py` (after S-07)
- `temporal/temporal_engine.py` (after each SVI segment)
- `agents/lip_sync_agent.py` (after each lip-synced segment)

---

## 13. New Validation Files (v17.0)

### 13.1 `vga/validation/composition_validator.py`

**Single responsibility:** Validate that a `CompositionPlanSchema` has all 6 mandatory fields present and non-empty. Assert CompositionPlan is present in pipeline context before any image/video generation stage.

**Owns:**
- `validate(plan: CompositionPlanSchema) → bool` — checks all 6 fields (RULE-88)
- `assert_in_context(context)` — used by BaseImageAgent, TemporalEngine as a gate

**Does NOT own:**
- Creating CompositionPlan (owned by `scene_composition_agent.py`)
- Storing CompositionPlan (owned by `storage.py`)
- HRG display (owned by `hrg_4_composition.py`)

**Raises:** `CompositionPlanValidationError` on any violation

**Called by:**
- `agents/scene_composition_agent.py` — after each generation attempt
- `agents/base_image_agent.py` — gate before generation (RULE-88)
- `agents/video_segment_generator.py` — gate before Wan2.2 call (RULE-88)
- `temporal/temporal_engine.py` — gate before each SVI call (RULE-88)

---

### 13.2 `vga/validation/audio_quality_validator.py`

**Single responsibility:** Validate SNR and peak level of mixed audio after AudioMixingAgent. Normalize audio if peak exceeds 0 dBFS. Return `AudioQualityRecord`. Does not raise — caller decides action based on record.

**Owns:**
- `validate(mixed, dialogue, scene_id, job_id) → AudioQualityRecord`
- `normalize(mixed, target_peak_db) → AudioSegment` — gain normalization
- `compute_snr(mixed, dialogue) → float` — dB SNR computation
- `compute_peak_db(audio) → float` — dBFS peak level
- Writing `AudioQualityRecord` to `validation/{job_id}/audio_quality_log.json`

**Does NOT own:**
- Re-mixing audio (owned by `audio_mixing_agent.py`)
- Deciding whether to halt on failure (owned by `audio_mixing_agent.py`)
- HRG display (owned by `hrg_11_final_qa.py`)

**Stateless:** No mutable internal state; pure computation returning record.

**Called by:** `agents/audio_mixing_agent.py` ONLY

---

### 13.3 `vga/validation/cross_modal_alignment_validator.py`

**Single responsibility:** Validate that video and audio segment durations are aligned within TIMING_TOLERANCE_S (±0.10s) per segment. Return `CrossModalAlignmentReport`. Does not raise on tolerance violation — caller decides.

**Owns:**
- `validate_alignment(video_segments, audio_segments, scene_id) → CrossModalAlignmentReport`
- Computing `alignment_error_s = abs(audio_dur - video_dur)` per segment
- Writing `CrossModalAlignmentRecord` per segment to storage
- Aggregating into `CrossModalAlignmentReport`

**Does NOT own:**
- Adjusting durations (owned by `dialogue_agent.py` / `temporal_engine.py`)
- HRG display (owned by `hrg_11_final_qa.py`)

**Stateless:** No mutable internal state.

**Called by:** `agents/audio_mixing_agent.py` ONLY

---

## 14. New API Route Files (v17.0)

### 14.1 `vga/api/routes/temporal.py`

**Single responsibility:** Expose HTTP endpoints for temporal engine status (buffer health, motion state, SVI generation metrics).

**Endpoints owned:**
- `GET /jobs/{job_id}/temporal/buffer` → `TemporalBufferStatusResponse`
- `GET /jobs/{job_id}/temporal/motion` → `MotionStateRecord` (latest)
- `GET /jobs/{job_id}/temporal/svi-log` → `List[SVIGenerationRecord]`

**Does NOT own:**
- Temporal generation logic (owned by `temporal_engine.py`)
- Job management (owned by `routes/jobs.py`)

---

### 14.2 `vga/api/routes/identity.py`

**Single responsibility:** Expose HTTP endpoints for identity state (cumulative drift, per-stage history).

**Endpoints owned:**
- `GET /jobs/{job_id}/identity/state` → `IdentityStateResponse`
- `GET /jobs/{job_id}/identity/log` → `List[IdentityStateRecord]`

**Does NOT own:**
- Identity computation (owned by `identity_state_tracker.py`)
- CLIP validation (owned by `clip_validator.py`)

---

### 14.3 `vga/api/routes/audio.py`

**Single responsibility:** Expose HTTP endpoints for audio quality validation results.

**Endpoints owned:**
- `GET /jobs/{job_id}/audio/validation` → `AudioValidationResponse`
- `GET /jobs/{job_id}/audio/alignment` → `CrossModalAlignmentReport`

**Does NOT own:**
- Audio mixing (owned by `audio_mixing_agent.py`)
- Validation computation (owned by `audio_quality_validator.py`)

---

### 14.4 `vga/api/routes/composition.py`

**Single responsibility:** Expose HTTP endpoints for CompositionPlan retrieval and update.

**Endpoints owned:**
- `GET /jobs/{job_id}/composition/{scene_id}` → `CompositionPlanSchema`
- `PATCH /jobs/{job_id}/composition/{scene_id}` (body: `CompositionPlanUpdateRequest`) → updated plan

**Does NOT own:**
- CompositionPlan generation (owned by `scene_composition_agent.py`)
- HRG-4 approval (owned by `routes/hrg.py`)

---

## 15. New UI Component Files (v17.0)

### 15.1 `vga/ui/components/temporal_engine_panel.py`

**Single responsibility:** Render the Streamlit panel showing TemporalEngine health: buffer frame count, motion direction per segment, SVI CFG + steps per segment, identity score per segment.

**Displays:**
- TemporalBuffer status: frame_count indicator (green if 5, red if not)
- Motion state log: direction + magnitude per segment (table or chart)
- SVI generation log: cfg, steps, clip_score, continuity per segment
- Identity drift chart: cumulative drift across all validation points

**Does NOT own:**
- Triggering regeneration (owned by HRG panels + `hrg_controller.py`)
- Fetching data (calls `GET /jobs/{id}/temporal/...` endpoints)

---

### 15.2 `vga/ui/components/hrg_panels/hrg_2_scene_plan.py`

**Single responsibility:** Render the HRG-2 checkpoint panel for Scene/Segment Plan Review. Display scene list with durations and beats, plus segment breakdown per scene. Gate S-03 on approval.

**Displays:** scene list (duration, beats_count), segment plan breakdown table.
**Actions:** approve | edit scene durations | trigger replanning

---

### 15.3 `vga/ui/components/hrg_panels/hrg_4_composition.py`

**Single responsibility:** Render the HRG-4 checkpoint panel for Scene Composition Review. Display all 6 CompositionPlan fields, each editable. Gate S-05 on approval (RULE-88).

**Displays:** all 6 CompositionPlan fields rendered as labeled + editable widgets.
**Actions:** approve | edit any field | trigger recompose

---

### 15.4 `vga/ui/components/hrg_panels/hrg_8_motion_qa.py` (updated from v16.0 hrg_6)

**Single responsibility:** Render HRG-8 checkpoint for Motion QA. Display video segments with continuity score breakdown. **v17.0 addition:** show `identity_per_segment` chart.

**Displays (v17.0 updates):** video player per segment + continuity scores + identity-per-segment chart (bar chart of CLIP scores).
**Actions:** approve | trigger scene regeneration

---

### 15.5 `vga/ui/components/hrg_panels/hrg_10_lipsync_qa.py` (updated from v16.0 hrg_8)

**Single responsibility:** Render HRG-10 checkpoint for Lip Sync QA. **v17.0 addition:** show `identity_delta_per_segment` alongside phoneme alignment scores.

**Displays (v17.0 updates):** lip-synced video + phoneme alignment scores + identity_delta per segment (red badge if > 0.03).
**Actions:** approve | trigger re-sync | upload alternative audio

---

### 15.6 `vga/ui/components/hrg_panels/hrg_11_final_qa.py` (updated from v16.0 hrg_9)

**Single responsibility:** Render HRG-11 checkpoint for Final Audio QA. **v17.0 addition:** show SNR badge, clipping status, and level meters.

**Displays (v17.0 updates):** full scene video + mixed audio player + SNR badge (green ≥ 10 dB, red < 10 dB) + clipping badge (green if none, red if detected) + level meter for dialogue/ambient/music.
**Actions:** approve (→ triggers export) | trigger remix | adjust levels

---

## 16. New Prompt File (v17.0)

### 16.1 `vga/config/prompts/composition_prompts.py`

**Single responsibility:** Contain all prompt templates used by `SceneCompositionAgent`. Including system prompt, user prompt template, and field-specific examples for CompositionPlan generation.

**Contains:**
- `COMPOSITION_SYSTEM_PROMPT` — tells Qwen to output structured CompositionPlanSchema JSON
- `COMPOSITION_USER_PROMPT_TEMPLATE` — f-string template with placeholders for dialogue, emotion, motion_intent, characters, environment
- `CAMERA_ANGLE_EXAMPLES` — examples of valid camera_angle values with use cases
- `MOTION_VECTOR_EXAMPLES` — examples of motion_vector values with narrative contexts

**Does NOT own:**
- Calling the model (owned by `scene_composition_agent.py`)
- Validation (owned by `composition_validator.py`)

---

## 17. Updated Core Files (v17.0 summary)

### 17.1 `vga/config/settings.py` (extended)

v17.0 additions to the constants block (all v16.0 constants unchanged):

```python
# Temporal Engine
TEMPORAL_BUFFER_SIZE: int = 5
TEMPORAL_MAX_RETRIES_PER_SEGMENT: int = 3
SEGMENT_CONTINUITY_MIN: float = 0.85
SVI_CFG_MIN: float = 5.0
SVI_CFG_MAX: float = 6.0
SVI_CFG_DEFAULT: float = 5.5
STEPS_CRITICAL: int = 50
STEPS_STANDARD: int = 30
LORA_WEIGHT_HIGH_NOISE: float = 0.6
LORA_WEIGHT_MID_NOISE: float = 0.5
LORA_WEIGHT_LOW_NOISE: float = 0.4
HIGH_NOISE_FRACTION: float = 0.67
MID_NOISE_FRACTION: float = 0.33
MOTION_STATIONARY_THRESHOLD: float = 0.02

# Scene Composition
COMPOSITION_MAX_RETRIES: int = 3

# Identity State
IDENTITY_CUMULATIVE_DRIFT_THRESHOLD: float = 0.15
IDENTITY_MAX_PHASE_REGENERATIONS: int = 1

# Audio Quality
MIN_SNR_DB: float = 10.0
MAX_PEAK_DBFS: float = 0.0
HEADROOM_DB: float = 1.0
AUDIO_QUALITY_MAX_RETRIES: int = 3

# HRG
HRG_CHECKPOINT_COUNT: int = 11

# Schema version
SCHEMA_VERSION: str = "v6.0"
```

---

### 17.2 `vga/state/immutable_context.py` (updated to 5-dimensional)

v17.0 expands from 4-dimensional context to **5-dimensional**:

```python
# v16.0 context dimensions (retained):
#   identity_state, motion_state, camera_state, lighting_state

# v17.0 adds:
#   temporal_state: TemporalState(buffer: TemporalBuffer, segment_index: int)

@dataclass(frozen=True)
class ImmutableContext:
    job_id: str
    scene_id: str
    identity_state: IdentityState
    motion_state: MotionState
    camera_state: CameraState
    lighting_state: LightingState
    temporal_state: TemporalState      # NEW v17.0
    adaptive_params: dict
    last_output: Optional[Any] = None

    def evolve(self, updates: dict) -> "ImmutableContext":
        """Return new ImmutableContext with applied updates (frozen pattern)."""
        return ImmutableContext(**{**self.__dict__, **updates})

    def has_output(self, stage_id: str) -> bool:
        """SYSTEM DIRECTIVE v17: check predecessor output exists."""
        return (self.last_output is not None and
                getattr(self.last_output, 'stage_id', None) == stage_id)
```

---

### 17.3 `vga/core/hrg_controller.py` (updated to 11 checkpoints)

**v17.0 change:** `VALID_CHECKPOINTS = {f"HRG-{i}" for i in range(1, 12)}` (was range(1, 10)).

HRG-1, HRG-3, HRG-5 through HRG-11: logic identical to v16.0 equivalents.
HRG-2 (Scene Plan): new checkpoint; `HRG2DisplayData` as display_data.
HRG-4 (Composition): new checkpoint; `HRG4DisplayData` as display_data.

---

### 17.4 `vga/core/master_orchestrator.py` (updated — execute_stage contract)

**v17.0 change:** All stage calls now route through `execute_stage(stage, input, context)`.
This is **SYSTEM DIRECTIVE v17**. Every pipeline stage without exception.

Old pattern (v16.0):
```python
output = agent.run(input)
context = context.evolve(output)
```

New pattern (v17.0 — SYSTEM DIRECTIVE):
```python
output, context = execute_stage(stage, input, context)
```

---

## 18. File Responsibility Anti-Patterns (v17.0)

The following are FORBIDDEN (extends v16.0 anti-patterns):

```
FORBIDDEN: scene_composition_agent.py calling flux_wrapper.generate()
  → Composition planning and image generation are separate responsibilities.

FORBIDDEN: temporal_engine.py calling model_manager.unload()
  → Model lifecycle is owned by master_orchestrator.py.

FORBIDDEN: svi_wrapper.py containing the autoregressive loop
  → Autoregressive logic is owned by temporal_engine.py.

FORBIDDEN: svi_wrapper.py applying a static LoRA weight
  → Weight schedule is owned by svi_scheduler.py; must receive scheduler, not float.

FORBIDDEN: audio_mixing_agent.py computing SNR directly (without audio_quality_validator)
  → SNR computation is owned by audio_quality_validator.py.

FORBIDDEN: audio_quality_validator.py calling AudioMixingAgent.remix()
  → Re-mixing is owned by audio_mixing_agent.py; validator is stateless.

FORBIDDEN: identity_state_tracker.py deciding what to do on drift threshold exceeded
  → It raises IdentityCumulativeDriftError; caller (temporal_engine.py / lip_sync_agent.py)
     decides recovery action.

FORBIDDEN: temporal_buffer_manager.py calling svi_wrapper.generate()
  → Buffer management and generation are separate responsibilities.

FORBIDDEN: composition_validator.py creating or modifying CompositionPlan
  → It validates only; never creates or modifies.

FORBIDDEN: Any agent bypassing execute_stage() contract (SYSTEM DIRECTIVE v17)
  → Direct agent.run() calls outside execute_stage() are a CGRL violation.

FORBIDDEN: hrg_controller.py having more than 11 valid checkpoints
  → HRG_CHECKPOINT_COUNT = 11 is authoritative; no dynamic addition.

FORBIDDEN: immutable_context.py being mutable
  → frozen=True; context.evolve() always returns new instance.

FORBIDDEN: Any import of vga.devtools in production files
  → devtools is dev-time only; zero runtime footprint.
```

---

## 19. File Responsibility Change Log (v17.0 vs v16.0)

| File | Change Type | What Changed |
|---|---|---|
| `bootstrap.py` | Extended | Steps 6Z-p through 6Z-z: initialize v17.0 singletons |
| `config/settings.py` | Extended | 27 new constants; all v16.0 constants unchanged |
| `models/schemas.py` | Extended | 8 new schemas; 3 updated schemas; all v16.0 unchanged |
| `core/exceptions.py` | Extended | 8 new exception classes; all v16.0 unchanged |
| `core/hrg_controller.py` | Updated | 9 → 11 valid checkpoints; HRG-2 and HRG-4 added |
| `core/master_orchestrator.py` | Updated | execute_stage() contract mandatory for all stages |
| `state/immutable_context.py` | Updated | 4 → 5 dimensional; `temporal_state` added; `has_output()` added |
| `state/context_factory.py` | Updated | Creates 5-dim initial context with `temporal_state: TemporalState(buffer=None, segment_index=0)` |
| `agents/base_image_agent.py` | Updated | Accepts CompositionPlan; asserts plan present (RULE-88) |
| `agents/scene_expansion_agent.py` | Updated | CompositionPlan fully bound to character_positions + motion_vector |
| `agents/image_refinement_agent.py` | Updated | Freezes char_identity_ref in context after best image selected |
| `agents/continuity_validation_agent.py` | Updated | Adds `identity_per_segment: List[float]` to ContinuityReport |
| `agents/lip_sync_agent.py` | Updated | Calls `IdentityStateTracker.update()` per synced segment |
| `agents/audio_mixing_agent.py` | Updated | Calls AudioQualityValidator + CrossModalAlignmentValidator post-mix |
| `agents/quality_agent.py` | Updated | Includes v17.0 fields in PipelineReport |
| `models/wrappers/svi_wrapper.py` | Updated | Accepts `init_latents` (5-frame Tensor) + `lora_scheduler` |
| `models/wrappers/wan_wrapper.py` | Updated | Accepts `motion_params: dict` from CompositionPlan |
| `ui/components/hrg_panels/hrg_8_motion_qa.py` | Updated | Added `identity_per_segment` chart |
| `ui/components/hrg_panels/hrg_10_lipsync_qa.py` | Updated | Added `identity_delta_per_segment` display |
| `ui/components/hrg_panels/hrg_11_final_qa.py` | Updated | Added SNR badge + clipping badge + level meters |
| All other HRG panels (1, 3, 5, 6, 7, 9) | Renumbered | Logic unchanged; only checkpoint ID updated |
