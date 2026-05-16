# VGA Development Sequence & Build Order
**Project:** Video Generation Automation (VGA) — Motivation System
**Version:** 17.2.0
**Status:** Authoritative Reference
**Audience:** Claude Code Agent, All Engineers

---

## Overview

This document defines the exact sequence in which every file must be written during implementation. **Build order is strict.** Every file depends on those above it in the sequence. Claude Code MUST follow this order exactly.

**Retained from v15.0 (Phases 0–9):** All build steps unchanged.
**Retained from v16.0 (Phases 10–11):** All v16.0 build steps unchanged.
**New in v17.0 (Phase 12):** 42 new build steps for all v17.0 components.

---

## Phase 0–9: All v15.0 Build Steps Retained (unchanged)

---

## Phase 10–11: All v16.0 Build Steps Retained (unchanged)

---

## Phase 12: v17.0 Architecture Components

Phase 12 builds all v17.0 additions in strict dependency order. Foundation files are built before the components that depend on them.

---

### Phase 12.0 — Foundation: Constants and Schema Additions

**Step 12.0.1 — Extend `config/settings.py` with v17.0 constants**

Add all v17.0 constants to the EXISTING settings.py. Do NOT create a new file.

```python
# Append to existing settings.py:

# ── Scene Composition ─────────────────────────────────────────
COMPOSITION_MAX_RETRIES: int = 3
SLA_COMPOSITION_MAX_S: float = 15.0

# ── Temporal Engine ───────────────────────────────────────────
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

# ── Identity State ────────────────────────────────────────────
IDENTITY_CUMULATIVE_DRIFT_THRESHOLD: float = 0.15
IDENTITY_MAX_PHASE_REGENERATIONS: int = 1

# ── Audio Quality ─────────────────────────────────────────────
MIN_SNR_DB: float = 10.0
MAX_PEAK_DBFS: float = 0.0
HEADROOM_DB: float = 1.0
AUDIO_QUALITY_MAX_RETRIES: int = 3

# ── HRG (11 checkpoints) ──────────────────────────────────────
HRG_CHECKPOINT_COUNT: int = 11

# ── Schema Version ────────────────────────────────────────────
SCHEMA_VERSION: str = "v6.0"
```

---

**Step 12.0.2 — Add v17.0 schemas to `models/schemas.py`**

Append to existing schemas.py WITHOUT modifying any existing schema:

```python
# New dataclasses/enums:
class TemporalState:
    buffer: Optional[Any]       # TemporalBuffer once initialized
    segment_index: int

class CameraState:
    angle: Optional[str]
    motion: Optional[str]

class LightingState:
    style: Optional[str]

# New Pydantic models (append):
class CompositionPlanSchema(BaseModel): ...
class TemporalBufferRecord(BaseModel): ...
class MotionStateRecord(BaseModel): ...
class SVIGenerationRecord(BaseModel): ...
class IdentityStateRecord(BaseModel): ...
class AudioQualityRecord(BaseModel): ...
class CrossModalAlignmentRecord(BaseModel): ...
class CrossModalAlignmentReport(BaseModel): ...
class HRG2DisplayData(BaseModel): ...
class HRG4DisplayData(BaseModel): ...
class HRG8DisplayData(BaseModel): ...       # updated: adds identity_per_segment
class HRG10DisplayData(BaseModel): ...      # updated: adds identity_delta_per_segment
class HRG11DisplayData(BaseModel): ...      # updated: adds snr_db, peak_db, clipping_detected
class TemporalBufferStatusResponse(BaseModel): ...
class IdentityStateResponse(BaseModel): ...
class AudioValidationResponse(BaseModel): ...
class CompositionPlanUpdateRequest(BaseModel): ...

# Update existing (add field):
# ContinuityReport: add identity_per_segment: List[float]
```

---

**Step 12.0.3 — Add v17.0 exceptions to `core/exceptions.py`**

Append to existing exceptions.py:

```python
class CompositionPlanValidationError(VGABaseError): ...
class TemporalBufferError(VGABaseError): ...
class SVICFGViolationError(VGABaseError): ...
class AutoregressiveViolationError(VGABaseError): ...
class TemporalSegmentFailureError(VGABaseError): ...
class IdentityCumulativeDriftError(VGABaseError): ...
class IdentityReferenceCorruptionError(VGABaseError): ...
class AudioQualityError(VGABaseError): ...
class MissingPredecessorOutputError(VGABaseError): ...
```

---

### Phase 12.1 — Foundation: State and Context

**Step 12.1.1 — Update `state/immutable_context.py`**

Extend the frozen dataclass to 5-dimensional context:
- Add `temporal_state: TemporalState`
- Add `camera_state: CameraState`
- Add `lighting_state: LightingState`
- Update `has_output(stage_id: str) → bool` method
- Update `evolve()` to pass through new fields

**Step 12.1.2 — Update `state/context_factory.py`**

Update `create_initial()` to produce 5-dimensional context:
```python
return ImmutableContext(
    job_id=job_id, scene_id=scene_id,
    identity_state=IdentityState(embedding_vector=None, drift_score=0.0, history=[]),
    motion_state=MotionState(velocity_x=0.0, velocity_y=0.0, velocity_magnitude=0.0,
                              direction="stationary", is_stationary=True),
    camera_state=CameraState(angle=None, motion=None),
    lighting_state=LightingState(style=None),
    temporal_state=TemporalState(buffer=None, segment_index=0),
    adaptive_params={},
    last_output=None
)
```

---

### Phase 12.2 — Foundation: Storage Paths

**Step 12.2.1 — Update `core/storage.py`**

Add v17.0 storage paths and write methods:

```python
COMPOSITION_PATH = "/workspace/composition/{job_id}/{scene_id}/"
TEMPORAL_LOG_PATH = "/workspace/temporal/{job_id}/{scene_id}/"
IDENTITY_LOG_PATH = "/workspace/identity/{job_id}/"
VIDEO_SEGMENT_PATH = "/workspace/video/{job_id}/{scene_id}/"

# New methods:
def write_composition_plan(job_id, scene_id, plan: dict): ...
def append_temporal_buffer_record(job_id, scene_id, record: TemporalBufferRecord): ...
def append_motion_state_record(scene_id, record: MotionStateRecord): ...
def append_svi_generation_record(job_id, scene_id, record: SVIGenerationRecord): ...
def append_identity_state_record(record: IdentityStateRecord): ...
def append_audio_quality_record(job_id, scene: str, record: AudioQualityRecord): ...
def write_cross_modal_alignment(scene_id, report: CrossModalAlignmentReport): ...
def write_video_segment(job_id, scene_id, segment_index: int, video_tensor): ...
```

---

### Phase 12.3 — Validation Layer

Build validation files before the agents that depend on them.

**Step 12.3.1 — Create `validation/composition_validator.py`**

```python
class CompositionPlanValidator:
    REQUIRED_FIELDS = ["camera_angle", "camera_motion", "character_positions",
                       "focus_subject", "lighting_style", "motion_vector"]
    def validate(self, plan: CompositionPlanSchema) -> bool: ...
    def assert_in_context(self, context) -> None: ...
```

No external dependencies beyond schemas and exceptions. Test: `unit/test_scene_composition_agent.py` (validator section).

**Step 12.3.2 — Create `validation/audio_quality_validator.py`**

```python
class AudioQualityValidator:
    def validate(self, mixed, dialogue, scene_id, job_id) -> AudioQualityRecord: ...
    def normalize(self, mixed, target_peak_db=-1.0) -> AudioSegment: ...
    def compute_snr(self, mixed, dialogue) -> float: ...
    def compute_peak_db(self, audio) -> float: ...
```

Dependencies: `pydub`, `math`, `models/schemas.py`, `config/settings.py`. Test: `unit/test_audio_quality_validator.py`.

**Step 12.3.3 — Create `validation/cross_modal_alignment_validator.py`**

```python
class CrossModalAlignmentValidator:
    def validate_alignment(self, video_segments, audio_segments, scene_id) -> CrossModalAlignmentReport: ...
```

Dependencies: `models/schemas.py`, `config/settings.py`, `core/storage.py`. Test: `unit/test_cross_modal_alignment_validator.py`.

---

### Phase 12.4 — Prompt Templates

**Step 12.4.1 — Create `config/prompts/composition_prompts.py`**

```python
COMPOSITION_SYSTEM_PROMPT = """
You are a cinematographer translating narrative intent into visual directives.
Output a JSON object with exactly these 6 fields:
  camera_angle, camera_motion, character_positions, focus_subject,
  lighting_style, motion_vector.
Output ONLY valid JSON. No preamble, no markdown, no explanation.
"""

COMPOSITION_USER_PROMPT_TEMPLATE = """
Scene data:
  dialogue: {dialogue}
  emotion: {emotion}
  motion_intent: {motion_intent}
  characters: {characters}
  environment: {environment}

Produce the CompositionPlan JSON.
"""

CAMERA_ANGLE_EXAMPLES = { ... }
MOTION_VECTOR_EXAMPLES = { ... }
```

---

### Phase 12.5 — Temporal Engine Subsystem

Build in strict dependency order within the subsystem.

**Step 12.5.1 — Create `temporal/temporal_buffer_manager.py`**

Dependencies: `torch`, `models/schemas.py` (TemporalBufferRecord), `core/exceptions.py` (TemporalBufferError), `config/settings.py` (TEMPORAL_BUFFER_SIZE). **No agent dependencies.**

Implement:
- `TemporalBuffer` dataclass
- `TemporalBufferManager.init(segment_1) → TemporalBuffer`
- `TemporalBufferManager.update(buffer, new_segment) → TemporalBuffer`
- `TemporalBufferManager.encode(buffer) → Tensor` (shape: 5, C', H', W')
- `_extract_last_n_frames(video_tensor, n) → Tensor`

Test: `unit/test_temporal_buffer_manager.py` — test init, update, encode; test TemporalBufferError on < 5 frames.

**Step 12.5.2 — Create `temporal/svi_scheduler.py`**

Dependencies: `config/settings.py`, `core/exceptions.py` (SVICFGViolationError). **No torch dependency (pure Python).**

Implement:
- `SVIScheduler.__init__(total_steps, tracer)`
- `apply_lora(timestep: int) → float`
- `assert_cfg_valid(cfg: float) → float`
- `get_lora_schedule() → list`

Test: `unit/test_svi_scheduler.py` — test weight for each phase; test cfg gate; test error on out-of-range.

**Step 12.5.3 — Create `temporal/motion_state_tracker.py`**

Dependencies: `torch`, `models/schemas.py` (MotionStateRecord), `config/settings.py` (MOTION_STATIONARY_THRESHOLD).

Implement:
- `MotionState` dataclass
- `MotionStateTracker.estimate(frames: Tensor) → MotionState`
- `log_state(segment_id, scene_id, state) → None`
- `_compute_optical_flow(frame_a, frame_b)` — use torchvision or cv2 fallback
- `_classify_direction(vx, vy, magnitude, stationary) → str`

Test: `unit/test_motion_state_tracker.py` — test stationary detection; test direction classification; test with 5-frame batch.

**Step 12.5.4 — Create `temporal/temporal_retry_controller.py`**

Dependencies: `config/settings.py` only. Pure Python.

Implement: `TemporalRetryController.adjust(attempt, clip_score, cont_score)`.

Test: `unit/test_temporal_retry_controller.py` (minimal — pure logic test).

**Step 12.5.5 — Update `models/wrappers/svi_wrapper.py`**

Modify existing `svi_wrapper.py` to:
- Accept `init_latents: torch.Tensor` (shape: 5, C', H', W') instead of `init_image`
- Accept `lora_scheduler: SVIScheduler` instead of `lora_weight: float`
- Call `lora_scheduler.apply_lora(timestep)` at each denoising step
- Raise `AutoregressiveViolationError` if `init_latents.shape[0] != TEMPORAL_BUFFER_SIZE`

Test: `unit/test_svi_wrapper.py` — test multi-frame acceptance; test single-frame rejection.

**Step 12.5.6 — Create `temporal/temporal_engine.py`**

Dependencies: ALL phase 12.5.1–12.5.5 components + `identity_state_tracker.py` (step 12.6.1) + `validation/clip_validator.py` (existing) + `state/immutable_context.py` (updated).

**Note:** `temporal_engine.py` depends on `identity_state_tracker.py`. Build identity_state_tracker FIRST (step 12.6.1 before 12.5.6).

Implement:
- `TemporalEngine.generate_scene(segment_plans, segment_1, context, char_identity_ref) → (List, ImmutableContext)`
- `_generate_segment_with_retry(n, segment_plan, buffer, motion_state, ...) → VideoSegment`
- `_build_temporal_prompt(segment_plan, motion_state, context) → str`

Test: `integration/test_temporal_engine.py` — test sequential ordering; test buffer update per segment; test identity check per segment; test retry loop.

---

### Phase 12.6 — Identity State Tracker

**Step 12.6.1 — Create `identity/identity_state_tracker.py`**

(Must be built before `temporal_engine.py` — see dependency note above)

Dependencies: `torch`, `validation/clip_validator.py` (existing), `models/schemas.py` (IdentityStateRecord), `core/exceptions.py` (IdentityCumulativeDriftError), `config/settings.py`.

Implement:
- `IdentityStateTracker.__init__(clip_encoder, tracer, storage)`
- `update(char_identity_ref, new_frame, stage_id) → dict`
- `reset()` — clear drift accumulator

Test: `unit/test_identity_state_tracker.py` — test delta computation; test cumulative accumulation; test threshold exception; test reset.

---

### Phase 12.7 — Scene Composition Agent

**Step 12.7.1 — Create `agents/scene_composition_agent.py`**

Dependencies: `validation/composition_validator.py` (step 12.3.1), `config/prompts/composition_prompts.py` (step 12.4.1), `models/schemas.py`, `core/exceptions.py`, `models/wrappers/qwen_wrapper.py` (existing), `config/settings.py`.

Implement:
- `SceneCompositionAgent.compose(scene_data, trace_id) → CompositionPlanSchema`
- Retry loop up to `COMPOSITION_MAX_RETRIES`
- Storage write: `composition_plan_{scene_id}.json`

Test: `unit/test_scene_composition_agent.py` — test valid generation; test retry on schema failure; test error after max retries.

---

### Phase 12.8 — Video Segment Generator (Wan2.2 Stage S-08)

**Step 12.8.1 — Create `agents/video_segment_generator.py`**

Dependencies: `temporal/temporal_buffer_manager.py` (step 12.5.1), `validation/clip_validator.py` (existing), `models/wrappers/wan_wrapper.py` (existing — needs update), `validation/composition_validator.py` (step 12.3.1), `core/storage.py` (updated, step 12.2.1).

Also update `models/wrappers/wan_wrapper.py` to accept `motion_params: dict` from CompositionPlan.

Implement:
- `VideoSegmentGenerator.generate(refined_image_path, composition_plan, context, char_identity_ref) → VideoSegment`
- CLIPValidator call on Segment_1 keyframe
- `TemporalBufferManager.init(segment_1)` call
- Buffer frame count assertion

Test: `integration/test_video_segment_generator.py` — test buffer initialization; test identity check; test TemporalBufferError on failure.

---

### Phase 12.9 — Agent Updates

Update existing agents to incorporate v17.0 features.

**Step 12.9.1 — Update `agents/base_image_agent.py`**

Add:
- `CompositionPlanValidator.assert_in_context(context)` gate before generation
- CompositionPlan fields injected into prompt: camera_angle, lighting_style, character_positions

**Step 12.9.2 — Update `agents/scene_expansion_agent.py`**

Add:
- Full CompositionPlan binding (character_positions, focus_subject, motion_vector → prompt injection)

**Step 12.9.3 — Update `agents/image_refinement_agent.py`**

Add:
- After finding `best_image` (highest CLIP score): freeze `char_identity_ref` in context
- `context = context.evolve({"identity_state": IdentityState(embedding_vector=clip_encoder(best_image), ...)})`
- Log `identity_ref_frozen` event

**Step 12.9.4 — Update `agents/continuity_validation_agent.py`**

Add:
- Compute `identity_per_segment = [clip_validator.score(seg.keyframe, char_identity_ref) for seg in segments]`
- Include `identity_per_segment` in `ContinuityReport`

**Step 12.9.5 — Update `agents/lip_sync_agent.py`**

Add:
- After each `latentsync.sync()`: call `identity_tracker.update(char_identity_ref, synced_frame, f"S-12_seg_{i}")`
- Check `delta <= LIPSYNC_IDENTITY_DELTA_THRESHOLD` (0.03)
- `context = context.evolve({"identity_state": identity_tracker.current_state})`

**Step 12.9.6 — Update `agents/audio_mixing_agent.py`**

Add:
- After `mixer.mix()`: call `audio_quality_validator.validate(mixed, dialogue, scene_id, job_id)`
- If `peak_db > 0`: call `audio_quality_validator.normalize(mixed)`
- Call `cross_modal_alignment_validator.validate_alignment(video_segments, audio_segments, scene_id)`
- Include SNR, peak, alignment in HRG-11 display data

**Step 12.9.7 — Update `agents/quality_agent.py`**

Add v17.0 fields to `PipelineReport`:
- `composition_plan_summary`, `temporal_engine_health`, `motion_state_summary`
- `identity_state_final`, `audio_quality_summary`, `cross_modal_alignment_summary`
- `identity_per_segment_video`, `identity_delta_per_segment_sync`
- `rule_compliance` (RULE-86 through RULE-99 compliance status)

---

### Phase 12.10 — HRG Controller Update

**Step 12.10.1 — Update `core/hrg_controller.py`**

Extend to 11 checkpoints:
- `VALID_CHECKPOINTS = {f"HRG-{i}" for i in range(1, 12)}`
- `_events` dict: 11 entries
- `_decisions` dict: 11 entries
- `submit_decision()`, `require_approval()` logic unchanged

Test: `unit/test_hrg_controller.py` — test all 11 checkpoints valid; test HRG-0 and HRG-12 rejected; test HRG-2 and HRG-4 flow.

---

### Phase 12.11 — Master Orchestrator Update

**Step 12.11.1 — Update `core/master_orchestrator.py`**

Add `execute_stage()` function and update `STAGE_ORDER` and `HRG_GATES`:

```python
STAGE_ORDER = [
    "S-01_script", "S-02_scene_segment", "S-03_identity", "S-04_composition",
    "S-05_base_image", "S-06A_multi_angle", "S-06B_image_merge",
    "S-06C_scene_expansion", "S-07_refinement", "PHASE_2B_cleanup",
    "S-08_wan_segment_1", "S-09_temporal_engine", "S-10_continuity_validation",
    "S-11_dialogue", "S-12_lip_sync", "S-13_ambient", "S-14_music",
    "S-15_audio_mix", "S-16a_assembly", "S-16b_export", "S-16c_quality"
]

HRG_GATES = {
    "S-01_script": "HRG-1",
    "S-02_scene_segment": "HRG-2",
    "S-03_identity": "HRG-3",
    "S-04_composition": "HRG-4",
    "S-05_base_image": "HRG-5",
    "S-06C_scene_expansion": "HRG-6",
    "S-07_refinement": "HRG-7",
    "S-10_continuity_validation": "HRG-8",
    "S-11_dialogue": "HRG-9",
    "S-12_lip_sync": "HRG-10",
    "S-15_audio_mix": "HRG-11",
}
```

Test: `integration/test_hrg_flow.py` — test 11-checkpoint flow; test HRG-2 gates S-03; test HRG-4 gates S-05.

---

### Phase 12.12 — API Routes (New)

**Step 12.12.1 — Create `api/routes/temporal.py`**

Endpoints: `GET /jobs/{id}/temporal/buffer`, `/motion`, `/svi-log`.
Read from storage; return appropriate response schemas.

**Step 12.12.2 — Create `api/routes/identity.py`**

Endpoints: `GET /jobs/{id}/identity/state`, `/log`.

**Step 12.12.3 — Create `api/routes/audio.py`**

Endpoints: `GET /jobs/{id}/audio/validation`, `/alignment`.

**Step 12.12.4 — Create `api/routes/composition.py`**

Endpoints: `GET /jobs/{id}/composition/{scene_id}`, `PATCH /jobs/{id}/composition/{scene_id}`.

**Step 12.12.5 — Update `api/routes/hrg.py`**

Update checkpoint validation to accept HRG-1 through HRG-11.
Add `HRG2DisplayData` and `HRG4DisplayData` to the display_data union type.

**Step 12.12.6 — Update `api/main.py`**

Register new routers: `temporal_router`, `identity_router`, `audio_router`, `composition_router`.

Test: `integration/test_fastapi_endpoints.py` — test all new endpoints; test HRG-2 and HRG-4 submission.

---

### Phase 12.13 — Streamlit UI (New Panels)

**Step 12.13.1 — Create `ui/components/hrg_panels/hrg_2_scene_plan.py`**

Renders: scene list table (scene_id, duration, beats_count), segment breakdown accordion.
Actions: approve button, edit scene durations form, trigger replanning button.

**Step 12.13.2 — Create `ui/components/hrg_panels/hrg_4_composition.py`**

Renders: 6 labeled editable fields for CompositionPlan.
Actions: approve button, individual field edit widgets, trigger recompose button.

**Step 12.13.3 — Create `ui/components/temporal_engine_panel.py`**

Renders: buffer status indicator, motion state per segment, SVI metrics, identity drift chart.
Data source: `GET /jobs/{id}/temporal/*` endpoints.

**Step 12.13.4 — Update `ui/components/hrg_panels/hrg_8_motion_qa.py`**

Add identity-per-segment bar chart below continuity scores.
Source: `continuity_report.identity_per_segment`.

**Step 12.13.5 — Update `ui/components/hrg_panels/hrg_10_lipsync_qa.py`**

Add identity_delta_per_segment column (red badge if delta > 0.03).

**Step 12.13.6 — Update `ui/components/hrg_panels/hrg_11_final_qa.py`**

Add SNR badge (green/red), clipping badge (green/red), and level meters.
Source: AudioQualityRecord and CrossModalAlignmentReport.

**Step 12.13.7 — Renumber remaining HRG panels**

Update checkpoint IDs in these existing panel files:
- `hrg_2_identity.py` → `hrg_3_identity.py`
- `hrg_3_base_images.py` → `hrg_5_base_images.py`
- `hrg_4_composed_images.py` → `hrg_6_composed_images.py`
- `hrg_5_refined_images.py` → `hrg_7_refined_images.py`
- `hrg_6_motion_qa.py` → `hrg_8_motion_qa.py` (renaming + update from step 12.13.4)
- `hrg_7_voice_qa.py` → `hrg_9_voice_qa.py`
- `hrg_8_lipsync_qa.py` → `hrg_10_lipsync_qa.py` (renaming + update from step 12.13.5)
- `hrg_9_final_qa.py` → `hrg_11_final_qa.py` (renaming + update from step 12.13.6)

Test: `integration/test_fastapi_endpoints.py` (Streamlit tested manually; FastAPI tested automatically).

---

### Phase 12.14 — Schema Migrations

**Step 12.14.1 — Update `core/schema_migrations.py`**

Add `_migrate_v5_2_to_v6_0()` function:
- Updates `schema_version` to `"v6.0"`
- Adds `identity_per_segment: []` to ContinuityReport artifacts
- Re-maps HRG checkpoint IDs for archived v16.0 decisions (old HRG-2 → new HRG-3, etc.)

---

### Phase 12.15 — Bootstrap Update

**Step 12.15.1 — Update `bootstrap.py`**

Add steps 6Z-p through 6Z-z (after all v16.0 steps):

```python
# 6Z-p: SceneCompositionAgent singleton
scene_composition_agent = SceneCompositionAgent(
    model_wrapper=qwen_wrapper,
    prompt_builder=PromptBuilder(composition_prompts),
    tracer=tracer,
    system_guard=system_guard,
    authority_manager=authority_manager,
    composition_validator=composition_validator
)

# 6Z-q: TemporalBufferManager singleton
temporal_buffer_manager = TemporalBufferManager(vae_encoder=vae_encoder, tracer=tracer)

# 6Z-r: SVIScheduler singleton (initialized per-scene; placeholder here)
svi_scheduler_factory = SVISchedulerFactory(tracer=tracer)

# 6Z-s: MotionStateTracker singleton
motion_state_tracker = MotionStateTracker(tracer=tracer, storage=storage)

# 6Z-t: TemporalRetryController singleton
temporal_retry_controller = TemporalRetryController()

# 6Z-u: IdentityStateTracker singleton (reset per scene)
identity_state_tracker = IdentityStateTracker(
    clip_encoder=clip_encoder, tracer=tracer, storage=storage
)

# 6Z-v: AudioQualityValidator singleton
audio_quality_validator = AudioQualityValidator(tracer=tracer, storage=storage)

# 6Z-w: CrossModalAlignmentValidator singleton
cross_modal_alignment_validator = CrossModalAlignmentValidator(tracer=tracer, storage=storage)

# 6Z-x: CompositionPlanValidator singleton
composition_validator = CompositionPlanValidator()

# 6Z-y: TemporalEngine singleton
temporal_engine = TemporalEngine(
    svi_wrapper=svi_wrapper,
    buffer_manager=temporal_buffer_manager,
    svi_scheduler=None,  # created per-scene with correct total_steps
    motion_tracker=motion_state_tracker,
    retry_controller=temporal_retry_controller,
    clip_validator=clip_validator,
    identity_tracker=identity_state_tracker,
    continuity_validator=continuity_validator,
    tracer=tracer
)

# 6Z-z: HRGController (update to 11 checkpoints — already initialized in step 16; verify here)
assert len(hrg_controller.VALID_CHECKPOINTS) == 11

# Bootstrap assertions (v17.0):
assert TEMPORAL_BUFFER_SIZE == 5
assert SCHEMA_VERSION == "v6.0"
assert HRG_CHECKPOINT_COUNT == 11
try:
    import torchvision
except ImportError:
    import cv2  # cv2 optical flow fallback
```

---

### Phase 12.16 — Snapshot and Deviation Log

**Step 12.16.1 — Create snapshot directory `snapshots/v17_candidate/`**

```bash
mkdir -p /workspace/vga/snapshots/v17_candidate/
cp -r /workspace/vga/vga/ /workspace/vga/snapshots/v17_candidate/vga_src/
```

**Step 12.16.2 — Update `DEVIATION_LOG.md`**

Document any known deviations from v17.0 spec during implementation:
- If SVI Pro 2 installation not available: document fallback
- If torchvision optical_flow not available: document cv2 fallback
- If any schema field cannot be populated (e.g., motion_vector for first segment): document default

---

### Phase 12.17 — Tests

**Step 12.17.1 — Create unit tests**

```
unit/test_scene_composition_agent.py     (CGRL-99, CGRL-86)
unit/test_temporal_buffer_manager.py     (CGRL-87, CGRL-88)
unit/test_svi_scheduler.py              (CGRL-89, CGRL-90)
unit/test_motion_state_tracker.py       (CGRL-93)
unit/test_identity_state_tracker.py     (CGRL-96)
unit/test_audio_quality_validator.py    (CGRL-97)
unit/test_cross_modal_alignment_validator.py (CGRL-98)
unit/test_composition_validator.py      (CGRL-86)
unit/test_hrg_controller.py             (CGRL-100; 11-checkpoint tests)
unit/test_immutable_context.py          (5-dim context tests)
```

**Step 12.17.2 — Create integration tests**

```
integration/test_temporal_engine.py         (CGRL-88, CGRL-91, CGRL-92, CGRL-93, CGRL-94)
integration/test_composition_to_image.py    (CGRL-86: S-04 → S-05)
integration/test_identity_cross_phase.py    (CGRL-94, CGRL-95, CGRL-96)
integration/test_hrg_flow.py               (CGRL-100: 11-checkpoint flow)
```

**Step 12.17.3 — Create chaos tests**

```
chaos/test_temporal_buffer_error.py         (CGRL-87)
chaos/test_svi_cfg_violation.py            (CGRL-90)
chaos/test_autoregressive_gate.py          (CGRL-88)
chaos/test_identity_cumulative_drift.py    (CGRL-96)
chaos/test_composition_plan_missing.py     (CGRL-86)
chaos/test_audio_quality_retry.py          (CGRL-97)
chaos/test_execute_stage_contract.py       (CGRL-85)
chaos/test_identity_reference_immutability.py (CGRL-95)
```

---

### Phase 12.18 — Temporal Loop Integration Test (v17.1 — CRITICAL NEW PHASE)

> **This phase is MANDATORY. It validates the full autoregressive loop contract before the system is considered complete. No deployment may occur without passing all tests in this phase.**

**Step 12.18.1 — Temporal Loop Integration Test Suite**

```
File: tests/integration/test_temporal_loop_integration.py

Purpose:
  Validate the complete autoregressive temporal generation contract
  as defined in TEMPORAL EXECUTION CONTRACT v17.1.

Test Cases (ALL must pass):

  test_temporal_loop_generates_minimum_three_segments():
    Given:  A scene plan with >= 3 segments
    When:   TemporalEngine.generate_scene() is called
    Then:
      - Exactly N segments returned (N = len(scene_plan))
      - segments[0] was generated by Wan2.2 (assert via metadata)
      - segments[1:] were generated by SVI (assert via metadata)
      - No segment was generated without a valid 5-frame buffer

  test_temporal_buffer_size_always_equals_five():
    Given:  A scene with 5 segments
    When:   Each segment is generated sequentially
    Then:
      - Buffer size == 5 at every checkpoint (init, update×4)
      - TemporalBufferRecord.frame_count == 5 for all 5 records
      - No buffer update produces frame_count != 5

  test_buffer_correctly_tracks_segment_continuity():
    Given:  Segments 1 through N
    When:   buffer is updated after each segment
    Then:
      - buffer.frames[4] (most recent) matches last frame of segment_n
      - buffer.frames[0] (oldest) matches 5th-from-last frame of segment_n
      - All 5 frames drawn from segment_n only (not prior segments)

  test_continuity_score_above_threshold():
    Given:  3+ generated segments
    When:   ContinuityValidationAgent.score() is called
    Then:
      - continuity_score >= CONTINUITY_THRESHOLD (0.90)
      - identity_per_segment: all values >= CLIP_IDENTITY_THRESHOLD (0.93)

  test_identity_preserved_across_all_segments():
    Given:  char_identity_ref frozen after S-07
    When:   CLIPValidator.score() called on each video segment keyframe
    Then:
      - All scores >= 0.93
      - IdentityStateRecord.threshold_exceeded == False for all records
      - Same char_identity_ref used across all segments (immutability check)

  test_single_call_generation_raises_error():
    Given:  A caller that tries to call generate_all_segments() or equivalent
    When:   The call is made
    Then:   AutoregressiveViolationError is raised immediately

  test_single_image_conditioning_raises_error():
    Given:  SVI wrapper called with init_image (single PIL.Image)
    When:   The call is made
    Then:   AutoregressiveViolationError is raised (signature-level enforcement)

  test_temporal_buffer_error_halts_pipeline():
    Given:  A TemporalBuffer with frame_count = 3 (simulated corruption)
    When:   TemporalEngine tries to generate next segment
    Then:
      - TemporalBufferError raised
      - CriticalPipelineError propagated by MasterOrchestrator
      - Pipeline halts cleanly (no partial segment written)

  test_buffer_is_cpu_resident_between_segments():
    Given:  A running temporal loop
    When:   Buffer state is inspected between segment generations
    Then:
      - buffer.frames.device == "cpu" at all inter-segment checkpoints
      - GPU tensors only exist inside TemporalBufferManager.encode()

Acceptance Criteria:
  ALL 9 test cases MUST pass with 0 failures.
  Test run MUST complete in <= 600 seconds on RTX 4090.
  If any test fails: development is BLOCKED until fixed.
  Failure is not a warning — it is a HALT condition.
```

### Phase 12.X — Temporal Loop Integration Test (NEW v17.2 — MANDATORY CERTIFICATION)

```
PHASE 12.X: Temporal Loop Integration Test

Objective:
  Validate that TemporalEngine behaves strictly as an autoregressive system.
  This test MUST pass before any pipeline run is considered deployable.

Test Procedure:

  1. Generate ≥ 3 segments from a single scene using TemporalEngine.

  2. For each generated segment n ≥ 2, verify:

     a. Buffer integrity BEFORE generation:
        assert len(buffer_before_segment_n.frames) == 5
        assert buffer_before_segment_n.frame_count == 5
        assert buffer_before_segment_n.device == "cpu"

     b. Dependency verification (autoregressive chain):
        Segment_{n+1} MUST have been generated conditioned on Segment_n buffer.
        Verify by checking SVIGenerationRecord.previous_segment_id == segment_n.id
        Verify by checking SVIGenerationRecord.source_buffer_frame_ids has exactly 5 IDs

     c. Continuity score:
        continuity_score for each segment boundary ≥ SEGMENT_CONTINUITY_MIN (0.85)

     d. Identity score:
        CLIP_similarity(segment_n.keyframe, char_identity_ref) ≥ 0.93

  3. Ensure NO forbidden pattern was used:
        - No segment generated independently (check SVIGenerationRecord.previous_segment_id is never None)
        - No merging (check no merge_segments call in temporal_engine.py execution trace)
        - No buffer corruption (all TemporalBufferRecord.frame_count == 5)

  4. Verify CPU residency rule:
        buffer.frames.device == "cpu" at all inter-segment checkpoints
        GPU tensors only exist inside TemporalBufferManager.encode()

Failure Conditions (any = FAIL → block deployment):
  - buffer size mismatch (frame_count != 5)
  - segment independence detected (missing previous_segment_id)
  - continuity_score < SEGMENT_CONTINUITY_MIN
  - identity_score (CLIP) < 0.93
  - buffer corruption (resolution mismatch, dtype mismatch)
  - any forbidden pattern detected in execution trace

Pass Criteria:
  ALL checks MUST pass with 0 failures.
  Test MUST complete in ≤ 600 seconds on RTX 4090.
  PASS → TemporalEngine CERTIFIED for this pipeline run
  FAIL → Deployment BLOCKED; development is HALTED until fixed

File: tests/integration/test_temporal_loop_integration.py
```

---

**Step 12.18.2 — Temporal Loop Regression Guard**

```
File: tests/regression/test_temporal_regression_guard.py

Purpose:
  Ensure that future changes to temporal_engine.py, temporal_buffer_manager.py,
  or svi_wrapper.py do NOT silently break the autoregressive contract.

Required checks:
  - temporal_engine.py must contain a for loop (not a comprehension or map)
    over scene_plan[1:] — regex check on AST
  - svi_wrapper.generate() signature must have init_latents: torch.Tensor
    (not init_image: PIL.Image) — AST check
  - temporal_buffer_manager.update() must contain assert frames.shape[0] == TEMPORAL_BUFFER_SIZE
  - TemporalBufferManager.encode() must call .to("cpu") before returning

  Failure of any check: raise RegressionGuardError with file and line reference.
```

---

## Phase 12 Build Order Summary

```
12.0.1  settings.py — v17.0 constants
12.0.2  schemas.py — v17.0 schemas
12.0.3  exceptions.py — v17.0 exceptions

12.1.1  immutable_context.py — 5-dim update
12.1.2  context_factory.py — 5-dim init

12.2.1  storage.py — v17.0 paths

12.3.1  validation/composition_validator.py
12.3.2  validation/audio_quality_validator.py
12.3.3  validation/cross_modal_alignment_validator.py

12.4.1  config/prompts/composition_prompts.py

12.5.1  temporal/temporal_buffer_manager.py
12.5.2  temporal/svi_scheduler.py
12.5.3  temporal/motion_state_tracker.py
12.5.4  temporal/temporal_retry_controller.py
12.5.5  models/wrappers/svi_wrapper.py (update)

12.6.1  identity/identity_state_tracker.py   ← BEFORE temporal_engine

12.5.6  temporal/temporal_engine.py           ← AFTER identity_state_tracker

12.7.1  agents/scene_composition_agent.py
12.8.1  agents/video_segment_generator.py + wan_wrapper.py update

12.9.1  agents/base_image_agent.py (update)
12.9.2  agents/scene_expansion_agent.py (update)
12.9.3  agents/image_refinement_agent.py (update)
12.9.4  agents/continuity_validation_agent.py (update)
12.9.5  agents/lip_sync_agent.py (update)
12.9.6  agents/audio_mixing_agent.py (update)
12.9.7  agents/quality_agent.py (update)

12.10.1 core/hrg_controller.py (update to 11 checkpoints)
12.11.1 core/master_orchestrator.py (execute_stage contract)

12.12.1 api/routes/temporal.py
12.12.2 api/routes/identity.py
12.12.3 api/routes/audio.py
12.12.4 api/routes/composition.py
12.12.5 api/routes/hrg.py (update)
12.12.6 api/main.py (update)

12.13.1 ui/components/hrg_panels/hrg_2_scene_plan.py
12.13.2 ui/components/hrg_panels/hrg_4_composition.py
12.13.3 ui/components/temporal_engine_panel.py
12.13.4–6 update hrg_8, hrg_10, hrg_11 panels
12.13.7 renumber all remaining HRG panels

12.14.1 core/schema_migrations.py (update)
12.15.1 bootstrap.py (steps 6Z-p through 6Z-z)
12.16.1 snapshots/v17_candidate/
12.16.2 DEVIATION_LOG.md

12.17.1 unit tests (all v17.1)
12.17.2 integration tests (v17.1)
12.17.3 chaos tests (v17.1)

12.18.1 temporal loop integration tests (v17.1 — MANDATORY)
12.18.2 temporal loop regression guard (v17.1 — MANDATORY)

12.19  v17.2 additions (NEW):
12.19.1 temporal/temporal_authority_guard.py
12.19.2 validation/cross_modal_validation_unified.py
12.19.3 validation/system_certification_validator.py
12.19.4 Update quality_agent.py to call SystemCertificationValidator
12.19.5 Update temporal_engine.py to call TemporalAuthorityGuard
12.19.6 Update svi_wrapper.py to call TemporalAuthorityGuard.guard_svi_invoke()
12.19.7 Update all agents calling cross-modal validation to use unified contract
12.19.8 Phase 12.X temporal loop integration test (MANDATORY certification)
12.19.9 Update schemas.py with CrossModalValidationContract + SystemCertificationFailureError
```
