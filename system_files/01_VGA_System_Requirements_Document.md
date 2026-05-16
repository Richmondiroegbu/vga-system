# VGA System Requirements Document
**Project:** Video Generation Automation (VGA) — Cinematic AI Video Production Engine
**Version:** 17.2.0
**Status:** Authoritative Reference
**Audience:** System Architects, Pipeline Engineers, Claude Code Agent

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Stakeholder Requirements](#2-stakeholder-requirements)
3. [Functional Requirements](#3-functional-requirements)
4. [Non-Functional Requirements](#4-non-functional-requirements)
5. [System Constraints](#5-system-constraints)
6. [System Execution Constraints](#6-system-execution-constraints)
7. [External Interface Requirements](#7-external-interface-requirements)
8. [AI Model Requirements](#8-ai-model-requirements)
9. [Data Requirements](#9-data-requirements)
10. [Security & Access Requirements](#10-security--access-requirements)
11. [Acceptance Criteria](#11-acceptance-criteria)
12. [GPU Pod Lifecycle Guarantee](#12-gpu-pod-lifecycle-guarantee)
13. [Storage & Cost Constraints](#13-storage--cost-constraints)
14. [Requirement → Enforcement Mapping](#14-requirement--enforcement-mapping)
15. [Failure Severity Levels](#15-failure-severity-levels)
16. [System SLA Guarantees](#16-system-sla-guarantees)
17. [Glossary](#17-glossary)

---

## 1. Project Overview

### 1.1 Purpose

The VGA system is a fully automated, end-to-end AI video production engine with **mandatory human-in-the-loop control at every consequential stage**. Given a single natural-language prompt, the system produces a complete cinematic-quality video through a deterministic, identity-consistent, self-improving, fully resilient, and measurably guaranteed pipeline.

**Retained from v16.0 — All architecture fully preserved:**
All SLA enforcement, adaptive calibration loop, smart gating, immutable context system, dev safety tooling, all prior layers through v14.1, and all RULE-21 through RULE-85 and FR-600 through FR-834 are **retained without modification**.

**New in v17.0 — Full v17 Specification Upgrade (16 Stages → 16 Stages, fully hardened):**

The v16.0 system defined a robust 15-stage pipeline but had the following architectural gaps addressed in v17.0:

1. **System Directive v17** — Formal deterministic, stateful, human-governed execution contract added to the system core
2. **Scene Composition Engine (S-04, NEW)** — `SceneCompositionAgent` inserted between IdentityDesign and BaseImageGeneration; produces mandatory `CompositionPlan` before any image generation
3. **Temporal Engine Architecture (FULL HARDENING)** — `TemporalEngine` decomposed into: `TemporalBufferManager`, `SVIScheduler` (noise-aware LoRA scheduling), `MotionStateTracker`, `ContinuityValidator`, `TemporalRetryController`
4. **TemporalBuffer enforced** — Rolling 5-frame buffer (`BUFFER_SIZE = 5` strict) with typed `TemporalBuffer` dataclass; single-frame conditioning forbidden
5. **Autoregressive generation loop** — Segment[n+1] MUST be conditioned on Segment[n]; batch segment merging forbidden
6. **Noise-aware LoRA Scheduler for SVI** — Dynamic LoRA weight per diffusion timestep (high/mid/low noise phases); static LoRA weight forbidden in temporal generation
7. **SVI CFG and Steps Contract** — CFG clamped to [5.0, 6.0]; steps dynamic by criticality (30–50 critical / 4–8 standard)
8. **Identity Directive v17** — Identity is a persistent system state constructed, reinforced, validated, and preserved across ALL phases (image + video + lip sync)
9. **Cross-phase identity propagation** — Identity validated in image stages, video segments, and lip-synced frames
10. **IdentityState tracker** — `IdentityState` dataclass tracking embedding vector, drift score, and history; cumulative drift triggers full regeneration
11. **Audio Directive v17** — Audio is temporally aligned, identity-safe, and mixed with deterministic priority
12. **Cross-modal validation** — Video ↔ Audio duration alignment; Video ↔ Lip Sync phoneme + identity; Identity global reference consistency
13. **HRG expanded to 11 checkpoints** — HRG-4 (Composition) and HRG-10 (updated Lip Sync QA) formally specified; all 11 mapped to stages
14. **Master Orchestrator execution contract** — `execute_stage()` pattern enforced: SystemGuard → validate_previous → HRG → run → validate_output → context.evolve()
15. **Context propagation contract** — Every stage MUST update: `identity_state`, `motion_state`, `camera_state`, `lighting_state`, `temporal_state` via `context.evolve()`
16. **Runtime/Deployment model hardened** — Only ONE heavy model loaded at a time enforced; sequential model lifecycle documented
17. **SNR and clipping validation** — SNR ≥ 10 dB and no clipping (peaks ≤ 0 dBFS) formally required at AudioMixingAgent
18. **RULE-86 through RULE-99** — 14 new mandatory rules covering temporal buffer, autoregressive generation, composition requirement, identity persistence, validation-before-progression, base generation purity, identity lock threshold, drift budget, multi-pass identity construction, cross-stage identity propagation, timing contract, lip sync identity guard, mixing priority, and minimum clarity

**Schema version advances to v6.0 for all new v17.0 artifacts.**

### 1.2 Mission Statement

> *Inspire audiences by telling stories of people who overcame adversity — restoring hope and faith.*

Every system design decision shall be evaluated against this mission. Long-form cinematic continuity (10–30 second scenes), stable character identity, evolving motion, emotional depth, guaranteed pipeline resilience, measurable quality guarantees, and self-improving runtime behavior are non-negotiable.

### 1.3 Lifecycle Philosophy

All v16.0 lifecycle principles retained. v17.0 additions:

| Principle | Implementation |
|---|---|
| **System is deterministic and stateful** | SYSTEM DIRECTIVE v17: sequential, state-aware, validated at every boundary, interruptible, recoverable |
| **Composition precedes all image generation** | SceneCompositionAgent (S-04) MUST run before BaseImageAgent; CompositionPlan is required input |
| **Temporal generation is autoregressive** | Segment[n+1] MUST be conditioned on Segment[n]; TemporalBuffer(5 frames) is mandatory |
| **Identity is a persistent state, not an output** | IDENTITY DIRECTIVE v17: constructed → reinforced → validated → preserved → tracked across all phases |
| **Audio is synchronised by contract** | AUDIO DIRECTIVE v17: timing validation, identity-safe lip sync, deterministic priority mixing |
| **Context evolves forward at every stage** | `context.evolve()` called after every stage; all five state dimensions updated |

### 1.4 Execution Model (v17.0)

All v16.0 execution model retained. v17.0 restructures and hardens the pipeline with the approved workflow:

```
PHASE 1 — Narrative Intelligence (CPU/LLM)
------------------------------------------
S-01  ScriptAgent (Qwen2.5-14B)
      → ScriptSchema [v6.0]
      → SchemaValidationGate
      → HRG-1

S-02  ScenePlanner + SegmentPlanner
      → ScenePlan + SegmentPlan
      → Duration enforcement: scene 10–30s / segment 3–5s
      → ImmutableContextSystem initialised
      → HRG-2

S-03  IdentityDesignAgent
      → character_identity, environment_description, reference_strategy (MANDATORY)
      → IdentityDesignSchema [v6.0]
      → HRG-3

S-04  SceneCompositionAgent (NEW v17.0)
      → CompositionPlan: camera_angle, camera_motion, character_positions,
        focus_subject, lighting_style, motion_vector
      → CompositionPlan is MANDATORY input for all image generation
      → HRG-4 (NEW)

------------------------------------------
PHASE 2 — Visual Grounding (GPU: Image Pipeline)
------------------------------------------
S-05  BaseImageAgent (FLUX.2-klein 4B — NO LoRA)
      → 6 Base Images (angle/lighting/pose diversity required)
      → CLIP identity score ≥ 0.93 per image
      → CompositionPlan MUST be input (RULE-88)
      → HRG-5

S-06  IdentityReinforcementLoop (FLUX.2-klein 4B + Consistance_Edit_LoRA)
      → Sub-stage 6A: Multi-Angle Expansion (min 5–8 angle variants)
      → Sub-stage 6B: Merge / Edit (identity-stabilised master image)
      → Sub-stage 6C: Scene Expansion (bound identity to CompositionPlan)
      → CLIP ≥ 0.93 enforced after EACH sub-stage
      → HRG-6

S-07  ImageRefinementAgent (Z-Image-Turbo)
      → denoise ∈ [0.05, 0.15]; cfg = 5.0
      → drift ≤ 0.02; CLIP ≥ 0.93
      → HRG-7

------------------------------------------
PHASE 2B — Hard GPU Cleanup
------------------------------------------
      model_manager.unload_all()
      gc + cuda_empty + sleep(3)
      ASSERT: free_ratio ≥ 0.90

------------------------------------------
PHASE 3 — Motion + Continuity (GPU: Video)
------------------------------------------
S-08  VideoSegmentGenerator (Wan2.2-I2V-A14B-FP8)
      → Segment_1 generated from refined image (init_image)

S-09  TemporalEngine (SVI Pro 2 — NEW CORE ARCHITECTURE)
      → TemporalBufferManager: rolling 5-frame buffer (BUFFER_SIZE=5 STRICT)
      → SVIScheduler: noise-aware LoRA scheduling per diffusion phase
      → MotionStateTracker: velocity, direction, magnitude
      → Autoregressive chaining: Segment[n+1] conditioned on Segment[n]
      → ContinuityValidator (per-segment)
      → TemporalRetryController

S-10  ContinuityValidationAgent
      → continuity_score ≥ 0.90 (0.40×motion + 0.30×lighting + 0.30×identity)
      → Identity validated per video segment (RULE-89)
      → HRG-8

------------------------------------------
PHASE 4 — Audio Realism
------------------------------------------
S-11  DialogueAgent (CosyVoice3-0.5B)
      → Segment-aligned speech; timing error ≤ 0.10s
      → HRG-9

S-12  LipSyncAgent (LatentSync-1.6)
      → phoneme_alignment ≥ 0.80
      → identity_delta ≤ 0.03
      → LipSyncValidation (RULE-89: identity validated after lip sync)
      → HRG-10

S-13  AmbientAudioAgent (MMAudio)
      → Per-scene environmental sound

S-14  MusicAgent (MusicGen-medium)
      → Per-scene background music

S-15  AudioMixingAgent (pydub / torchaudio)
      → Dialogue (0 dB) > Ambient (−12 dB) > Music (−18 dB)
      → Ducking: ambient/music −6 dB during dialogue
      → AudioValidation: SNR ≥ 10 dB; peaks ≤ 0 dBFS (no clipping)
      → HRG-11

------------------------------------------
PHASE 5 — Finalization
------------------------------------------
S-16  ExportAgent
      → AssemblyAgent (ffmpeg merge)
      → ExportAgent (/workspace/output/{job_id}/{scene_id}/)
      → QualityAgent (PipelineReport + SLA + adaptive snapshots)
```

### 1.5 Scope

| In Scope | Out of Scope |
|---|---|
| All v16.0 in-scope items retained | All v16.0 out-of-scope items retained |
| **S-04: SceneCompositionAgent producing CompositionPlan (NEW v17.0)** | |
| **TemporalEngine full architecture: TemporalBufferManager, SVIScheduler, MotionStateTracker (NEW v17.0)** | |
| **TemporalBuffer(5 frames) mandatory; autoregressive generation loop (NEW v17.0)** | |
| **Noise-aware SVI LoRA scheduling per diffusion timestep (NEW v17.0)** | |
| **IdentityState dataclass tracking embedding + drift + history (NEW v17.0)** | |
| **Cross-phase identity validation: image + video + lip sync (NEW v17.0)** | |
| **Cross-modal validation: Video ↔ Audio ↔ Identity (NEW v17.0)** | |
| **11 HRG checkpoints (HRG-1 through HRG-11) (NEW v17.0)** | |
| **Audio validation: SNR ≥ 10 dB + no clipping (peaks ≤ 0 dBFS) (NEW v17.0)** | |
| **context.evolve() at every stage with 5-dimensional state (NEW v17.0)** | |
| **RULE-86 through RULE-99 enforcement (NEW v17.0)** | |

---

## 2. Stakeholder Requirements

### 2.1 All v16.0 Stakeholder Requirements Retained (SR-001 through SR-046)

v17.0 additions:

| ID | Requirement |
|---|---|
| SR-047 | User reviews CompositionPlan (camera, blocking, motion) at HRG-4 before any image generation begins |
| SR-048 | User can edit CompositionPlan fields at HRG-4 including camera_angle, camera_motion, character_positions, lighting_style |
| SR-049 | User reviews video segments with continuity score at HRG-8 with sub-score breakdown (motion/lighting/identity) |
| SR-050 | User reviews lip sync at HRG-10 with identity delta and phoneme alignment scores |
| SR-051 | User reviews final audio mix at HRG-11 with SNR and clipping validation results |
| SR-052 | All 11 HRG checkpoint decisions are logged with user, action, and timestamp |

### 2.2 Operator Requirements

All OR-001 through OR-032 retained. v17.0 additions:

| ID | Requirement |
|---|---|
| OR-033 | CompositionPlan MUST be written to `composition_plan_{scene_id}.json` after S-04 |
| OR-034 | TemporalBuffer state MUST be logged after each segment generation (frame count, scene_id, timestamp) |
| OR-035 | MotionState MUST be logged per segment: velocity_vector, direction, magnitude |
| OR-036 | IdentityState cumulative drift MUST be logged per stage transition |
| OR-037 | SNR and clipping validation results MUST be included in AudioMixReport |
| OR-038 | All 11 HRG decisions MUST be written to `hrg_log_{job_id}.json` |

---

## 3. Functional Requirements

### 3.1–3.65: All v16.0 Functional Requirements Retained (FR-001 through FR-834)

### 3.66 Stage 4 — Scene Composition Engine (NEW v17.0)

| ID | Requirement | Priority |
|---|---|---|
| FR-900 | System MUST implement `SceneCompositionAgent` in `vga/agents/scene_composition_agent.py` | MUST |
| FR-901 | `SceneCompositionAgent` MUST accept: dialogue, emotion, motion_intent, characters, environment | MUST |
| FR-902 | `SceneCompositionAgent` MUST output `CompositionPlan` with ALL fields: camera_angle, camera_motion, character_positions, focus_subject, lighting_style, motion_vector | MUST |
| FR-903 | `CompositionPlan` MUST be validated against `CompositionPlanSchema` before HRG-4 | MUST |
| FR-904 | HRG-4 MUST display full CompositionPlan for human review and editing | MUST |
| FR-905 | No image generation stage (S-05, S-06, S-07) MAY execute without a valid CompositionPlan (RULE-88) | MUST |
| FR-906 | CompositionPlan MUST be propagated to FLUX wrapper (for image generation) and Wan2.2 wrapper (for video generation) | MUST |

### 3.67 TemporalEngine Architecture (NEW v17.0)

| ID | Requirement | Priority |
|---|---|---|
| FR-910 | System MUST implement `TemporalEngine` as a dedicated subsystem in `vga/temporal/temporal_engine.py` | MUST |
| FR-911 | `TemporalEngine` MUST contain: `TemporalBufferManager`, `SVIScheduler`, `MotionStateTracker`, `ContinuityValidator`, `TemporalRetryController` | MUST |
| FR-912 | `TemporalBufferManager` MUST maintain a rolling buffer of exactly 5 frames (BUFFER_SIZE=5 STRICT; RULE-86) | MUST |
| FR-913 | `TemporalBuffer` MUST be a typed dataclass: `frames: Tensor (5,C,H,W)`, `timestamps: List[float]`, `motion_vector: Optional[Tensor]`, `scene_id: str` | MUST |
| FR-914 | TemporalBuffer frames MUST be: same resolution, same color space, normalized identically | MUST |
| FR-915 | If `buffer.frame_count < 5` → TemporalEngine MUST NOT run segment generation (RULE-86) | MUST |
| FR-916 | Each segment MUST be generated conditioned on previous segment's TemporalBuffer (RULE-87); single-frame conditioning is forbidden | MUST |
| FR-917 | Batch segment merging is FORBIDDEN; segments must be generated one-by-one autoregressively (RULE-87) | MUST |
| FR-918 | TemporalBuffer MUST be updated after every segment: `extract_last_5_frames(new_segment)` | MUST |
| FR-919 | Segment_1 MUST be generated by Wan2.2 from refined image `init_image`; TemporalBuffer initialized from Segment_1 | MUST |
| FR-920 | Segment[n+1] (n ≥ 1) MUST be generated by SVI using multi-frame latent conditioning from TemporalBuffer (NOT single-image conditioning) | MUST |

### 3.68 SVIScheduler — Noise-Aware LoRA Scheduling (NEW v17.0)

| ID | Requirement | Priority |
|---|---|---|
| FR-930 | System MUST implement `SVIScheduler` in `vga/temporal/svi_scheduler.py` | MUST |
| FR-931 | `SVIScheduler` MUST apply dynamic LoRA weights per diffusion timestep phase (RULE-86) | MUST |
| FR-932 | High-noise phase (t > threshold_high): LoRA weight = 0.6 (structure + motion) | MUST |
| FR-933 | Mid-noise phase (threshold_mid < t ≤ threshold_high): LoRA weight = 0.5 | MUST |
| FR-934 | Low-noise phase (t ≤ threshold_mid): LoRA weight = 0.4 (detail preservation) | MUST |
| FR-935 | Static LoRA weight in temporal generation is FORBIDDEN (RULE-86) | MUST |
| FR-936 | SVI CFG MUST be clamped to [5.0, 6.0]; CFG > 6.0 is INVALID (color banding risk) | MUST |
| FR-937 | SVI steps MUST be dynamic: 30–50 for critical segments; 4–8 for standard segments | MUST |

### 3.69 MotionStateTracker (NEW v17.0)

| ID | Requirement | Priority |
|---|---|---|
| FR-940 | System MUST implement `MotionStateTracker` in `vga/temporal/motion_state_tracker.py` | MUST |
| FR-941 | `MotionState` MUST be a typed dataclass: `velocity_vector`, `direction`, `magnitude` | MUST |
| FR-942 | `MotionState` MUST be estimated from TemporalBuffer frames via optical flow | MUST |
| FR-943 | `motion_state.velocity_vector` MUST be applied to each segment's prompt/plan to prevent motion reset | MUST |
| FR-944 | MotionState MUST be logged per segment and propagated via `context.evolve()` | MUST |

### 3.70 Context Propagation Contract (NEW v17.0)

| ID | Requirement | Priority |
|---|---|---|
| FR-950 | Every pipeline stage MUST call `context.evolve(output)` after completion | MUST |
| FR-951 | Context MUST track ALL five state dimensions: `identity_state`, `motion_state`, `camera_state`, `lighting_state`, `temporal_state` | MUST |
| FR-952 | Context is IMMUTABLE between stages; `context.evolve()` is the ONLY permitted update mechanism | MUST |
| FR-953 | `IdentityState` MUST track: `embedding_vector`, `drift_score`, `history: List[float]` | MUST |
| FR-954 | If cumulative `IdentityState.drift_score` exceeds threshold → full regeneration triggered | MUST |

### 3.71 Cross-Phase Identity Validation (NEW v17.0)

| ID | Requirement | Priority |
|---|---|---|
| FR-960 | CLIP identity validation MUST be applied in image stages (S-05, S-06, S-07) — retained from v16.0 | MUST |
| FR-961 | CLIP identity validation MUST be applied to video segments after each TemporalEngine generation (RULE-89) | MUST |
| FR-962 | CLIP identity validation MUST be applied to lip-synced frames after LipSyncAgent (RULE-89) | MUST |
| FR-963 | The SAME `char_identity_ref` MUST be used as the CLIP reference embedding across ALL phases | MUST |
| FR-964 | `CLIPValidator` MUST use frozen identity reference embedding — not recomputed per call | MUST |

### 3.72 Audio Directive v17 — Enhanced Validation (NEW v17.0)

| ID | Requirement | Priority |
|---|---|---|
| FR-970 | `AudioMixingAgent` MUST validate SNR ≥ 10 dB after mixing (RULE-99) | MUST |
| FR-971 | `AudioMixingAgent` MUST validate no clipping: peaks ≤ 0 dBFS (RULE-99) | MUST |
| FR-972 | Cross-modal validation MUST verify: `duration(dialogue_segment) ≈ duration(video_segment) ± 0.10s` (RULE-96) | MUST |
| FR-973 | Cross-modal validation MUST verify: segment boundaries are identical between audio and video | MUST |
| FR-974 | Identity reference embedding used for CLIP validation MUST be the same across image, video, and lip sync phases (RULE-95) | MUST |

### 3.73 CompositionPlan Enforcement (NEW v17.0)

| ID | Requirement | Priority |
|---|---|---|
| FR-980 | FLUX wrapper MUST receive `CompositionPlan` fields in prompt construction (RULE-88) | MUST |
| FR-981 | Wan2.2 wrapper MUST receive `CompositionPlan.motion_vector` as generation parameter (RULE-88) | MUST |
| FR-982 | SVI MUST receive `CompositionPlan.camera_motion` for temporal continuation | MUST |
| FR-983 | No image generation or video generation may bypass CompositionPlan (RULE-88) | MUST |

---

## 4. Non-Functional Requirements

### 4.1–4.14: All v16.0 NFRs Retained (NFR-001 through NFR-165)

### 4.15 v17.0 Non-Functional Requirements (NEW)

| ID | Requirement |
|---|---|
| NFR-166 | `SceneCompositionAgent` MUST produce valid `CompositionPlan` in ≤ 15 seconds |
| NFR-167 | `TemporalBufferManager.update()` MUST complete in ≤ 0.5 seconds per segment |
| NFR-168 | `MotionStateTracker.estimate()` MUST complete in ≤ 1.0 second per segment |
| NFR-169 | SVI generation per segment MUST complete in ≤ 120 seconds (standard) or ≤ 300 seconds (critical) |
| NFR-170 | `IdentityState.update()` MUST complete in ≤ 0.2 seconds per frame |
| NFR-171 | Audio SNR/clipping validation MUST complete in ≤ 5 seconds per scene |
| NFR-172 | All 11 HRG checkpoints MUST be non-blocking to pipeline state; timeout 300s triggers safe-halt |

---

## 5. System Constraints

All v16.0 §5.1 through §5.22 retained.

### 5.23 System Directive v17 Constraints (NEW v17.0)

- **SEQUENTIAL EXECUTION**: No stage may skip validation of its predecessor output
- **STATE-AWARE**: Every stage consumes the context evolved by all prior stages
- **VALIDATED AT EVERY BOUNDARY**: Transition from stage N to stage N+1 requires validate_output(N) to pass
- **INTERRUPTIBLE**: Any stage may be interrupted by HRG checkpoints (11 total)
- **RECOVERABLE**: Controlled retry logic applies at every stage (max retries defined per stage)
- No stage MAY execute without validation of prior output
- No stage MAY mutate identity state without CLIP validation
- No stage MAY bypass temporal continuity constraints

### 5.24 TemporalEngine Constraints (NEW v17.0)

- TemporalBuffer MUST contain exactly 5 frames; fewer = TemporalEngine does not run
- Frames in TemporalBuffer MUST be: same resolution (H×W), same color space (RGB/BGR consistent), normalized to same range
- Multi-frame latent conditioning is the ONLY permitted temporal input; single-image input is forbidden for segments ≥ 2
- SVI CFG is bounded: [5.0, 6.0]; no exceptions
- Static LoRA weight during SVI generation is forbidden; SVIScheduler must vary weight per noise phase

### 5.25 CompositionPlan Constraints (NEW v17.0)

- SceneCompositionAgent (S-04) MUST complete before BaseImageAgent (S-05)
- CompositionPlan ALL fields required: camera_angle, camera_motion, character_positions, focus_subject, lighting_style, motion_vector
- CompositionPlan bypassing requires DEVIATION_LOG.md entry and operator approval

### 5.26 Identity State Constraints (NEW v17.0)

- `char_identity_ref` embedding is computed ONCE from the best-scoring base image at S-05
- Embedding is frozen and stored in ImmutableContext; never recomputed downstream
- IdentityState drift history is append-only; no history pruning during active pipeline run
- If cumulative drift exceeds `IDENTITY_CUMULATIVE_DRIFT_THRESHOLD` → full regeneration from current phase start

### 5.27 Audio Cross-Modal Constraints (NEW v17.0)

- SNR ≥ 10 dB is a hard requirement; mixing completes only after SNR validation passes
- No clipping: peaks ≤ 0 dBFS enforced; automatic normalization applied before final write
- Segment boundaries must align within ±0.10s between video segments and audio segments

### 5.28 TEMPORAL ENFORCEMENT BLOCK (NEW v17.2 — MANDATORY ISOLATION)

This block is a dedicated, formally isolated enforcement definition. It supersedes all ambiguous references to temporal generation elsewhere in this document. Every component of this block is a hard system requirement.

```
TEMPORAL ENFORCEMENT BLOCK v17.2:

Temporal generation MUST be implemented as a strict AUTOREGRESSIVE LOOP.
This is a first-class system constraint — not a convention, not a preference.

MANDATORY:
  - Sequential execution: one segment at a time, in strict order
  - Buffer-driven conditioning: every segment conditioned on B_t (exactly 5 frames)
  - Autoregressive chaining: Segment_{n+1} depends ONLY on B_n = last_5_frames(Segment_n)
  - TemporalBuffer ordered as frames[0]=t-4 ... frames[4]=t (most recent)
  - SVI MUST operate ONLY inside the autoregressive loop body

FORBIDDEN (any of these = HARD FAILURE → CriticalPipelineError):
  - Batch segment generation (generating all segments in one call)
  - Segment merging after independent generation
  - Single-frame conditioning for Segment_2 or later
  - Precomputed segment lists passed to TemporalEngine
  - External injection of segments into the autoregressive loop
  - Static LoRA weight during SVI generation (SVIScheduler MUST vary per timestep)

ENFORCEMENT:
  Violation of any item above SHALL terminate pipeline execution immediately.
  ArchitectureGuard is the primary enforcement mechanism.
  TemporalBufferGate is the secondary enforcement mechanism.
  TemporalEngine is the ONLY component allowed to invoke SVI, update TemporalBuffer,
  or control segment iteration.
```

---

## 6. System Execution Constraints

All v16.0 SEC-001 through SEC-058 retained. v17.0 additions:

### SEC-059: Scene Composition Gate (NEW v17.0)

```python
# Before any image generation (S-05, S-06, S-07):
assert composition_plan is not None, "CompositionPlan REQUIRED before image generation"
schema_validator.validate(composition_plan, CompositionPlanSchema)
# On failure: regenerate SceneCompositionAgent output; no image generation until passed
```

### SEC-060: TemporalBuffer Integrity Gate (NEW v17.0)

```python
# Before TemporalEngine generates any segment beyond Segment_1:
assert len(buffer.frames) == TEMPORAL_BUFFER_SIZE  # = 5
assert all(f.shape == buffer.frames[0].shape for f in buffer.frames)  # same resolution
# If buffer < 5: TEMPORAL_ENGINE_MUST_NOT_RUN = True → raise TemporalBufferError
```

### SEC-061: Autoregressive Conditioning Gate (NEW v17.0)

```python
# Before SVI generates Segment[n+1]:
assert n >= 1, "Segment 1 must use Wan2.2 (init_image); not SVI"
latents = temporal_buffer_manager.encode_frames(buffer.frames)  # multi-frame latent
assert latents.shape[0] == TEMPORAL_BUFFER_SIZE  # must be 5-frame tensor
# Single-frame latent: FORBIDDEN
```

### SEC-062: SVI CFG Gate (NEW v17.0)

```python
# In SVIScheduler before every diffusion step:
assert 5.0 <= cfg <= 6.0, f"SVI CFG {cfg} outside [5.0, 6.0]; color banding risk"
```

### SEC-063: Cross-Phase Identity Gate (NEW v17.0)

```python
# After video segment generation (S-09):
clip_video = clip_validator.score(video_keyframe, char_identity_ref)
assert clip_video >= CLIP_IDENTITY_THRESHOLD  # ≥ 0.93; RULE-89

# After lip sync (S-12):
clip_synced = clip_validator.score(synced_frame, char_identity_ref)
delta = abs(clip_synced - clip_stage7_score)
assert delta <= LIPSYNC_IDENTITY_DELTA_THRESHOLD  # ≤ 0.03; RULE-97
```

### SEC-064: Audio Quality Gate (NEW v17.0)

```python
# After AudioMixingAgent completes:
snr = compute_snr(mixed_audio)
assert snr >= MIN_SNR_DB, f"SNR {snr:.1f} dB < required {MIN_SNR_DB} dB"
assert peak_db(mixed_audio) <= 0.0, "Audio clipping detected; peaks exceed 0 dBFS"
```

---

## 7. External Interface Requirements

### 7.1 Streamlit UI Interface (v17.0 additions)

All v16.0 UI requirements retained. v17.0 additions:

**11 HRG Checkpoint Panels (v17.0 — replaces 9-panel spec):**
```
HRG-1: Script Review (unchanged from v16.0)
HRG-2: Scene/Segment Plan Review (NEW v17.0)
  Display: scenes.json + segment_plan.json; durations and segment counts
  Actions: approve, edit durations, trigger replanning
HRG-3: Identity/Environment Design Review (was HRG-2 in v16.0)
HRG-4: Scene Composition Review (NEW v17.0)
  Display: CompositionPlan JSON (camera_angle, camera_motion, character_positions,
           focus_subject, lighting_style, motion_vector)
  Actions: approve, edit fields, trigger recompose
HRG-5: Base Image Review (was HRG-3)
HRG-6: Identity Reinforcement Review (was HRG-4)
HRG-7: Refined Image Review (was HRG-5)
HRG-8: Motion QA Review (was HRG-6 — now includes identity-per-segment score)
HRG-9: Dialogue QA Review (was HRG-7)
HRG-10: Lip Sync QA Review (was HRG-8 — now includes identity delta per segment)
HRG-11: Final Audio QA Review (was HRG-9 — now includes SNR and clipping status)
```

**TemporalEngine Status Panel (NEW v17.0):**
```
Temporal Engine Status:
  Buffer size:           [5/5 frames]
  Segment 1 (Wan2.2):    ✓ [init_image generation]
  Segment 2 (SVI):       ✓ [buffer conditioned] CFG: 5.5 Steps: 40
  Segment 3 (SVI):       ✓ [buffer conditioned] CFG: 5.5 Steps: 30
  Motion State:          velocity: [0.12, 0.03] direction: forward magnitude: 0.12
  Identity per segment:  [0.94, 0.93, 0.95, ...]
```

### 7.2 FastAPI Endpoint Specification (v17.0 additions)

All v16.0 endpoints retained. v17.0 additions:

```
POST /jobs/{job_id}/hrg/HRG-2              → Scene/segment plan review decision
POST /jobs/{job_id}/hrg/HRG-4              → Composition plan review decision
GET  /jobs/{job_id}/temporal/buffer        → Current TemporalBuffer state
GET  /jobs/{job_id}/identity/state         → Current IdentityState (drift, history)
GET  /jobs/{job_id}/audio/validation       → SNR and clipping validation results
```

### 7.3 File System Interface (v17.0 additions)

All v16.0 file system interface retained. v17.0 additions:

```
/workspace/
├── composition/
│   └── {job_id}/
│       └── {scene_id}/
│           └── composition_plan_{scene_id}.json   ← SceneCompositionAgent output
├── temporal/
│   └── {job_id}/
│       └── {scene_id}/
│           ├── temporal_buffer_log.json            ← buffer state per segment
│           └── motion_state_log.json               ← MotionState per segment
├── identity/
│   └── {job_id}/
│       └── identity_state_log.json                 ← cumulative drift history
└── validation/
    └── {job_id}/
        ├── audio_quality_log.json                  ← SNR + clipping per scene
        └── cross_modal_alignment_log.json          ← video ↔ audio alignment
```

---

## 8. AI Model Requirements (v17.0 — Complete Specification)

All v16.0 model requirements retained. v17.0 clarifications and additions:

| Stage | Model | Role | Constraint |
|---|---|---|---|
| S-01 | unsloth/Qwen2.5-14B-Instruct-bnb-4bit | Script generation | ScriptSchema v6.0 |
| S-03 | Qwen (structured generation) | Identity design | reference_strategy required |
| S-04 | Qwen (structured generation) | Scene composition | CompositionPlan ALL fields required |
| S-05 | FLUX.2-klein 4B | Base image generation | No LoRA; CompositionPlan input; 6 images |
| S-06A/B/C | FLUX.2-klein 4B + lrzjason/Consistance_Edit_LoRA | Identity reinforcement | Conditional LoRA [0.4–0.7] |
| S-07 | Tongyi-MAI/Z-Image-Turbo | Image refinement | denoise [0.05–0.15], cfg=5, drift ≤ 0.02 |
| S-08 | nalexand/Wan2.2-I2V-A14B-FP8 | Segment 1 generation | init_image from S-07; CompositionPlan input |
| S-09 | SVI Pro 2 (via TemporalEngine) | Autoregressive continuation | 5-frame buffer; noise-aware LoRA; CFG [5.0–6.0] |
| S-10 | Custom CLIP-based + optical flow | Continuity validation | Score ≥ 0.90; identity per segment |
| S-11 | FunAudioLLM/CosyVoice3-0.5B | Dialogue generation | Segment-aligned; timing ≤ 0.10s |
| S-12 | ByteDance/LatentSync-1.6 | Lip sync | Phoneme ≥ 0.80; identity delta ≤ 0.03 |
| S-13 | MMAudio | Ambient audio | Per-scene; loop-safe |
| S-14 | facebook/musicgen-medium | Background music | Mood/tempo/intensity controlled |
| All image | openai/clip-vit-large-patch14 | Identity enforcement | ≥ 0.93 at every image stage |
| All video | openai/clip-vit-large-patch14 | Identity per segment | ≥ 0.93; RULE-89 |
| Lip sync | openai/clip-vit-large-patch14 | Identity after sync | delta ≤ 0.03; RULE-97 |

---

## 9. Data Requirements

### 9.1–9.26: All v16.0 Data Requirements Retained

### 9.27 v17.0 New Schemas

```python
from pydantic import BaseModel
from typing import List, Optional, Tuple
import torch

class CompositionPlanSchema(BaseModel):
    """Mandatory output of SceneCompositionAgent. RULE-88. schema_version v6.0."""
    scene_id: str
    camera_angle: str          # e.g. "medium shot", "close-up", "wide shot"
    camera_motion: str         # e.g. "slow dolly forward", "static", "pan left"
    character_positions: List[dict]  # [{character_id, position, facing}]
    focus_subject: str         # e.g. "main_character"
    lighting_style: str        # e.g. "low-key dramatic", "soft natural"
    motion_vector: str         # e.g. "forward_slow", "stationary", "right_medium"
    schema_version: str = "v6.0"

class TemporalBufferRecord(BaseModel):
    """Logged after each TemporalBuffer update. OR-034."""
    segment_id: str
    scene_id: str
    frame_count: int           # MUST be 5
    timestamps: List[float]
    scene_id_ref: str
    schema_version: str = "v6.0"

class MotionStateRecord(BaseModel):
    """Logged per segment by MotionStateTracker. OR-035."""
    segment_id: str
    scene_id: str
    velocity_magnitude: float
    direction: str
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
    snr_passed: bool           # snr_db >= MIN_SNR_DB
    clipping_passed: bool      # peak_db <= 0.0
    schema_version: str = "v6.0"

class CrossModalAlignmentRecord(BaseModel):
    """Logged after cross-modal validation. FR-972."""
    scene_id: str
    segment_id: str
    video_duration_s: float
    audio_duration_s: float
    alignment_error_s: float
    within_tolerance: bool
    schema_version: str = "v6.0"
```

---

## 10. Security & Access Requirements (unchanged from v16.0)

---

## 11. Acceptance Criteria

### 11.1–11.5: All v16.0 Acceptance Criteria Retained

### 11.6 v17.0 Acceptance Criteria (NEW)

**Scene Composition:**
- [ ] `SceneCompositionAgent` produces valid `CompositionPlan` with all 6 fields
- [ ] CompositionPlan with missing `camera_angle` fails schema validation and triggers recompose
- [ ] HRG-4 displays full CompositionPlan for editing
- [ ] BaseImageAgent (S-05) raises error if CompositionPlan is None

**TemporalEngine:**
- [ ] TemporalBuffer rejects initialization with fewer than 5 frames
- [ ] TemporalEngine refuses to run if buffer frame count < 5
- [ ] Segment_1 is generated by Wan2.2 from init_image
- [ ] Segment_2+ are generated by SVI using 5-frame multi-frame latent conditioning
- [ ] Single-image input to SVI for Segment_2+ raises TemporalBufferError

**SVIScheduler:**
- [ ] CFG = 6.5 is rejected with assertion error
- [ ] LoRA weight at high-noise phase = 0.6; mid-noise = 0.5; low-noise = 0.4
- [ ] Static LoRA weight assignment raises SVISchedulerViolationError

**Identity Cross-Phase:**
- [ ] CLIP score computed on video segment keyframe using same char_identity_ref
- [ ] CLIP score < 0.93 on video segment triggers segment regeneration
- [ ] Lip sync identity delta > 0.03 triggers lip sync retry
- [ ] Cumulative drift exceeding threshold triggers full phase regeneration

**Audio Quality:**
- [ ] SNR < 10 dB triggers re-mixing
- [ ] Clipping detected (peak > 0 dBFS) triggers normalization and re-mixing
- [ ] AudioQualityRecord written with snr_db and clipping_detected fields

**HRG Expansion:**
- [ ] HRG-4 (Composition) displays CompositionPlan; human can edit camera_angle
- [ ] HRG-8 displays continuity score with identity-per-segment breakdown
- [ ] HRG-10 displays lip sync identity delta alongside phoneme alignment score
- [ ] HRG-11 displays SNR badge and clipping status

---

## 12. GPU Pod Lifecycle Guarantee (unchanged from v16.0)

---

## 13. Storage & Cost Constraints (unchanged from v16.0)

---

## 14. Requirement → Enforcement Mapping

All v16.0 mapping entries retained. v17.0 additions:

| Requirement ID | Description | Enforcement Mechanism | File |
|---|---|---|---|
| FR-900 | SceneCompositionAgent | Qwen structured output; CompositionPlanSchema validation | `agents/scene_composition_agent.py` |
| FR-905 | CompositionPlan required before image gen | SEC-059 gate; assert not None | `agents/base_image_agent.py` |
| FR-912 | TemporalBuffer 5-frame strict | SEC-060 gate; BUFFER_SIZE assertion | `temporal/temporal_buffer_manager.py` |
| FR-916 | Autoregressive generation | SEC-061 gate; multi-frame latent assert | `temporal/temporal_engine.py` |
| FR-931 | Noise-aware LoRA scheduling | SVIScheduler; timestep-conditional weight | `temporal/svi_scheduler.py` |
| FR-936 | SVI CFG [5.0, 6.0] | SEC-062 gate; CFG assert | `temporal/svi_scheduler.py` |
| FR-961 | CLIP per video segment | SEC-063 gate after each segment | `temporal/temporal_engine.py` |
| FR-962 | CLIP after lip sync | SEC-063 gate in lip_sync_agent | `agents/lip_sync_agent.py` |
| FR-970 | SNR ≥ 10 dB | SEC-064 gate; AudioQualityRecord | `agents/audio_mixing_agent.py` |
| FR-971 | No clipping | SEC-064 gate; peak_db ≤ 0.0 assert | `agents/audio_mixing_agent.py` |
| FR-950 | context.evolve() mandatory | Stage execution contract in orchestrator | `temporal/temporal_orchestrator.py` |
| RULE-86 | Temporal buffer enforcement | SEC-060; TemporalBufferManager | `temporal/temporal_buffer_manager.py` |
| RULE-87 | Autoregressive generation | SEC-061; TemporalEngine generate loop | `temporal/temporal_engine.py` |
| RULE-88 | CompositionPlan required | SEC-059; assertion in image wrappers | `agents/scene_composition_agent.py` |
| RULE-89 | Identity in all phases | SEC-063; CLIPValidator in video + lip sync | `temporal/temporal_engine.py`, `agents/lip_sync_agent.py` |
| RULE-90 | Validation before progression | Stage execution contract | `core/master_orchestrator.py` |
| RULE-91 | Base generation purity | LoRAConditionalGate; lora_manager.assert_unloaded() | `agents/base_image_agent.py` |
| RULE-92 | Identity lock ≥ 0.93 | CLIPValidator; CLIP_IDENTITY_THRESHOLD | `validation/clip_validator.py` |
| RULE-93 | Drift budget ≤ 0.02 | CLIP_DRIFT_THRESHOLD in refinement agent | `agents/image_refinement_agent.py` |
| RULE-94 | Multi-pass identity | Sub-stages 6A, 6B, 6C enforced by image_edit_agent | `agents/image_edit_agent.py` |
| RULE-95 | Identity propagation all phases | Frozen char_identity_ref in ImmutableContext | `validation/clip_validator.py` |
| RULE-96 | Timing contract ±0.10s | SEC-057; AudioTimingGate | `agents/dialogue_agent.py` |
| RULE-97 | Lip sync identity guard | delta ≤ 0.03; SEC-063 | `agents/lip_sync_agent.py` |
| RULE-98 | Mixing priority | SEC-058; AudioPriorityGate | `agents/audio_mixing_agent.py` |
| RULE-99 | Minimum clarity SNR | SEC-064; SNR ≥ 10 dB | `agents/audio_mixing_agent.py` |

---

## 15. Failure Severity Levels

### 15.1–15.6: All v16.0 Severity Definitions Retained

### 15.7 v17.0 Failure Severity Classification (NEW)

| Failure | Severity | Rationale |
|---|---|---|
| CompositionPlan missing before image generation | 🔴 CRITICAL | System directive violation; pipeline halts |
| CompositionPlan schema validation failure | 🟡 DEGRADED | Triggers SceneCompositionAgent retry |
| TemporalBuffer frame count < 5 | 🔴 CRITICAL | TemporalEngine cannot run; pipeline halts |
| Single-frame SVI conditioning attempted | 🔴 CRITICAL | Architecture violation; pipeline halts |
| SVI CFG > 6.0 | 🔴 CRITICAL | Config violation; color banding risk; halt |
| Static LoRA weight in SVI | 🟡 DEGRADED | SVIScheduler violation; retry with correct scheduler |
| CLIP < 0.93 on video segment | 🟡 DEGRADED | Triggers segment regeneration (≤ 3 retries) |
| CLIP < 0.93 on lip-synced frame | 🟡 DEGRADED | Triggers lip sync retry |
| Identity delta > 0.03 after lip sync | 🟡 DEGRADED | Triggers lip sync retry with reduced sync_strength |
| Cumulative identity drift exceeded | 🟡 DEGRADED | Triggers full phase regeneration |
| SNR < 10 dB after mixing | 🟡 DEGRADED | Triggers re-mixing with adjusted levels |
| Clipping detected after mixing | 🟡 DEGRADED | Triggers normalization + re-mixing |
| context.evolve() not called after stage | 🔴 CRITICAL | State propagation failure; pipeline integrity risk |
| char_identity_ref recomputed mid-pipeline | 🔴 CRITICAL | Identity consistency violation; halt |
| HRG-4 (Composition) checkpoint skipped | 🔴 CRITICAL | Policy violation; DEVIATION_LOG.md required |

---

## 16. System SLA Guarantees

All v16.0 SLA guarantees retained (§16.1–§16.5).

### 16.6 v17.0 Full-Pipeline SLA Additions (NEW)

```
## 16.6 v17.0 Stage SLA Additions

  ✔ SceneCompositionAgent          ≤ 15 seconds
  ✔ TemporalBuffer update          ≤ 0.5 seconds per segment
  ✔ MotionState estimation         ≤ 1.0 second per segment
  ✔ SVI segment generation         ≤ 120 seconds (standard) / ≤ 300 (critical)
  ✔ IdentityState update           ≤ 0.2 seconds per frame
  ✔ Audio SNR/clipping validation  ≤ 5 seconds per scene

## 16.7 v17.0 Quality Guarantees

  ✔ CompositionPlan all 6 fields   required per scene
  ✔ TemporalBuffer size            = 5 frames exactly
  ✔ SVI CFG                        ∈ [5.0, 6.0]
  ✔ CLIP score per video segment   ≥ 0.93
  ✔ Lip sync identity delta        ≤ 0.03 per segment
  ✔ Audio SNR                      ≥ 10 dB
  ✔ Audio peaks                    ≤ 0 dBFS (no clipping)
  ✔ Autoregressive generation      Segment[n+1] conditioned on Segment[n]
  ✔ Identity reference             Same embedding across all phases
  ✔ All 11 HRG checkpoints         passed per pipeline run
```

---

## 17. Glossary

All v16.0 glossary terms retained. v17.0 additions:

| Term | Definition |
|---|---|
| **SYSTEM DIRECTIVE v17** | Formal contract: VGA operates as deterministic, stateful, human-governed engine; sequential; validated at every boundary; interruptible; recoverable |
| **IDENTITY DIRECTIVE v17** | Character identity is a persistent system state; constructed → reinforced → validated → preserved → tracked across all phases |
| **AUDIO DIRECTIVE v17** | Audio is temporally aligned to video, identity-safe during lip sync, mixed with deterministic priority and measurable clarity |
| **SceneCompositionAgent** | Stage S-04 agent producing CompositionPlan (camera, blocking, motion) from dialogue/emotion/motion_intent; mandatory before image generation |
| **CompositionPlan** | 6-field structured output of SceneCompositionAgent: camera_angle, camera_motion, character_positions, focus_subject, lighting_style, motion_vector |
| **TemporalEngine** | Dedicated subsystem containing TemporalBufferManager, SVIScheduler, MotionStateTracker, ContinuityValidator, TemporalRetryController |
| **TemporalBuffer** | Typed dataclass holding exactly 5 frames (shape: 5,C,H,W), timestamps, motion_vector, scene_id; strict rolling update |
| **TemporalBufferManager** | Component managing TemporalBuffer lifecycle: init from Segment_1, update after each segment, encode for SVI conditioning |
| **SVIScheduler** | Noise-aware LoRA scheduler for SVI; applies dynamic weight (0.6/0.5/0.4) per diffusion phase (high/mid/low noise) |
| **MotionStateTracker** | Estimates velocity_vector, direction, magnitude from TemporalBuffer frames; prevents motion reset between segments |
| **MotionState** | Typed dataclass: velocity_vector, direction, magnitude; propagated via context.evolve() |
| **IdentityState** | Typed dataclass tracking embedding_vector, drift_score, history; cumulative drift triggers regeneration |
| **Autoregressive generation** | Segment[n+1] is generated conditioned on Segment[n] via TemporalBuffer; batch merging forbidden |
| **Multi-frame latent conditioning** | Temporal input to SVI: 5-frame tensor encoded as latents; single-image conditioning forbidden for segments ≥ 2 |
| **Cross-modal validation** | Verifying consistency across video/audio/identity: duration alignment, phoneme alignment, identity reference consistency |
| **char_identity_ref** | Frozen CLIP embedding computed once from best base image; used as identity reference across ALL pipeline phases |
| **context.evolve()** | Immutable context update mechanism; called after every stage; updates identity_state, motion_state, camera_state, lighting_state, temporal_state |
| **SNR** | Signal-to-Noise Ratio; must be ≥ 10 dB in final audio mix; enforced by AudioMixingAgent |
| **RULE-86 through RULE-99** | New mandatory rules in v17.0 governing temporal buffer, autoregressive generation, composition, identity persistence, validation progression, audio quality |
| **schema_version v6.0** | All artifacts written by v17.0 agents carry this version string |
| **HRG-1 through HRG-11** | The 11 mandatory human review checkpoints in v17.0 (expanded from 9 in v16.0) |
| **TemporalEngine Authority** | TemporalEngine is the ONLY component allowed to control segment iteration, update TemporalBuffer, invoke SVI, and manage the autoregressive loop. All other components are FORBIDDEN from these operations. |
| **SYSTEM GUARANTEE v17.2** | Formal guarantee that VGA system produces identity-consistent, temporally-continuous, auditable, deterministic, human-governed outputs. |

---

## 18. Temporal Execution Contract (v17.1 — CRITICAL ADDITION)

### 18.1 TEMPORAL EXECUTION CONTRACT v17.1 (AUTHORITATIVE — GLOBAL ENFORCEMENT)

> **This contract is the highest-priority temporal constraint in the system. It supersedes any ambiguous description of S-09 found elsewhere. Violation of any clause SHALL terminate pipeline execution immediately.**

```
TEMPORAL EXECUTION CONTRACT v17.1:

The Temporal Engine SHALL operate strictly as an AUTOREGRESSIVE LOOP.

Definition:

  Let Segment_1 be generated by Wan2.2 (S-08).

  For all n ≥ 1:
      Segment_{n+1} MUST be generated using a TemporalBuffer B_n
      where B_n = last_5_frames(Segment_n).

Constraints:

  C-1: |B_n| = 5           (fixed buffer size; non-negotiable)
  C-2: Segment_{n+1} = f(B_n, prompt_n)
         where f = SVI Pro 2 diffusion conditioned on B_n as multi-frame latents
  C-3: Batch generation of all segments simultaneously is FORBIDDEN
  C-4: Post-hoc merging of independently generated segments is FORBIDDEN
  C-5: Single-frame conditioning (init_image) for Segment_2+ is FORBIDDEN

  Violation of any constraint SHALL raise CriticalPipelineError and terminate execution.
```

### 18.2 System Type Declaration (v17.1)

The VGA system is formally defined as:

```
System Type:
  - Deterministic
  - Stateful
  - Autoregressive
  - Human-Governed
  - Multi-Agent Cinematic Engine
```

### 18.3 Universal Validation Propagation Rule (v17.1)

Every segment generated anywhere in the pipeline MUST pass ALL THREE validators before
it may be committed to the segment list or used to update the TemporalBuffer:

```
VALIDATION PROPAGATION CONTRACT:

Every segment MUST pass:
  1. ContinuityValidator — continuity_score ≥ SEGMENT_CONTINUITY_MIN
  2. CLIPValidator       — clip_score ≥ CLIP_IDENTITY_THRESHOLD (0.93)  [RULE-89]
  3. TemporalConsistencyValidator — buffer size = 5; no frame resolution mismatch

Failure of ANY validator:
  → TemporalRetryController.retry() invoked
  → If retries exhausted: raise TemporalSegmentFailureError(scene_id, segment_id=n)
  → Pipeline halts for this scene; resume flow initiated
```

### 18.4 Temporal Buffer Lifecycle Contract (v17.1 — Canonical Definition)

```
TemporalBuffer Lifecycle:

  Definition:
    A fixed-size rolling window of exactly 5 frames representing temporal memory.
    It is the sole valid conditioning input for all SVI-generated segments.

  Properties:
    - size = 5 (strict; no deviation permitted)
    - ordered sequence: frames[0]=t-4, frames[1]=t-3, ... frames[4]=t (most recent)
    - all frames: same resolution (H×W)
    - all frames: same color space (consistent RGB or BGR throughout)
    - all frames: normalized identically (same range and pipeline)

  Lifecycle States:

    INIT:
      Called once after Segment_1 generation (S-08).
      buffer = TemporalBufferManager.init(segment_1)
      assert len(buffer.frames) == 5

    UPDATE:
      Called after each SVI segment generation (S-09 loop body).
      buffer = TemporalBufferManager.update(buffer, segment_n)
      assert len(buffer.frames) == 5

    VALIDATION (before each SVI call):
      assert len(buffer.frames) == 5
      assert all frame resolutions match
      → on failure: raise TemporalBufferError; TemporalEngine MUST NOT run

    DEVICE RULE:
      buffer MUST remain CPU-resident between segment generation steps
      buffer moves to GPU ONLY during the TemporalBufferManager.encode() step
      After encode(): tensor immediately transferred back context; GPU memory released

  Hard Rule:
    TemporalBuffer size ≠ 5 at any point → SYSTEM FAILURE → pipeline halts
```


---

## 19. System Guarantee (NEW v17.2)



---

## 20. System Certification Rules (NEW v17.2)

A system build is considered VALID and deployable ONLY when ALL of the following conditions are confirmed:



---

## 21. System Type Declaration (NEW v17.2 — Formal)

The VGA system is formally classified as:


