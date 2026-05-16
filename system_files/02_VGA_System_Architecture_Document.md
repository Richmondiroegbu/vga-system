# VGA System Architecture Document
**Project:** Video Generation Automation (VGA)
**Version:** 17.2.0
**Status:** Authoritative Reference
**Audience:** System Architects, Senior Engineers, Claude Code Agent

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Architectural Principles](#2-architectural-principles)
3. [System Execution Constraints](#3-system-execution-constraints)
4. [System Layers](#4-system-layers)
5. [Component Catalogue](#5-component-catalogue)
6. [Human Control Layer Architecture](#6-human-control-layer-architecture)
7. [Adaptive Layer Architecture](#7-adaptive-layer-architecture)
8. [Identity & Prompt Layer Architecture](#8-identity--prompt-layer-architecture)
9. [Scene Composition Engine Architecture](#9-scene-composition-engine-architecture)
10. [Image Pipeline Architecture](#10-image-pipeline-architecture)
11. [Temporal Engine Architecture](#11-temporal-engine-architecture)
12. [Identity Long-Sequence Hardening Architecture](#12-identity-long-sequence-hardening-architecture)
13. [Audio Pipeline Architecture](#13-audio-pipeline-architecture)
14. [Enforcement Architecture](#14-enforcement-architecture)
15. [Observability Architecture](#15-observability-architecture)
16. [VRAM Management Architecture](#16-vram-management-architecture)
17. [Data Flow Architecture](#17-data-flow-architecture)
18. [GPU Execution Model](#18-gpu-execution-model)
19. [Job Lifecycle Architecture](#19-job-lifecycle-architecture)
20. [Streamlit UI Architecture](#20-streamlit-ui-architecture)
21. [FastAPI Backend Architecture](#21-fastapi-backend-architecture)
22. [Storage Architecture](#22-storage-architecture)
23. [Error & Recovery Architecture](#23-error--recovery-architecture)
24. [Layer Interface Contracts](#24-layer-interface-contracts)
25. [Deployment Architecture (No-Docker)](#25-deployment-architecture-no-docker)
26. [GPU Ownership Rules](#26-gpu-ownership-rules)
27. [Architecture Decision Records](#27-architecture-decision-records)

---

## 1. Architecture Overview

### 1.1 System Identity

VGA is a **sequential multi-agent orchestration system** with:
- Mandatory human-in-the-loop control (11 HRG checkpoints in v17.0)
- Continuous adaptive self-improvement (AdaptiveMemory + CalibrationEngine)
- Identity-consistent cinematic production (CLIP validation at every image + video + lip sync stage)
- Full temporal continuity (Wan2.2 Segment_1 + SVI TemporalEngine autoregressive continuation)
- Scene Composition enforcement (CompositionPlan required before all visual generation)
- Complete resilience and authority layers
- Measurable SLA enforcement
- Closed-loop adaptive calibration
- Smart validation gating
- Immutable state management with 5-dimensional context propagation
- Automated dev safety tooling

The system is deployed as a **single-process Python application on RunPod RTX 4090**. It has **eighteen architectural tiers in v17.0**:

1. **UI Layer** — Streamlit on port 8501; 11 HRG panels, Temporal Engine Panel, SLA monitor, adaptive state
2. **Human Control Layer (HCL)** — State machine, 11 HRG checkpoints, approval enforcement, intervention
3. **Adaptive Layer** — AdaptiveMemory, CalibrationEngine, PerformanceLearner, StrategyOptimizer
4. **Identity & Prompt Layer** — Character identity locking, temporal tracking, deterministic prompt construction
5. **Enforcement Layer** — SystemGuard, ArchitectureGuard, ExecutionAuthorityManager, GatingController
6. **Agent Layer** — All 16 stage agents + 11 HRG checkpoint agents
7. **Scene Composition Layer (NEW v17.0)** — SceneCompositionAgent, CompositionPlanSchema, CompositionPlanValidator
8. **Image Pipeline Layer** — BaseImageAgent, ImageEditAgent (6A/6B/6C), ImageRefinementAgent, CLIPValidator
9. **Performance Layer** — FlashAttention2, dynamic CFG, model wrappers, benchmarking
10. **Temporal Engine Layer (NEW v17.0)** — TemporalBufferManager, SVIScheduler, MotionStateTracker, ContinuityValidator, TemporalRetryController
11. **Long-Sequence Identity Layer** — IdentityTracker, DriftController, LightingNormalizer, TemporalIdentityValidator, IdentityState
12. **Audio Pipeline Layer** — DialogueAgent, LipSyncAgent, AmbientAudioAgent, MusicAgent, AudioMixingAgent, AudioQualityValidator
13. **Observability Layer** — Tracer, Metrics, Logger, Dashboard, AuditSystem, SessionHealthMonitor
14. **Failure Recovery Layer** — FailureClassifier, RetryStrategyEngine, RetryLimiter, SafeFallbackEngine
15. **Resilience Layer** — StabilityManager, MemorySanitizer, AsyncIOManager, ExecutionScheduler
16. **SLA & Measurement Layer** — SLAManager, SLAConfig, violation logging and escalation
17. **Immutable State Layer** — ImmutableContext, ContextFactory, ContextHistory, ContextDiff (5-dimensional state)
18. **Dev Safety Layer** — ArchitectureLinter, RuleChecker, RegressionTester, SnapshotSystem, RollbackManager

### 1.1.1 Temporal Execution Flow Diagram (NEW v17.2)

```
TEMPORAL EXECUTION FLOW (Authoritative Visual — v17.2):

  [S-08: Wan2.2]
       │
       ▼
  Segment_1 generated from init_image + CompositionPlan
       │
       ▼
  Initialize TemporalBuffer B_1 (last 5 frames of Segment_1)
  assert len(B_1.frames) == 5
       │
  ┌────▼────────────────────────────────────────────┐
  │  S-09 AUTOREGRESSIVE LOOP (TemporalEngine)      │
  │                                                  │
  │  for segment_plan in scene_plan[1:]:            │
  │    assert len(buffer.frames) == 5  ← gate       │
  │    latents = encode(buffer)  ← 5-frame tensor   │
  │    assert latents.shape[0] == 5  ← gate         │
  │    cfg = clamp(cfg, 5.0, 6.0)   ← CFG gate     │
  │                                                  │
  │    Segment_{n+1} ← SVI(B_n, prompt_n)           │
  │         │                                        │
  │         ▼                                        │
  │    validate(Segment_{n+1}):                      │
  │      CLIPValidator ≥ 0.93                        │
  │      ContinuityValidator ≥ SEGMENT_CONTINUITY_MIN│
  │      TemporalConsistencyValidator: buffer==5     │
  │         │                                        │
  │         ▼                                        │
  │    B_{n+1} = last_5_frames(Segment_{n+1})        │
  │    assert len(B_{n+1}.frames) == 5  ← gate       │
  │    context = context.evolve(...)                 │
  │                                                  │
  └──────────────────────────────────────────────────┘
       │
       ▼
  All segments validated → S-10: ContinuityValidationAgent

FORBIDDEN at any step:
  - Batch generation of all segments simultaneously
  - Post-hoc merge of independently generated segments
  - Single-frame conditioning for Segment_2+
  - Any component OTHER THAN TemporalEngine controlling this loop
```

### 1.2 New in v17.0

**Scene Composition Layer (NEW v17.0):**
- `SceneCompositionAgent` — Qwen structured output; CompositionPlan (6 fields); mandatory before all image generation
- `CompositionPlanValidator` — validates CompositionPlanSchema before HRG-4

**Temporal Engine Layer (FULL HARDENING v17.0):**
- `TemporalBufferManager` — rolling 5-frame buffer; typed TemporalBuffer dataclass; update after each segment
- `SVIScheduler` — noise-aware LoRA scheduling: 0.6 (high), 0.5 (mid), 0.4 (low) per diffusion phase
- `MotionStateTracker` — optical flow estimation; velocity/direction/magnitude; prevents motion reset
- `ContinuityValidator` — per-segment; integrated into TemporalEngine
- `TemporalRetryController` — error recycling loop; max retries per segment

**Identity Architecture (v17.0 hardening):**
- `IdentityState` dataclass — embedding_vector, drift_score, history; cumulative drift triggers regeneration
- Cross-phase identity validation: image stages + video segments + lip-synced frames
- `char_identity_ref` frozen in ImmutableContext; never recomputed downstream

**Audio Architecture (v17.0 hardening):**
- `AudioQualityValidator` — SNR ≥ 10 dB; no clipping (peaks ≤ 0 dBFS)
- `CrossModalAlignmentValidator` — video ↔ audio duration; segment boundary alignment

**11 HRG Checkpoints formally specified:**
HRG-1 (Script) → HRG-2 (Scene/Segment Plan) → HRG-3 (Identity/Environment) → HRG-4 (Composition — NEW) → HRG-5 (Base Images) → HRG-6 (Identity Reinforcement) → HRG-7 (Refined Images) → HRG-8 (Motion QA) → HRG-9 (Dialogue QA) → HRG-10 (Lip Sync QA) → HRG-11 (Final Audio QA)

### 1.2.1 TemporalEngine Authority (NEW v17.2)

```
TEMPORAL ENGINE AUTHORITY v17.2:

TemporalEngine is the ONLY component authorized to:
  - Control segment iteration (the autoregressive loop)
  - Update TemporalBuffer after each segment generation
  - Invoke SVI Pro 2 for temporal segment generation
  - Manage the full autoregressive generation flow

No other component MAY:
  - Generate Segment_{n+1} outside TemporalEngine
  - Modify TemporalBuffer state directly
  - Bypass TemporalEngine to call SVI directly
  - Inject externally generated segments into the loop

Violation → ArchitectureGuard violation → HARD FAILURE
```

### 1.2.2 Audit System Requirements (NEW v17.2)

```
AUDIT REQUIREMENTS v17.2:

The system MUST log the following for every segment:
  - segment_id
  - buffer_frame_ids (list of 5 frame identifiers)
  - continuity_score
  - identity_score (CLIP)
  - retry_count
  - generation_parameters (cfg, steps, lora_schedule)

All audit logs MUST be:
  - Immutable (append-only; no in-place modification)
  - Timestamped with ISO 8601 format
  - Traceable across the full pipeline (segment_id links all records)

HARD RULE:
  Every segment MUST be fully auditable.
  Missing audit data for any segment = INVALID OUTPUT
  QualityAgent MUST verify audit completeness before writing PipelineReport.
```

### 1.3 System Execution Model (v17.0)

```
[STARTUP]
  [All v16.0 startup steps retained]
  SceneCompositionAgent initialized
  TemporalBufferManager initialized (buffer cleared)
  SVIScheduler initialized (noise thresholds set)
  MotionStateTracker initialized
  AudioQualityValidator initialized
  HRGController initialized with all 11 checkpoint definitions

PHASE 1 — CPU/LLM (Narrative Intelligence)
  S-01: ScriptAgent (Qwen2.5-14B)
    → script.json [ScriptSchema v6.0]
    → HRG-1

  S-02: SceneAgent + SegmentAgent
    → scenes.json + segment_plan.json
    → ImmutableContextSystem initialised
    → HRG-2

  S-03: IdentityDesignAgent (Qwen structured)
    → identity_design.json [reference_strategy REQUIRED]
    → HRG-3

  S-04: SceneCompositionAgent (NEW v17.0)
    → composition_plan.json [CompositionPlan ALL 6 fields]
    → CompositionPlanValidator: schema validated
    → HRG-4 (NEW)

PHASE 2 — GPU (Image Pipeline / Visual Grounding)
  S-05: BaseImageAgent (FLUX.2-klein 4B, NO LoRA)
    → 6 base images (angle/lighting/pose diversity)
    → CompositionPlan MUST be input (RULE-88)
    → CLIPValidator: assert score ≥ 0.93 per image
    → HRG-5

  S-06: IdentityReinforcementLoop (ImageEditAgent)
    6A: MultiAngleAgent (FLUX + LoRA if editing_required)
        → 5–8 angle variants; CLIPValidator ≥ 0.93
    6B: ImageMergeAgent (FLUX + LoRA)
        → identity-stabilised master image; CLIPValidator ≥ 0.93
    6C: SceneExpansionAgent (FLUX + LoRA)
        → CompositionPlan bound; CLIPValidator ≥ 0.93
    → HRG-6

  S-07: ImageRefinementAgent (Z-Image-Turbo)
    → denoise=0.05–0.15, cfg=5.0
    → CLIPValidator: drift ≤ 0.02 AND score ≥ 0.93
    → char_identity_ref FROZEN in ImmutableContext from best base image
    → HRG-7

PHASE 2B — Hard GPU Cleanup
  model_manager.unload_all()
  gc + cuda_empty + sleep(3)
  ASSERT: free_ratio ≥ 0.90

PHASE 3 — GPU (Motion + Continuity)
  S-08: VideoSegmentGenerator (Wan2.2-I2V-A14B-FP8)
    → Segment_1 from refined image (init_image + CompositionPlan)
    → TemporalBuffer initialized from Segment_1

  S-09: TemporalEngine (SVI Pro 2 — Autoregressive)
    → For each subsequent segment:
      TemporalBufferManager provides 5-frame tensor
      SVIScheduler applies noise-aware LoRA (0.6/0.5/0.4)
      MotionStateTracker estimates motion state
      SVI generates via multi-frame latent conditioning
      CLIPValidator: identity per segment ≥ 0.93 (RULE-89)
      TemporalBuffer updated
      context.evolve(segment_output)

  S-10: ContinuityValidationAgent
    → continuity_score ≥ 0.90 (motion×0.40 + lighting×0.30 + identity×0.30)
    → HRG-8

PHASE 4 — Audio Realism
  S-11: DialogueAgent (CosyVoice3-0.5B)
    → timing_error ≤ 0.10s per segment
    → HRG-9

  S-12: LipSyncAgent (LatentSync-1.6)
    → phoneme_alignment ≥ 0.80
    → CLIPValidator: identity_delta ≤ 0.03 (RULE-89, RULE-97)
    → HRG-10

  S-13: AmbientAudioAgent (MMAudio)
  S-14: MusicAgent (MusicGen-medium)

  S-15: AudioMixingAgent (pydub / torchaudio)
    → Dialogue (0 dB) > Ambient (−12 dB) > Music (−18 dB)
    → AudioQualityValidator: SNR ≥ 10 dB; peaks ≤ 0 dBFS (RULE-99)
    → CrossModalAlignmentValidator: video ↔ audio duration
    → HRG-11

PHASE 5 — Finalization
  S-16a: AssemblyAgent → final_video.mp4
  S-16b: ExportAgent → /workspace/output/{job_id}/{scene_id}/
  S-16c: QualityAgent → PipelineReport + SLA + adaptive state
```

### 1.4 Top-Level Architecture Diagram (v17.0)

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                         RUNPOD INSTANCE  RTX 4090                                │
│                    (Native Python — No Docker — Host CUDA)                       │
│                                                                                  │
│  ╔════════════════════════════════════════════════════════════════════════════╗  │
│  ║  STREAMLIT UI (port 8501)                                                  ║  │
│  ║  HRG-1..HRG-11 panels · Temporal Engine Panel · Image Pipeline Panel      ║  │
│  ║  SLA Monitor · Adaptive State · Gating Mode · Session Health              ║  │
│  ╚═══════════════════════════════╤════════════════════════════════════════════╝  │
│                                  │ HTTP only                                     │
│  ╔═══════════════════════════════▼════════════════════════════════════════════╗  │
│  ║  FASTAPI BACKEND (port 8000)                                                ║  │
│  ║  /jobs · /jobs/{id}/hrg/{checkpoint} (11 checkpoints) · /health            ║  │
│  ║  /jobs/{id}/temporal/buffer · /jobs/{id}/identity/state                    ║  │
│  ╚═══════════════════════════════╤════════════════════════════════════════════╝  │
│                                  │                                               │
│  ╔═══════════════════════════════▼════════════════════════════════════════════╗  │
│  ║  HUMAN CONTROL LAYER                                                        ║  │
│  ║  HCL state machine · HRGController (11 checkpoints) · InterventionHandler  ║  │
│  ╚════════════════════════════════════════════════════════════════════════════╝  │
│                                  ↕                                               │
│  ╔════════════════════════════════════════════════════════════════════════════╗  │
│  ║  ENFORCEMENT LAYER                                                          ║  │
│  ║  ArchitectureGuard · SystemGuard · AuthorityManager · GatingController     ║  │
│  ║  CompositionPlanGate · TemporalBufferGate · SVICFGGate                    ║  │
│  ╚════════════════════════════════════════════════════════════════════════════╝  │
│                                  ↕                                               │
│  ╔════════════════════════════════════════════════════════════════════════════╗  │
│  ║  MASTER ORCHESTRATOR                                                        ║  │
│  ║  execute_stage() contract · Context propagation · 16-stage ordering        ║  │
│  ╚═════════╤═══════════════════════════╤══════════════════════════════════════╝  │
│             │                           │                                         │
│  ╔══════════▼═══════════╗  ╔═══════════▼═════════════════════════════════════╗  │
│  ║  SCENE COMPOSITION   ║  ║  TEMPORAL ENGINE LAYER (v17.0)                  ║  │
│  ║  LAYER (v17.0)       ║  ║  TemporalBufferManager (5-frame strict)         ║  │
│  ║  SceneCompositionAgt ║  ║  SVIScheduler (noise-aware: 0.6/0.5/0.4)       ║  │
│  ║  CompositionPlan     ║  ║  MotionStateTracker (velocity/direction)        ║  │
│  ║  6 fields mandatory  ║  ║  ContinuityValidator (per-segment)              ║  │
│  ╚══════════════════════╝  ║  TemporalRetryController                        ║  │
│                             ╚═════════════════════════════════════════════════╝  │
│                                  ↕                                               │
│  ╔════════════════════════════════════════════════════════════════════════════╗  │
│  ║  IMAGE PIPELINE LAYER                                                       ║  │
│  ║  BaseImageAgent (FLUX + CompositionPlan, no LoRA)                          ║  │
│  ║  ImageEditAgent 6A/6B/6C (FLUX + LoRA + CompositionPlan)                  ║  │
│  ║  ImageRefinementAgent (Z-Image-Turbo) · CLIPValidator (shared)             ║  │
│  ╚════════════════════════════════════════════════════════════════════════════╝  │
│                                  ↕                                               │
│  ╔════════════════════════════════════════════════════════════════════════════╗  │
│  ║  AUDIO PIPELINE LAYER                                                       ║  │
│  ║  DialogueAgent (CosyVoice3) · LipSyncAgent (LatentSync-1.6)               ║  │
│  ║  AmbientAudioAgent (MMAudio) · MusicAgent (MusicGen-medium)               ║  │
│  ║  AudioMixingAgent (pydub/torchaudio)                                        ║  │
│  ║  AudioQualityValidator (SNR ≥ 10 dB, peaks ≤ 0 dBFS)                      ║  │
│  ║  CrossModalAlignmentValidator                                                ║  │
│  ╚════════════════════════════════════════════════════════════════════════════╝  │
│                                  ↕                                               │
│  ╔════════════════════════════════════════════════════════════════════════════╗  │
│  ║  LONG-SEQUENCE IDENTITY LAYER                                               ║  │
│  ║  IdentityTracker · DriftController · LightingNormalizer                    ║  │
│  ║  TemporalIdentityValidator · IdentityState (embedding/drift/history)       ║  │
│  ╚════════════════════════════════════════════════════════════════════════════╝  │
│                                  ↕                                               │
│  ╔════════════════════════════════════════════════════════════════════════════╗  │
│  ║  IMMUTABLE STATE LAYER (5-dimensional context)                              ║  │
│  ║  ImmutableContext · identity_state · motion_state · camera_state           ║  │
│  ║  lighting_state · temporal_state · ContextFactory · context.evolve()       ║  │
│  ╚════════════════════════════════════════════════════════════════════════════╝  │
│                                  ↕                                               │
│  ╔════════════════════════════════════════════════════════════════════════════╗  │
│  ║  SLA + ADAPTIVE + FAILURE RECOVERY + OBSERVABILITY + DEV SAFETY LAYERS    ║  │
│  ║  [All v16.0 layers retained unchanged]                                      ║  │
│  ╚════════════════════════════════════════════════════════════════════════════╝  │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Architectural Principles

All v16.0 principles AP-31 through AP-35 retained. v17.0 additions:

**AP-36: Scene composition is the visual contract between narrative and generation.**
CompositionPlan translates dialogue, emotion, and motion intent into camera angles, character blocking, and motion vectors. No visual generation (image or video) may proceed without this plan. The plan is the bridge between what the story needs and what the models produce.

**AP-37: Temporal generation is autoregressive by architecture, not by convention.**
The TemporalEngine enforces that Segment[n+1] is always conditioned on Segment[n] via TemporalBuffer. This is not a preference — it is enforced by the TemporalBufferGate. Single-frame input to SVI is a system violation equivalent to a data corruption.

**AP-38: Identity is a first-class pipeline citizen.**
Identity is not an image output. It is a persistent system state (IdentityState) that is constructed (S-05), reinforced (S-06), validated (S-07), preserved through video segments (S-09), and re-validated after lip sync (S-12). The frozen char_identity_ref embedding is the identity anchor across all phases. Any stage that drifts identity beyond threshold must retry.

**AP-39: Context is the memory of the pipeline.**
The 5-dimensional ImmutableContext (identity_state, motion_state, camera_state, lighting_state, temporal_state) is the only mechanism by which stages share state. Every stage consumes the context evolved by all prior stages, and evolves it for all subsequent stages. Bypassing context.evolve() is a system directive violation.

**AP-40: Audio quality is measurable, not assumed.**
SNR and peak level are not approximated or subjectively evaluated. AudioQualityValidator computes them after every mixing operation. The pipeline cannot export a scene with audio that fails these measurable gates.

---

## 3. System Execution Constraints (v17.0 additions)

All v16.0 constraints retained. v17.0 additions:

- `SceneCompositionAgent.compose()` MUST be called and produce a valid CompositionPlan before S-05 begins
- `CompositionPlan` MUST be passed to FLUX wrapper prompt construction (S-05, S-06)
- `CompositionPlan.motion_vector` MUST be passed to Wan2.2 wrapper (S-08) and SVI wrapper (S-09)
- `TemporalBufferManager.init(segment_1)` MUST be called after Wan2.2 generates Segment_1
- `TemporalBuffer` MUST have exactly 5 frames before ANY SVI call; fewer = TemporalEngine must not run
- `SVIScheduler.apply_lora(timestep)` MUST be called at every diffusion timestep during SVI generation
- `MotionStateTracker.estimate(buffer.frames)` MUST be called before each SVI call
- `CLIPValidator.score(video_keyframe, char_identity_ref)` MUST be called after each SVI segment (RULE-89)
- `CLIPValidator.score(synced_frame, char_identity_ref)` MUST be called after LipSyncAgent (RULE-89)
- `AudioQualityValidator.validate(mixed_audio)` MUST be called after AudioMixingAgent
- `context.evolve(output)` MUST be called after EVERY stage before proceeding to the next stage
- `char_identity_ref` MUST be set in ImmutableContext after S-07 and MUST NOT change thereafter

---

## 4. System Layers (v17.0)

```
Layer  Name                              New in
──────────────────────────────────────────────────────────────────────────────
  1    UI Layer                          v10.0 + v15.0 + v16.0 + v17.0 (11 HRG panels, Temporal panel)
  2    Human Control Layer               v10.0 + v14.1 + v16.0 + v17.0 (11 HRG checkpoints)
  3    Adaptive Layer                    v12.0 + v15.0
  4    Identity & Prompt Layer           v13.0 + v14.0
  5    Enforcement Layer                 v14.0 + v15.0 + v17.0 (Composition/Buffer/CFG gates)
  6    Agent Layer                       v10.0 + v13.0 + v14.0 + v16.0 + v17.0 (16 stage agents)
  7    Scene Composition Layer           v17.0 NEW
  8    Image Pipeline Layer              v16.0 + v17.0 (CompositionPlan integration)
  9    Performance Layer                 v13.0 + v14.1
 10    Temporal Engine Layer             v14.0 + v14.1 + v17.0 (FULL HARDENING)
 11    Long-Sequence Identity Layer      v14.0 + v17.0 (IdentityState dataclass)
 12    Audio Pipeline Layer              v16.0 + v17.0 (AudioQualityValidator)
 13    Observability Layer               v14.0 + v15.0
 14    Failure Recovery Layer            v14.1
 15    Resilience Layer                  v14.1
 16    SLA & Measurement Layer           v15.0
 17    Immutable State Layer             v15.0 + v17.0 (5-dimensional context)
 18    Dev Safety Layer                  v15.0 (dev-time only)
```

---

## 5. Component Catalogue

All v16.0 components retained (§5.1 through §5.17). v17.0 additions:

### 5.18 Scene Composition Layer (NEW v17.0)

| Component | File | Role |
|---|---|---|
| **SceneCompositionAgent** | `vga/agents/scene_composition_agent.py` | S-04: produces CompositionPlan from dialogue/emotion/motion_intent |
| **CompositionPlanValidator** | `vga/validation/composition_validator.py` | validates CompositionPlanSchema; all 6 fields required |
| **CompositionPlanSchema** | `vga/models/schemas.py` | Pydantic schema; camera_angle, camera_motion, character_positions, focus_subject, lighting_style, motion_vector |

### 5.19 Temporal Engine Layer (NEW v17.0 full architecture)

| Component | File | Role |
|---|---|---|
| **TemporalEngine** | `vga/temporal/temporal_engine.py` | Top-level subsystem; autoregressive loop; integrates all sub-components |
| **TemporalBufferManager** | `vga/temporal/temporal_buffer_manager.py` | Manages 5-frame rolling buffer; init, update, encode for SVI |
| **SVIScheduler** | `vga/temporal/svi_scheduler.py` | Noise-aware LoRA scheduling: 0.6/0.5/0.4 per diffusion phase; CFG [5.0–6.0] |
| **MotionStateTracker** | `vga/temporal/motion_state_tracker.py` | Optical flow estimation; velocity/direction/magnitude; propagated via context |
| **TemporalRetryController** | `vga/temporal/temporal_retry_controller.py` | Error recycling loop; max retries per segment with parameter adjustment |

### 5.20 Identity State System (NEW v17.0)

| Component | File | Role |
|---|---|---|
| **IdentityState** | `vga/models/schemas.py` | Typed dataclass: embedding_vector, drift_score, history |
| **IdentityStateTracker** | `vga/identity/identity_state_tracker.py` | Updates IdentityState after each stage; detects cumulative drift |
| **CrossPhaseIdentityValidator** | `vga/validation/cross_phase_identity_validator.py` | CLIP validation across image, video, lip sync phases |

### 5.21 Audio Quality System (NEW v17.0)

| Component | File | Role |
|---|---|---|
| **AudioQualityValidator** | `vga/validation/audio_quality_validator.py` | SNR ≥ 10 dB; peaks ≤ 0 dBFS; logs AudioQualityRecord |
| **CrossModalAlignmentValidator** | `vga/validation/cross_modal_alignment_validator.py` | Video ↔ audio duration; segment boundary alignment |

---

## 6. Human Control Layer Architecture (v17.0 upgrade)

All v16.0 HCL architecture retained. v17.0 — 11 HRG Checkpoints:

```
HRG State Machine (v17.0):

  HRG-1: SCRIPT_REVIEW (unchanged from v16.0)
    gate: no SceneAgent execution without HRG-1 approval

  HRG-2: SCENE_SEGMENT_PLAN_REVIEW (NEW v17.0)
    trigger: SceneAgent + SegmentAgent complete
    display: scenes.json (scene durations) + segment_plan.json (segment timings)
    actions: approve | edit durations | trigger replanning
    pipeline_gate: no IdentityDesignAgent execution without HRG-2 approval

  HRG-3: IDENTITY_DESIGN_REVIEW (was HRG-2 in v16.0)
    gate: no SceneCompositionAgent without HRG-3 approval

  HRG-4: COMPOSITION_REVIEW (NEW v17.0)
    trigger: SceneCompositionAgent completes + CompositionPlanValidator passes
    display: CompositionPlan JSON (all 6 fields editable)
    actions: approve | edit camera_angle | edit motion_vector | trigger recompose
    pipeline_gate: NO image generation without HRG-4 approval (RULE-88)

  HRG-5: BASE_IMAGE_REVIEW (was HRG-3)
    gate: no IdentityReinforcementLoop without HRG-5 approval

  HRG-6: IDENTITY_REINFORCEMENT_REVIEW (was HRG-4)
    gate: no ImageRefinementAgent without HRG-6 approval

  HRG-7: POLISHED_IMAGE_REVIEW (was HRG-5)
    gate: no video generation without HRG-7 approval
    NOTE: char_identity_ref FROZEN in ImmutableContext after HRG-7 approval

  HRG-8: MOTION_QA_REVIEW (was HRG-6)
    display: video segments + continuity score + identity-per-segment scores
    gate: no audio pipeline without HRG-8 approval

  HRG-9: VOICE_QA_REVIEW (was HRG-7)
    gate: no LipSyncAgent without HRG-9 approval

  HRG-10: LIP_SYNC_QA_REVIEW (was HRG-8)
    display: lip-synced video + phoneme score + identity_delta per segment
    gate: no ambient/music generation without HRG-10 approval

  HRG-11: FINAL_AUDIO_QA_REVIEW (was HRG-9)
    display: full scene video + mixed audio + SNR badge + clipping status + level meters
    actions: approve (→ scene lock + export) | trigger remix | adjust levels
    gate: no export without HRG-11 approval
```

---

## 7–8. Adaptive Layer & Identity/Prompt Layer (unchanged from v16.0)

---

## 9. Scene Composition Engine Architecture (NEW v17.0)

```
SCENE COMPOSITION DATA FLOW:

[S-03 output: identity_design.json]
  │
  ↓ HRG-3 approved
  │
[S-04: SceneCompositionAgent]
  │  model: Qwen2.5-14B (structured output mode)
  │  input: {
  │    dialogue: script[scene_id].dialogue,
  │    emotion: script[scene_id].emotion,
  │    motion_intent: script[scene_id].motion_intent,
  │    characters: identity_design.character_identity,
  │    environment: identity_design.environment_description
  │  }
  │  output: CompositionPlan {
  │    camera_angle: str,       # "medium shot" | "close-up" | "wide shot" | ...
  │    camera_motion: str,      # "slow dolly forward" | "static" | "pan left" | ...
  │    character_positions: List[{character_id, position, facing}],
  │    focus_subject: str,      # "main_character" | "environment" | ...
  │    lighting_style: str,     # "low-key dramatic" | "soft natural" | ...
  │    motion_vector: str       # "forward_slow" | "stationary" | "right_medium" | ...
  │  }
  │  validation: CompositionPlanValidator.validate(plan) [all 6 fields required]
  │
  ↓ HRG-4 approved
  │
[CompositionPlan flows to:]
  → S-05 BaseImageAgent: prompt construction (camera_angle, lighting_style)
  → S-06 ImageEditAgent: scene expansion prompt (character_positions, focus_subject)
  → S-08 Wan2.2: generation parameters (motion_vector, camera_motion)
  → S-09 TemporalEngine/SVI: continuation conditioning (motion_vector, camera_motion)
  → ImmutableContext: camera_state, lighting_state (propagated forward)
```

---

## 10. Image Pipeline Architecture (v17.0 — CompositionPlan integrated)

```
IMAGE PIPELINE DATA FLOW (v17.0):

[CompositionPlan (from S-04) + identity_design.json (from S-03)]
  │
  ↓ HRG-4 approved
  │
[S-05: BaseImageAgent]
  │  model: FLUX.2-klein 4B
  │  input: identity_design.json + CompositionPlan (RULE-88)
  │  output: 6 base images (angle/lighting/pose diversity as per reference_strategy)
  │  constraint: LoRA=False, composition_plan required
  │  validation: CLIPValidator.validate(image, char_identity_ref) ≥ 0.93 per image
  │  char_identity_ref: set to embedding of best-scoring base image (frozen in context)
  │
  ↓ HRG-5 approved
  │
[S-06A: MultiAngleAgent]
  │  Minimum 5–8 angle variants; RULE-94
[S-06B: ImageMergeAgent]
  │  Identity-stabilised master image; LoRA weight 0.5–0.6
[S-06C: SceneExpansionAgent]
  │  CompositionPlan binds identity to environment; LoRA weight 0.6
  │  Each sub-stage: CLIPValidator ≥ 0.93
  │
  ↓ HRG-6 approved
  │
[S-07: ImageRefinementAgent]
  │  model: Tongyi-MAI/Z-Image-Turbo
  │  denoise ∈ [0.05, 0.15]; cfg = 5.0
  │  CLIPValidator: drift ≤ 0.02 AND score ≥ 0.93
  │  [char_identity_ref FROZEN here — never recomputed downstream]
  │
  ↓ HRG-7 approved → char_identity_ref locked in ImmutableContext
  │
  ↓ [Phase 2B: FULL GPU CLEANUP]
```

---

## 11. Temporal Engine Architecture (v17.1 — HARDENED FULL SPECIFICATION)

```
TEMPORAL ENGINE DATA FLOW:

[Phase 2B cleanup complete + HRG-7 approved]
  │
  ↓ Load Wan2.2-I2V-A14B-FP8
  │
[S-08: VideoSegmentGenerator (Wan2.2)]
  init_image = load_refined_image(scene_id)  ← S-07 output
  composition_input = context.camera_state + context.motion_state
  segment_1 = wan.generate(
    init_image=init_image,
    prompt=compose_prompt(scene_plan[0], composition_plan),
    cfg=7.0, steps=30
  )
  CLIPValidator.score(segment_1.keyframe, char_identity_ref) ≥ 0.93

  buffer = TemporalBufferManager.init(segment_1)  ← buffer now holds 5 frames
  segments = [segment_1]

  ↓ Swap Wan2.2 → SVI Pro 2 (or unified if VRAM allows)
  │
[S-09: TemporalEngine (SVI Pro 2 — Autoregressive Loop)]
  ┌─────────────────────────────────────────────────────────────────┐
  │  for segment_plan in scene_plan[1:]:                           │
  │                                                                 │
  │    # 1. Estimate motion state from buffer                      │
  │    motion_state = MotionStateTracker.estimate(buffer.frames)   │
  │    context = context.evolve({motion_state: motion_state})      │
  │                                                                 │
  │    # 2. Prepare multi-frame conditioning                       │
  │    latents = TemporalBufferManager.encode(buffer.frames)       │
  │    # latents.shape = (5, C, H/8, W/8) — MULTI-FRAME REQUIRED  │
  │    # Single-frame forbidden (RULE-86)                          │
  │                                                                 │
  │    # 3. Generate with noise-aware LoRA                         │
  │    segment_n = SVI.generate(                                   │
  │      init_latents=latents,                                     │
  │      prompt=compose_temporal_prompt(segment_plan, motion_state)│
  │      lora_scheduler=SVIScheduler,   # noise-aware (RULE-86)   │
  │      cfg=clamp(cfg, 5.0, 6.0),      # CFG gate (RULE-86)     │
  │      steps=dynamic_steps(segment_plan.is_critical)            │
  │    )                                                            │
  │                                                                 │
  │    # 4. Validate identity per segment (RULE-89)               │
  │    score = CLIPValidator.score(segment_n.keyframe,             │
  │                                 char_identity_ref)              │
  │    if score < CLIP_IDENTITY_THRESHOLD:                         │
  │      segment_n = TemporalRetryController.retry(...)            │
  │                                                                 │
  │    # 5. ContinuityValidator (per-segment)                      │
  │    cont_score = ContinuityValidator.score_segment(             │
  │      buffer.last_frame, segment_n.first_frame                  │
  │    )                                                            │
  │                                                                 │
  │    # 6. Update buffer (RULE-86)                                │
  │    buffer = TemporalBufferManager.update(buffer, segment_n)    │
  │    assert len(buffer.frames) == TEMPORAL_BUFFER_SIZE  # = 5   │
  │                                                                 │
  │    segments.append(segment_n)                                  │
  │    context = context.evolve({temporal_state: buffer})          │
  └─────────────────────────────────────────────────────────────────┘

SVIScheduler (noise-aware LoRA):
  def apply_lora(timestep: int, threshold_high: int, threshold_mid: int) -> float:
    if timestep > threshold_high:
      return 0.6   # structure + motion phase
    elif timestep > threshold_mid:
      return 0.5   # transition phase
    else:
      return 0.4   # detail preservation phase
  NOTE: Static weight is FORBIDDEN (RULE-86)

MotionStateTracker:
  def estimate(frames: Tensor) -> MotionState:
    flows = [optical_flow(frames[i], frames[i+1]) for i in range(4)]
    velocity_vector = mean_flow(flows)
    direction = classify_direction(velocity_vector)
    magnitude = norm(velocity_vector)
    return MotionState(velocity_vector, direction, magnitude)

[S-10: ContinuityValidationAgent]
  all_segments = scene.get_all_video_segments()
  S_continuity = 0.40*S_motion + 0.30*S_lighting + 0.30*S_identity
  assert S_continuity >= 0.90
  → HRG-8
```

---

### 11.1 Temporal Execution Loop Diagram (v17.1 — AUTHORITATIVE)

> **TemporalEngine is a STATEFUL AUTOREGRESSIVE LOOP CONTROLLER.**
>
> It is NOT:
> - a stateless function
> - a batch processor
> - a segment merger
>
> It IS:
> - a sequential controller
> - a temporal state manager
> - a continuity-preserving generator loop

```
TEMPORAL EXECUTION LOOP (canonical flow):

  S-08 completes
  ┌──────────────────────────────────────────────────────┐
  │  Segment_1 = Wan2.2.generate(init_image)             │
  │  buffer    = TemporalBufferManager.init(Segment_1)   │
  │  assert len(buffer.frames) == 5          [RULE-86]   │
  │  segments  = [Segment_1]                             │
  └──────────────┬───────────────────────────────────────┘
                 │
                 ▼
  S-09 AUTOREGRESSIVE LOOP (one iteration per remaining segment):
  ┌──────────────────────────────────────────────────────┐
  │  FOR each segment_plan in scene_plan[1:]:            │
  │                                                      │
  │    [TemporalBufferGate]                              │
  │    assert len(buffer.frames) == 5                    │
  │                 │                                    │
  │    [MotionStateTracker]                              │
  │    motion_state = estimate(buffer.frames)            │
  │                 │                                    │
  │    [TemporalBufferManager.encode]                    │
  │    latents = encode(buffer)  → shape (5,C',H',W')   │
  │                 │                                    │
  │    [SVICFGGate + SVIScheduler]                       │
  │    cfg    = clamp(cfg, 5.0, 6.0)                     │
  │    lora_w = SVIScheduler(timestep)  ← noise-aware    │
  │                 │                                    │
  │    [SVI.generate]                                    │
  │    segment_n = SVI(init_latents=latents, ...)        │
  │                 │                                    │
  │    [Validation Triple — ALL must pass]               │
  │    1. CLIPValidator.score(keyframe, ref) >= 0.93     │
  │    2. ContinuityValidator.score_segment(...) >= min  │
  │    3. TemporalConsistencyValidator (buffer integrity)│
  │       → any fail: retry; retries exhausted: HALT     │
  │                 │                                    │
  │    [TemporalBufferManager.update]                    │
  │    buffer = update(buffer, segment_n)                │
  │    assert len(buffer.frames) == 5                    │
  │                 │                                    │
  │    segments.append(segment_n)                        │
  │    context = context.evolve({temporal_state: buffer})│
  │                                                      │
  └──────────────┬───────────────────────────────────────┘
                 │  (loop repeats for next segment)
                 ▼
  return segments  →  S-10 ContinuityValidationAgent
```

### 11.2 TemporalBuffer Lifecycle Architecture (v17.1 — AUTHORITATIVE)

```
TemporalBuffer Lifecycle:

  INIT (called once, after S-08):
    buffer = last_5_frames(segment_1)
    buffer.frame_count = 5
    buffer.device      = CPU           (CPU-resident always between steps)

  UPDATE (called after every SVI segment in S-09 loop):
    buffer = last_5_frames(segment_n)
    buffer.frame_count = 5
    assert buffer.frame_count == 5    (hard check; failure = SYSTEM FAILURE)

  ENCODE (called once per segment, inside loop, GPU only during encode):
    latents = encoder(buffer.frames.to(GPU))
    latents.to(CPU)                   (return to CPU immediately)
    assert latents.shape[0] == 5

  VALIDATION (executed before every SVI call):
    assert len(buffer.frames) == 5
    assert all(f.shape == buffer.frames[0].shape for f in buffer.frames)
    assert buffer.device == "cpu"     (must be CPU before encode step)

  DEVICE RULE (strict):
    Buffer frames are CPU tensors at rest.
    They move to GPU ONLY inside TemporalBufferManager.encode().
    encode() returns CPU tensors immediately.
    No GPU-resident buffer between segments.

  Hard Invariant:
    buffer.frame_count != 5 at any validation point → raise TemporalBufferError
    → TemporalEngine MUST NOT proceed → CriticalPipelineError → pipeline halts
```

---

## 12. Identity Long-Sequence Hardening Architecture (v17.0 additions)

All v16.0 architecture retained. v17.0 additions:

```python
# IdentityState dataclass (new v17.0)
@dataclass(frozen=False)  # mutable internally but only updated via IdentityStateTracker
class IdentityState:
    embedding_vector: torch.Tensor   # frozen from char_identity_ref
    drift_score: float = 0.0        # cumulative drift from baseline
    history: List[float] = field(default_factory=list)  # per-stage drift

# IdentityStateTracker usage
def update_identity_state(state: IdentityState, new_embedding: torch.Tensor) -> IdentityState:
    current_drift = 1.0 - cosine_similarity(state.embedding_vector, new_embedding)
    new_drift_score = state.drift_score + current_drift
    new_history = state.history + [current_drift]

    if new_drift_score > IDENTITY_CUMULATIVE_DRIFT_THRESHOLD:
        raise IdentityCumulativeDriftError(
            scene_id=context.scene_id,
            cumulative_drift=new_drift_score,
            threshold=IDENTITY_CUMULATIVE_DRIFT_THRESHOLD
        )
    return IdentityState(
        embedding_vector=state.embedding_vector,  # embedding never changes
        drift_score=new_drift_score,
        history=new_history
    )
```

---

## 13. Audio Pipeline Architecture (v17.0 additions)

All v16.0 audio pipeline architecture retained. v17.0 additions:

```
AUDIO PIPELINE DATA FLOW (v17.0 additions):

[S-15: AudioMixingAgent — ENHANCED]
  mixed = mixer.mix(dialogue, ambient, music)

  # AudioPriorityGate (v16.0 — retained)
  assert actual_d > actual_a > actual_m

  # AudioQualityValidator (NEW v17.0 — RULE-99)
  snr = audio_quality_validator.compute_snr(mixed)
  peak = audio_quality_validator.compute_peak_db(mixed)

  if snr < MIN_SNR_DB:
    # Re-mix with adjusted levels
    mixer.adjust_levels(dialogue_boost=+3.0)
    mixed = mixer.remix()

  if peak > 0.0:
    # Normalize to prevent clipping
    mixed = audio_quality_validator.normalize(mixed, target_peak=-1.0)

  assert snr >= MIN_SNR_DB, f"SNR {snr:.1f} dB insufficient"
  assert peak <= 0.0, "Clipping detected"

  # CrossModalAlignmentValidator (NEW v17.0)
  for seg_id, (video_seg, audio_seg) in enumerate(zip(video_segments, dialogue_segments)):
    alignment_error = abs(audio_seg.duration_s - video_seg.duration_s)
    assert alignment_error <= TIMING_TOLERANCE_S

  storage.write("audio_quality_log.json", AudioQualityRecord(...))
  storage.write("cross_modal_alignment_log.json", CrossModalAlignmentRecord(...))
```

---

## 14. Enforcement Architecture (v17.0 additions)

All v16.0 enforcement architecture retained. v17.0 new gates:

```
CompositionPlanGate (NEW v17.0):
  Position: Before S-05 (BaseImageAgent), S-06 (ImageEditAgent), S-08 (Wan2.2)
  Check: composition_plan is not None AND CompositionPlanSchema.validate(plan)
  On failure: CRITICAL — pipeline halts; rerun SceneCompositionAgent

TemporalBufferGate (NEW v17.0):
  Position: Before every SVI call in TemporalEngine (segments ≥ 2)
  Check: len(buffer.frames) == TEMPORAL_BUFFER_SIZE (= 5)
  On failure: CRITICAL — TemporalEngine must not run; raise TemporalBufferError

SVICFGGate (NEW v17.0):
  Position: In SVIScheduler before every generation call
  Check: 5.0 <= cfg <= 6.0
  On failure: CRITICAL — reject generation; fix CFG value

AutoregressiveGate (NEW v17.0):
  Position: Before SVI conditioning in TemporalEngine
  Check: isinstance(init_latents, torch.Tensor) AND init_latents.shape[0] == 5
  On failure: CRITICAL — single-frame input detected; halt

CrossPhaseIdentityGate (NEW v17.0):
  Position: After each SVI segment; after LipSyncAgent
  Check: clip_validator.score(keyframe, char_identity_ref) >= CLIP_IDENTITY_THRESHOLD
  On failure: DEGRADED — segment regeneration (≤ 3 retries)

AudioQualityGate (NEW v17.0):
  Position: In AudioMixingAgent after mixing
  Check: snr >= MIN_SNR_DB AND peak_db <= 0.0
  On failure: DEGRADED — re-mix or normalize; retry
```

---

## 15. Observability Architecture (v17.0 additions)

All v16.0 observability retained. v17.0 additions:

```python
# TemporalEngine Events
tracer.log({
    "event": "temporal_buffer_update",
    "segment_id": seg_id,
    "scene_id": scene_id,
    "frame_count": 5,
    "buffer_timestamps": [...],
    "trace_id": trace_id
})

tracer.log({
    "event": "svi_generation",
    "segment_id": seg_id,
    "cfg": cfg_value,
    "steps": steps_used,
    "lora_schedule": [0.6, 0.5, 0.4],  # per-phase weights used
    "clip_score": clip_score,
    "trace_id": trace_id
})

tracer.log({
    "event": "motion_state_update",
    "segment_id": seg_id,
    "velocity_magnitude": motion_state.magnitude,
    "direction": motion_state.direction,
    "trace_id": trace_id
})

# Identity State Events
tracer.log({
    "event": "identity_state_update",
    "stage_id": stage_id,
    "drift_score": drift_score,
    "cumulative_drift": cumulative_drift,
    "threshold_exceeded": False,
    "trace_id": trace_id
})

# Audio Quality Events
tracer.log({
    "event": "audio_quality_validation",
    "scene_id": scene_id,
    "snr_db": snr_db,
    "peak_db": peak_db,
    "snr_passed": snr_db >= MIN_SNR_DB,
    "clipping_passed": peak_db <= 0.0,
    "trace_id": trace_id
})

# Scene Composition Events
tracer.log({
    "event": "composition_plan_created",
    "scene_id": scene_id,
    "camera_angle": plan.camera_angle,
    "camera_motion": plan.camera_motion,
    "motion_vector": plan.motion_vector,
    "trace_id": trace_id
})
```

---

## 16. VRAM Management Architecture (v17.0 additions)

All v16.0 VRAM management retained. v17.0 additions:

```
PHASE 3 (Video — v17.0 clarification):
  S-08: Wan2.2-I2V-A14B-FP8 (~14GB VRAM)
    → Generate Segment_1
    → TemporalBuffer initialized
    Unload Wan2.2 if VRAM tight before SVI load

  S-09: SVI Pro 2 (~8-12GB VRAM estimated)
    → Autoregressive loop (all subsequent segments)
    → TemporalBufferManager: tensors kept on CPU, moved to GPU for SVI call
    → MotionStateTracker: runs on CPU (optical flow)
    → CLIPValidator: CPU with GPU batch for per-segment validation
    Unload SVI after all segments complete

VRAM DISCIPLINE for TemporalBuffer:
  TemporalBuffer.frames tensor: CPU-resident between segments
  GPU transfer: only during SVI encode_frames() call
  Return to CPU: immediately after SVI generation completes (finally block)

RULE: At no point during S-09 may image models (FLUX, Z-Image) be resident.
RULE: TemporalBuffer frames tensor must return to CPU after each SVI call.
```

---

## 17. Data Flow Architecture (v17.0 — Full Pipeline)

```
[IDEA] → ScriptAgent → [script.json] → HRG-1
  ↓ Approved
[script.json] → SceneAgent + SegmentAgent → [scenes.json + segment_plan.json] → HRG-2
  ↓ Approved
[script.json + char_profiles] → IdentityDesignAgent → [identity_design.json] → HRG-3
  ↓ Approved
[dialogue + emotion + motion_intent + characters + environment]
  → SceneCompositionAgent → [composition_plan.json] → HRG-4 (NEW)
  ↓ Approved
[identity_design + composition_plan] → BaseImageAgent → [6 base images]
  → CLIPValidator → char_identity_ref FROZEN → HRG-5
  ↓ Approved
[base images] → IdentityReinforcementLoop (6A→6B→6C)
  → CLIPValidator per sub-stage → HRG-6
  ↓ Approved
[6C images] → ImageRefinementAgent → [refined images]
  → CLIPValidator (score + drift) → HRG-7
  ↓ Approved
[Phase 2B GPU Cleanup]
  ↓
[refined images + composition_plan] → VideoSegmentGenerator (Wan2.2) → [Segment_1]
  → TemporalBuffer.init(Segment_1) → CLIPValidator(Segment_1.keyframe)
  ↓
[TemporalBuffer] → TemporalEngine (SVI autoregressive loop) → [Segments 2..N]
  → CLIPValidator per segment → TemporalBuffer.update per segment
  ↓
[all segments] → ContinuityValidationAgent → [continuity_report] → HRG-8
  ↓ Approved
[script + segment_plan] → DialogueAgent → [dialogue audio]
  → TimingValidator → HRG-9
  ↓ Approved
[video segments + dialogue audio] → LipSyncAgent → [synced video]
  → CLIPValidator(identity_delta) → HRG-10
  ↓ Approved
[identity_design] → AmbientAudioAgent → [ambient audio]
[script] → MusicAgent → [background music]
[dialogue + ambient + music] → AudioMixingAgent → [mixed audio]
  → AudioPriorityGate → AudioQualityValidator (SNR + clipping)
  → CrossModalAlignmentValidator → HRG-11
  ↓ Approved
[synced video + mixed audio] → AssemblyAgent → [final_video.mp4]
  → ExportAgent → [/workspace/output/{job_id}/{scene_id}/]
  → QualityAgent → [pipeline_report.json, sla_summary.json, adaptive_state.json]
```

---

## 18–19. GPU Execution Model & Job Lifecycle (unchanged from v16.0)

---

## 20. Streamlit UI Architecture (v17.0 additions)

All v16.0 UI architecture retained. v17.0 additions:

**11 HRG Panels (v17.0):**

| Panel | Trigger | UI Components |
|---|---|---|
| HRG-1: Script Review | ScriptAgent completes | JSON editor per scene; approve/edit/regenerate |
| HRG-2: Scene/Segment Plan Review (NEW) | SceneAgent + SegmentAgent complete | Scene list with durations; segment breakdown |
| HRG-3: Identity Design Review | IdentityDesignAgent completes | reference_strategy highlighted; JSON editor |
| HRG-4: Composition Review (NEW) | SceneCompositionAgent completes | CompositionPlan all 6 fields editable; approve/edit/recompose |
| HRG-5: Base Image Review | Stage S-05 completes | 6 images + CLIP scores; upload widget |
| HRG-6: Identity Reinforcement Review | Stage S-06C completes | 3 sub-stage tabs (6A/6B/6C); CLIP per tab |
| HRG-7: Polished Image Review | Stage S-07 completes | Before/after; drift badge; CLIP score |
| HRG-8: Motion QA Review | Stage S-10 completes | Video player; continuity score; identity-per-segment chart |
| HRG-9: Voice QA Review | Stage S-11 completes | Audio per segment; timing Gantt chart |
| HRG-10: Lip Sync QA Review | Stage S-12 completes | Synced video; phoneme score; identity delta badge |
| HRG-11: Final Audio QA Review | Stage S-15 completes | Full scene video; audio level meters; SNR badge; clipping status |

**Temporal Engine Panel (NEW v17.0):**
```
Temporal Engine Status:
  Buffer size:          [5/5 frames]  [green badge]
  Segment 1 (Wan2.2):  ✓ [init_image generation; CLIP: 0.94]
  Segment 2 (SVI):     ✓ [conditioned] CFG: 5.5  Steps: 40  CLIP: 0.93
  Segment N (SVI):     ✓ [conditioned] CFG: 5.5  Steps: 30  CLIP: 0.95
  LoRA schedule:       [0.6 → 0.5 → 0.4] per segment
  Motion direction:    forward  Magnitude: 0.12
  Identity drift:      [0.005, 0.008, 0.006, ...]  Cumulative: 0.019
```

---

## 21. FastAPI Backend Architecture (v17.0 additions)

All v16.0 FastAPI architecture retained. v17.0 additions:

```
Additional endpoints:
  GET  /jobs/{job_id}/temporal/buffer    → TemporalBuffer state (frame count, timestamps)
  GET  /jobs/{job_id}/temporal/motion    → MotionState per segment
  GET  /jobs/{job_id}/identity/state     → IdentityState (drift score + history)
  GET  /jobs/{job_id}/audio/validation   → AudioQualityRecord (SNR + clipping)
  GET  /jobs/{job_id}/composition        → CompositionPlan for scene
  POST /jobs/{job_id}/hrg/HRG-2         → Scene/segment plan decision
  POST /jobs/{job_id}/hrg/HRG-4         → Composition plan decision

HRG Checkpoint count updated to 11 (HRG-1 through HRG-11).
All existing HRG-1 through HRG-9 endpoints retained and renumbered where necessary.
```

---

## 22. Storage Architecture (v17.0 additions)

All v16.0 storage retained. v17.0 additions covered in §7.3 of Document 01.

---

## 23. Error & Recovery Architecture (v17.0 additions)

All v16.0 error/recovery architecture retained. v17.0 additions:

```
TemporalBuffer Failure Recovery:
  TemporalBufferManager raises TemporalBufferError(scene_id, frame_count, required=5)
  → CRITICAL: pipeline halts — temporal engine cannot run
  → Diagnosis: check segment_1 generation; re-initialize from Wan2.2 output
  → No retry without valid 5-frame buffer

SVI CFG Violation Recovery:
  SVIScheduler raises SVICFGViolationError(cfg, min=5.0, max=6.0)
  → CRITICAL: pipeline halts
  → Fix: correct CFG in settings; restart pipeline from S-09

CompositionPlan Missing Recovery:
  CompositionPlanGate raises CompositionPlanMissingError(stage, scene_id)
  → CRITICAL: pipeline halts; no image generation without plan
  → Rerun SceneCompositionAgent for affected scene

Cross-Phase Identity Failure Recovery (Video Segment):
  CLIPValidator raises CLIPValidationError(stage="S-09", segment_id, score)
  → TemporalRetryController.retry(segment_id) [up to 3 attempts]
  → On failure: HRG-8 escalation with identity failure badge per segment
  → Human decides: accept suboptimal or trigger segment regeneration

Audio Quality Failure Recovery:
  AudioQualityValidator raises AudioQualityError(snr, peak, scene_id)
  → Re-mix with adjusted parameters (dialogue_boost or normalization)
  → Up to 3 re-mix attempts
  → If SNR still failing: escalate to HRG-11 with warning badge

Cumulative Identity Drift Recovery:
  IdentityStateTracker raises IdentityCumulativeDriftError(scene, drift, threshold)
  → Trigger full phase regeneration (image or video phase depending on where detected)
  → Max 1 full phase regeneration per scene before SceneHaltError
```

---

## 24. Layer Interface Contracts (v17.0 additions)

All v16.0 interface contracts retained. v17.0 additions:

```
Scene Composition Layer (Layer 7):
  CAN call:  Qwen wrapper (structured generation), CompositionPlanValidator
  CAN read:  script.json, identity_design.json, Context (frozen)
  CAN write: composition_plan_{scene_id}.json
  CANNOT call: Image agents, TemporalEngine
  CANNOT modify: Context (only context.evolve() via orchestrator)

TemporalEngine (Layer 10):
  CAN call:  SVIScheduler, TemporalBufferManager, MotionStateTracker,
             CLIPValidator (identity per segment), ContinuityValidator
  CAN read:  Context (temporal_state, motion_state, camera_state)
  CAN write: temporal_buffer_log.json, motion_state_log.json
  CANNOT call: ImagePipelineLayer, AudioPipelineLayer
  CANNOT start: before char_identity_ref is frozen in ImmutableContext

CrossPhaseIdentityValidator:
  CALLABLE from: TemporalEngine (per segment), LipSyncAgent (per synced segment)
  Uses: char_identity_ref from ImmutableContext (frozen)
  Returns: CLIPValidationResult with score and delta
  NEVER modifies: char_identity_ref embedding

AudioQualityValidator:
  CALLABLE from: AudioMixingAgent only
  Returns: AudioQualityRecord (snr_db, peak_db, passed)
  STATELESS: no side effects

IdentityStateTracker:
  CALLABLE from: All image agents, TemporalEngine, LipSyncAgent
  Maintains: Append-only history; cumulative drift
  Raises: IdentityCumulativeDriftError if threshold exceeded
```

---

## 25–26. Deployment Architecture & GPU Ownership Rules (unchanged from v16.0)

---

## 27. Architecture Decision Records (v17.0 additions)

All v16.0 ADR-029 through ADR-033 retained.

**ADR-034: SceneCompositionAgent as Mandatory Pre-Image Gate**
- **Decision:** A dedicated SceneCompositionAgent (S-04) translates narrative intent (dialogue, emotion, motion_intent) into visual directives (CompositionPlan) before any image generation can begin.
- **Rationale:** In v16.0, image generation consumed narrative fields directly from script.json, causing inconsistent visual-narrative alignment. Camera angles, character blocking, and motion vectors must be explicitly decided by the composition layer, not implicitly assumed by image models.
- **Consequence:** All image and video wrappers receive CompositionPlan fields as explicit inputs. HRG-4 gives human control over visual composition.

**ADR-035: TemporalBuffer Strict 5-Frame Policy**
- **Decision:** TemporalBuffer holds exactly 5 frames at all times. TemporalEngine refuses to run with fewer.
- **Rationale:** Single-frame conditioning causes temporal jitter (discontinuous motion) between segments. 5-frame conditioning empirically provides sufficient motion context for smooth continuation. Strict enforcement prevents silent degradation.
- **Consequence:** Segment_1 initializes the buffer by extracting its last 5 frames. Any failure in segment_1 generation blocks all subsequent segments.

**ADR-036: Noise-Aware LoRA Scheduling for SVI**
- **Decision:** SVIScheduler applies dynamic LoRA weight (0.6/0.5/0.4) based on diffusion timestep phase rather than a static value.
- **Rationale:** High-noise phases need strong LoRA influence to establish structure/motion patterns. Low-noise phases need reduced LoRA to preserve fine detail and prevent over-sharpening artifacts. Static weighting optimizes for neither extreme.
- **Consequence:** All SVI generation calls must go through SVIScheduler; direct static weight assignment is forbidden.

**ADR-037: Frozen char_identity_ref After S-07**
- **Decision:** The CLIP embedding of the best-scoring refined image is computed once after S-07 and stored frozen in ImmutableContext. All downstream CLIP validation uses this exact embedding.
- **Rationale:** Recomputing the reference embedding at each phase (video, lip sync) introduces reference drift — a different embedding baseline causes apparently passing scores that actually represent different identity states. A frozen reference is the only architecturally correct approach.
- **Consequence:** char_identity_ref is a mandatory field in ImmutableContext from HRG-7 approval onward. Any code that attempts to update it post-freeze raises an ImmutableContextError.

**ADR-038: AudioQualityValidator as Mandatory Post-Mix Gate**
- **Decision:** AudioQualityValidator runs after every AudioMixingAgent completion, measuring SNR and peak levels, before audio is written to storage.
- **Rationale:** Subjective audio review at HRG-11 is insufficient for objective quality requirements. SNR < 10 dB indicates dialogue is buried under background audio. Clipping causes distortion artifacts that survive export. Both are objectively measurable and should be caught before human review.
- **Consequence:** HRG-11 always shows SNR and clipping badges. Re-mixing or normalization may occur before HRG-11 is displayed.
