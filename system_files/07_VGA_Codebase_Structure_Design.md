# VGA Codebase Structure Design
**Project:** Video Generation Automation (VGA) — Motivation System
**Version:** 17.2.0
**Status:** Authoritative Reference
**Audience:** All Engineers, Claude Code Agent

---

## 1. Repository Root

```
vga/
├── vga/                    ← main source package
├── tests/
├── config/
├── scripts/
├── docs/
├── snapshots/
│   ├── v15_baseline/
│   ├── v16_candidate/
│   └── v17_candidate/      ← NEW v17.0
├── devtools_reports/
├── .pre-commit-config.yaml
├── DEVIATION_LOG.md
├── .env.example
├── requirements.txt
├── requirements.lock
├── requirements-dev.txt
├── pyproject.toml
├── Makefile
└── README.md
```

---

## 2. Full Directory Tree

```
vga/
└── vga/
    ├── __init__.py
    ├── bootstrap.py              ← startup: env validation, CUDA assert, ArchitectureGuard,
    │                               singletons 6A–6Y (v14.1); 6Z-a through 6Z-f (v15.0);
    │                               6Z-g through 6Z-o (v16.0: CLIPValidator, LoRAManager,
    │                               ContinuityValidator, TimingValidator, HRGController,
    │                               FastAPI startup, Streamlit startup);
    │                               6Z-p through 6Z-z (v17.0: SceneCompositionAgent,
    │                               TemporalBufferManager, SVIScheduler, MotionStateTracker,
    │                               TemporalRetryController, IdentityStateTracker,
    │                               AudioQualityValidator, CrossModalAlignmentValidator,
    │                               CompositionPlanValidator, HRGController(11 checkpoints))
    │
    ├── api/                      ← FastAPI HTTP layer (MANDATORY — RULE-85)
    │   ├── __init__.py
    │   ├── main.py
    │   ├── middleware/
    │   │   ├── auth.py
    │   │   └── logging.py
    │   └── routes/
    │       ├── __init__.py
    │       ├── jobs.py
    │       ├── hrg.py            ← POST/GET /jobs/{id}/hrg/{checkpoint} (11 checkpoints)
    │       ├── health.py
    │       ├── report.py
    │       ├── temporal.py       ← NEW v17.0: GET /jobs/{id}/temporal/buffer + /motion
    │       ├── identity.py       ← NEW v17.0: GET /jobs/{id}/identity/state
    │       ├── audio.py          ← NEW v17.0: GET /jobs/{id}/audio/validation
    │       └── composition.py    ← NEW v17.0: GET /jobs/{id}/composition
    │
    ├── ui/                       ← Streamlit UI (MANDATORY — RULE-85)
    │   ├── app.py
    │   └── components/
    │       ├── hrg_panels/
    │       │   ├── hrg_1_script.py
    │       │   ├── hrg_2_scene_plan.py       ← NEW v17.0: scene/segment plan review
    │       │   ├── hrg_3_identity.py         ← was hrg_2 in v16.0
    │       │   ├── hrg_4_composition.py      ← NEW v17.0: composition plan review
    │       │   ├── hrg_5_base_images.py      ← was hrg_3 in v16.0
    │       │   ├── hrg_6_composed_images.py  ← was hrg_4 in v16.0
    │       │   ├── hrg_7_refined_images.py   ← was hrg_5 in v16.0
    │       │   ├── hrg_8_motion_qa.py        ← was hrg_6; now shows identity_per_segment
    │       │   ├── hrg_9_voice_qa.py         ← was hrg_7
    │       │   ├── hrg_10_lipsync_qa.py      ← was hrg_8; now shows identity_delta
    │       │   └── hrg_11_final_qa.py        ← was hrg_9; now shows SNR + clipping
    │       ├── temporal_engine_panel.py      ← NEW v17.0: buffer status, motion, identity
    │       ├── image_pipeline_panel.py       ← v16.0 (CLIP scores + LoRA status)
    │       ├── sla_monitor_panel.py          ← v15.0
    │       ├── adaptive_learning_panel.py    ← v15.0
    │       └── gating_mode_panel.py          ← v15.0
    │
    ├── core/
    │   ├── __init__.py
    │   ├── model_manager.py
    │   ├── character_registry.py
    │   ├── storage.py            ← v17.0: new paths (composition, temporal, identity)
    │   ├── logger.py
    │   ├── retry.py
    │   ├── exceptions.py         ← v17.0: adds CompositionPlanValidationError,
    │   │                           TemporalBufferError, SVICFGViolationError,
    │   │                           AutoregressiveViolationError,
    │   │                           TemporalSegmentFailureError,
    │   │                           IdentityCumulativeDriftError,
    │   │                           IdentityReferenceCorruptionError,
    │   │                           AudioQualityError,
    │   │                           MissingPredecessorOutputError
    │   ├── shutdown.py
    │   ├── schema_migrations.py  ← _migrate_v5_2_to_v6_0()
    │   ├── cost_guard.py
    │   ├── vram_guard.py
    │   ├── checkpoint_manager.py
    │   ├── scene_metrics_store.py
    │   ├── segment_save_guard.py
    │   ├── hrg_state_manager.py  ← v17.0: updated for 11 checkpoints
    │   ├── hrg_controller.py     ← v17.0: 11-checkpoint HRGController
    │   ├── master_orchestrator.py ← v17.0: execute_stage() contract
    │   ├── state_sync_service.py
    │   └── queue.py, worker.py
    │
    ├── quality/
    │   └── validator.py
    │
    ├── regeneration/
    │   └── engine.py
    │
    ├── adaptive/                 ← v15.0 (unchanged)
    │   ├── adaptive_memory.py
    │   ├── calibration_engine.py
    │   ├── performance_learner.py
    │   └── strategy_optimizer.py
    │
    ├── state/                    ← v15.0 + v17.0 5-dimensional context
    │   ├── immutable_context.py  ← v17.0: 5-dim (identity, motion, camera, lighting, temporal)
    │   ├── context_factory.py    ← v17.0: creates 5-dim initial context
    │   ├── context_history.py
    │   └── context_diff.py
    │
    ├── runtime/
    │   ├── sla_manager.py
    │   ├── gating_controller.py
    │   ├── system_guard.py
    │   ├── failure/
    │   │   ├── failure_classifier.py
    │   │   ├── retry_strategy_engine.py
    │   │   ├── retry_limiter.py
    │   │   ├── safe_fallback_engine.py
    │   │   └── output_integrity_checker.py
    │   ├── authority/
    │   │   └── execution_authority_manager.py
    │   └── resilience/
    │       ├── stability_manager.py
    │       ├── memory_sanitizer.py
    │       ├── async_io_manager.py
    │       ├── execution_scheduler.py
    │       ├── resource_monitor.py
    │       └── session_health_monitor.py
    │
    ├── models/
    │   ├── enums.py              ← v17.0: adds TemporalPhase, CompositionState
    │   ├── sla.py
    │   ├── schemas.py            ← v17.0: adds CompositionPlanSchema,
    │   │                           TemporalBufferRecord, MotionStateRecord,
    │   │                           SVIGenerationRecord, IdentityStateRecord,
    │   │                           AudioQualityRecord, CrossModalAlignmentRecord,
    │   │                           HRG2DisplayData, HRG4DisplayData,
    │   │                           HRG8DisplayData (updated), HRG10DisplayData,
    │   │                           HRG11DisplayData (updated),
    │   │                           TemporalBufferStatusResponse,
    │   │                           IdentityStateResponse, AudioValidationResponse
    │   ├── lora_manager.py
    │   └── wrappers/
    │       ├── flux_wrapper.py
    │       ├── z_image_wrapper.py
    │       ├── wan_wrapper.py    ← v17.0: accepts CompositionPlan motion params
    │       ├── svi_wrapper.py    ← v17.0: accepts init_latents (5-frame), lora_scheduler
    │       ├── cosyvoice_wrapper.py
    │       ├── latentsync_wrapper.py
    │       ├── mmaudio_wrapper.py
    │       ├── musicgen_wrapper.py
    │       └── qwen_wrapper.py
    │
    ├── validation/               ← NEW v17.0 additions
    │   ├── __init__.py
    │   ├── clip_validator.py     ← v16.0 (uses frozen char_identity_ref; RULE-95)
    │   ├── timing_validator.py   ← v16.0
    │   ├── composition_validator.py    ← NEW v17.0: CompositionPlanValidator
    │   ├── audio_quality_validator.py  ← NEW v17.0: SNR + peak level (RULE-99)
    │   └── cross_modal_alignment_validator.py
│       ├── cross_modal_validation_unified.py    # NEW v17.2: Unified cross-modal contract
│       ├── system_certification_validator.py    # NEW v17.2: 7-condition system certification ← NEW v17.0: video ↔ audio alignment
    │
    ├── agents/
    │   ├── __init__.py
    │   ├── base_agent.py
    │   │
    │   │  ── Phase 1: CPU/LLM ─────────────────────────────────────────
    │   ├── script_agent.py       ← v16.0
    │   ├── scene_agent.py        ← v16.0
    │   ├── segment_agent.py      ← v16.0
    │   ├── identity_design_agent.py ← v16.0
    │   ├── scene_composition_agent.py ← NEW v17.0: Stage S-04
    │   │
    │   │  ── Phase 2: GPU (Image) ───────────────────────────────────
    │   ├── base_image_agent.py    ← v16.0 + v17.0: CompositionPlan input
    │   ├── image_edit_agent.py    ← v16.0 + v17.0: CompositionPlan input to 6C
    │   ├── multi_angle_agent.py   ← v16.0
    │   ├── image_merge_agent.py   ← v16.0
    │   ├── scene_expansion_agent.py ← v16.0 + v17.0: CompositionPlan fully bound
    │   ├── image_refinement_agent.py ← v16.0 + v17.0: char_identity_ref freeze
    │   │
    │   │  ── Phase 3: GPU (Video) ───────────────────────────────────
    │   ├── video_segment_generator.py  ← NEW v17.0: Stage S-08 (Wan2.2 Segment_1)
    │   ├── continuity_validation_agent.py ← v16.0 + v17.0: identity_per_segment field
    │   │
    │   │  ── Phase 4: Audio ──────────────────────────────────────────
    │   ├── dialogue_agent.py      ← v16.0
    │   ├── lip_sync_agent.py      ← v16.0 + v17.0: IdentityStateTracker.update
    │   ├── ambient_audio_agent.py ← v16.0
    │   ├── music_agent.py         ← v16.0
    │   ├── audio_mixing_agent.py  ← v16.0 + v17.0: AudioQualityValidator + CrossModal
    │   │
    │   │  ── Phase 5: Assembly & Export ──────────────────────────────
    │   ├── assembly_agent.py      ← v16.0
    │   ├── export_agent.py        ← v16.0
    │   └── quality_agent.py       ← v16.0 + v17.0: v17.0 fields in PipelineReport
    │
    ├── temporal/                 ← v17.0: full TemporalEngine subsystem
    │   ├── __init__.py
    │   ├── temporal_engine.py        ← NEW v17.0: TemporalEngine (SVI autoregressive)
    │   ├── temporal_buffer_manager.py ← NEW v17.0: TemporalBufferManager
    │   ├── svi_scheduler.py           ← NEW v17.0: SVIScheduler (noise-aware LoRA)
    │   ├── motion_state_tracker.py    ← NEW v17.0: MotionStateTracker
    │   ├── temporal_retry_controller.py ← NEW v17.0: TemporalRetryController
    │   ├── temporal_orchestrator.py   ← v15.0 + v17.0: wires to TemporalEngine
    │   ├── segment_role_router.py     ← v14.0
    │   ├── motion_evolution_engine.py ← v14.0
    │   └── merge_engine.py            ← v14.0
    │
    ├── identity/                 ← v14.0 + v17.0 additions
    │   ├── identity_manager.py
    │   ├── identity_tracker.py
    │   ├── identity_drift_controller.py
    │   ├── lighting_normalizer.py
    │   ├── temporal_identity_validator.py
    │   ├── identity_reinforcement_engine.py
    │   └── identity_state_tracker.py  ← NEW v17.0: IdentityStateTracker (cross-phase drift)
    │
    ├── observability/            ← v14.0+ (unchanged)
    │   ├── tracer.py
    │   ├── metrics.py
    │   ├── logger.py
    │   ├── dashboard.py
    │   └── audit_system.py
    │
    ├── config/
    │   ├── settings.py           ← v17.0: all new constants (temporal, composition,
    │   │                           identity state, audio quality, SVI CFG/steps)
    │   └── prompts/
    │       ├── script_prompts.py
    │       ├── identity_prompts.py
    │       ├── image_prompts.py
    │       ├── video_prompts.py
    │       └── composition_prompts.py ← NEW v17.0: SceneCompositionAgent prompts
    │
    └── devtools/                 ← v15.0 (ZERO runtime import; dev-time only)
        ├── architecture_linter.py
        ├── rule_checker.py
        ├── regression_tester.py
        ├── snapshot_system.py
        ├── rollback_manager.py
        └── validator.py

tests/
├── unit/
│   ├── test_script_agent.py
│   ├── test_scene_composition_agent.py     ← NEW v17.0
│   ├── test_temporal_buffer_manager.py     ← NEW v17.0
│   ├── test_svi_scheduler.py               ← NEW v17.0
│   ├── test_motion_state_tracker.py        ← NEW v17.0
│   ├── test_identity_state_tracker.py      ← NEW v17.0
│   ├── test_audio_quality_validator.py     ← NEW v17.0
│   ├── test_cross_modal_alignment_validator.py ← NEW v17.0
│   ├── test_composition_validator.py       ← NEW v17.0
│   ├── test_base_image_agent.py
│   ├── test_image_edit_agent.py
│   ├── test_image_refinement_agent.py
│   ├── test_clip_validator.py
│   ├── test_lora_manager.py
│   ├── test_continuity_validation_agent.py
│   ├── test_dialogue_agent.py
│   ├── test_lip_sync_agent.py
│   ├── test_audio_mixing_agent.py          ← v17.0: SNR + clipping tests
│   ├── test_sla_manager.py
│   ├── test_adaptive_memory.py
│   ├── test_gating_controller.py
│   └── test_immutable_context.py           ← v17.0: 5-dim context tests
├── integration/
│   ├── test_full_image_pipeline.py
│   ├── test_full_audio_pipeline.py
│   ├── test_temporal_engine.py             ← NEW v17.0: autoregressive loop
│   ├── test_composition_to_image.py        ← NEW v17.0: S-04 → S-05 integration
│   ├── test_identity_cross_phase.py        ← NEW v17.0: CLIP across phases
│   ├── test_hrg_flow.py                    ← v17.0: updated for 11 checkpoints
│   ├── test_fastapi_endpoints.py           ← v17.0: new endpoints
│   └── test_sla_adaptive_loop.py
└── chaos/
    ├── test_clip_failure_recovery.py
    ├── test_lora_violation_halt.py
    ├── test_continuity_regen.py
    ├── test_audio_timing_fallback.py
    ├── test_hrg_timeout_resume.py
    ├── test_sla_violation_recovery.py
    ├── test_temporal_buffer_error.py        ← NEW v17.0
    ├── test_svi_cfg_violation.py            ← NEW v17.0
    ├── test_autoregressive_gate.py          ← NEW v17.0
    ├── test_identity_cumulative_drift.py    ← NEW v17.0
    ├── test_composition_plan_missing.py     ← NEW v17.0
    └── test_audio_quality_retry.py         ← NEW v17.0
```

---

## 3. Package Dependency Rules (v17.0)

All v16.0 dependency rules retained. v17.0 additions:

```
LAYER IMPORT RULES — violation detected by ArchitectureLinter:

vga/temporal/              CAN import:  torch, vga/validation/clip_validator.py,
                                         vga/models/schemas.py, vga/config/settings.py,
                                         vga/core/exceptions.py, vga/core/tracer.py
                           CANNOT import: vga/agents/ (no circular dependency)
                           CANNOT import: vga/api/, vga/ui/
                           CANNOT import: vga/audio/ or vga/adaptive/ directly

vga/identity/              CAN import:  torch, vga/validation/clip_validator.py,
                                         vga/models/schemas.py, vga/config/settings.py
                           CANNOT import: vga/temporal/ (separation of concerns)
                           CANNOT import: vga/agents/ (agents import identity, not reverse)

vga/validation/            CAN import:  torch, transformers, PIL, pydub, torchaudio (external)
                           CANNOT import: vga/agents/, vga/temporal/, vga/runtime/

vga/agents/scene_composition_agent.py:
  ONLY model consumer:    Qwen wrapper (structured output mode)
  MUST call:              CompositionPlanValidator.validate() before returning
  MUST NOT call:          Any image or video agent

vga/temporal/temporal_engine.py:
  ONLY SVI consumer:      vga/models/wrappers/svi_wrapper.py
  MUST use:               TemporalBufferManager, SVIScheduler, MotionStateTracker
  MUST call:              CLIPValidator per segment (RULE-89)
  MUST call:              IdentityStateTracker.update per segment
  MUST call:              context.evolve() after each segment

vga/identity/identity_state_tracker.py:
  CALLABLE from:          All image agents (S-05, S-06A/B/C, S-07)
                          vga/temporal/temporal_engine.py (per segment)
                          vga/agents/lip_sync_agent.py (per synced segment)
  NOT from:               vga/api/, vga/ui/

vga/validation/audio_quality_validator.py:
  CALLABLE from:          vga/agents/audio_mixing_agent.py ONLY
  STATELESS:              No side effects; returns AudioQualityRecord

vga/validation/cross_modal_alignment_validator.py:
  CALLABLE from:          vga/agents/audio_mixing_agent.py ONLY
  STATELESS:              No side effects; returns CrossModalAlignmentReport

vga/validation/composition_validator.py:
  CALLABLE from:          vga/agents/scene_composition_agent.py
                          vga/agents/base_image_agent.py (assert_in_context)
                          vga/temporal/temporal_engine.py (assert_in_context)
  STATELESS:              validates and returns; never modifies plan

vga/models/wrappers/svi_wrapper.py:
  MUST accept:            init_latents: Tensor (shape[0] == 5) — multi-frame
                          lora_scheduler: SVIScheduler — noise-aware
  MUST NOT accept:        init_image: single image (single-frame forbidden, RULE-87)
  MUST NOT accept:        lora_weight: float (static weight forbidden, RULE-86)

vga/models/wrappers/wan_wrapper.py:
  MUST accept:            init_image: PIL.Image, motion_params: dict from CompositionPlan
  Used by:                vga/agents/video_segment_generator.py (S-08 ONLY)
```

---

## 4. Key File Responsibilities Summary (v17.0)

| File | Single Responsibility |
|---|---|
| `bootstrap.py` | System startup; all singleton initialization including v17.0 components |
| `api/routes/hrg.py` | All 11 HRG REST endpoints; decision routing |
| `api/routes/temporal.py` | Temporal buffer + motion state status endpoints |
| `api/routes/identity.py` | Identity state + cumulative drift endpoint |
| `api/routes/audio.py` | Audio quality (SNR + clipping) endpoint |
| `api/routes/composition.py` | CompositionPlan endpoint |
| `validation/composition_validator.py` | All 6 CompositionPlan fields required; no image gen without plan |
| `validation/audio_quality_validator.py` | SNR ≥ 10 dB; peaks ≤ 0 dBFS; normalize on violation (RULE-99) |
| `validation/cross_modal_alignment_validator.py` | Video ↔ audio duration alignment per segment |
| `temporal/temporal_engine.py` | Autoregressive SVI loop; buffer management; identity per segment |
| `temporal/temporal_buffer_manager.py` | 5-frame buffer init, update, encode; invariant enforcement |
| `temporal/svi_scheduler.py` | Noise-aware LoRA scheduling: 0.6/0.5/0.4; CFG [5.0, 6.0] gate |
| `temporal/motion_state_tracker.py` | Optical flow estimation; velocity/direction/magnitude per segment |
| `temporal/temporal_retry_controller.py` | Parameter adjustment between segment retry attempts |
| `identity/identity_state_tracker.py` | Cumulative drift tracking across image + video + lip sync phases |
| `agents/scene_composition_agent.py` | S-04: CompositionPlan (all 6 fields) from narrative data |
| `agents/video_segment_generator.py` | S-08: Wan2.2 Segment_1 + TemporalBuffer initialization |
| `agents/base_image_agent.py` | S-05: 6 base images + CompositionPlan input + char_identity_ref candidate |
| `agents/image_refinement_agent.py` | S-07: Z-Image refinement + char_identity_ref FREEZE |
| `agents/audio_mixing_agent.py` | S-15: Priority mixing + AudioQualityValidator + CrossModal |
| `agents/lip_sync_agent.py` | S-12: LatentSync + IdentityStateTracker.update per segment |
| `core/hrg_controller.py` | 11-checkpoint HRG state machine |
| `core/master_orchestrator.py` | execute_stage() SYSTEM DIRECTIVE v17 contract |
| `state/immutable_context.py` | 5-dimensional context (identity, motion, camera, lighting, temporal) |
| `config/settings.py` | All constants including v17.0 (temporal, composition, identity, audio quality) |
| `models/schemas.py` | All schemas including v17.0 additions; schema_version v6.0 |
| `core/exceptions.py` | All exceptions including v17.0 additions |
| `ui/components/temporal_engine_panel.py` | Temporal buffer health, motion state, SVI metrics display |
| `ui/components/hrg_panels/hrg_4_composition.py` | CompositionPlan display; all 6 fields editable |
| `ui/components/hrg_panels/hrg_11_final_qa.py` | Final scene + SNR badge + clipping status + level meters |
