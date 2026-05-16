# VGA Pipeline & Execution Flow Specification
**Project:** Video Generation Automation (VGA)
**Version:** 17.2.0
**Status:** Authoritative Reference
**Audience:** Pipeline Engineers, Agent Implementors, Claude Code Agent

---

## Table of Contents

1. [Pipeline Overview](#1-pipeline-overview)
2. [Stage Registry](#2-stage-registry)
3. [Stage Specifications](#3-stage-specifications)
4. [Director Orchestration Logic](#4-director-orchestration-logic)
5. [Inter-Stage Data Contracts](#5-inter-stage-data-contracts)
6. [Error Handling Flows](#6-error-handling-flows)
7. [Resume Flow](#7-resume-flow)
8. [Pod Stop Flow](#8-pod-stop-flow)
9. [Rollback Plans Per Phase](#9-rollback-plans-per-phase)
10. [Full Execution Sequence Diagram](#10-full-execution-sequence-diagram)

---

## 1. Pipeline Overview

The VGA v17.0 pipeline consists of **16 sequential stages** (plus sub-stages) across 5 phases, executed by dedicated agents under `MasterOrchestrator` control. **Every stage is wrapped by the execute_stage() contract** (SYSTEM DIRECTIVE v17). **Every cross-component action is validated by `ExecutionAuthorityManager`** (RULE-54).

### 1.1 Master Execution Contract (SYSTEM DIRECTIVE v17)

Every stage MUST follow this execution contract without exception:

```python
def execute_stage(stage: Stage, input: Any, context: ImmutableContext) -> tuple[Any, ImmutableContext]:
    """
    SYSTEM DIRECTIVE v17: mandatory execution wrapper for all pipeline stages.
    No stage may bypass this contract.
    """
    # 1. System guard wrapping
    SystemGuard.execute(stage)
    authority_manager.validate(stage.authority_level, stage.action_name)

    # 2. Validate previous output exists and is valid
    validate_previous_output(stage.predecessor_stage, context)

    # 3. HRG checkpoint (if required for this stage)
    if stage.requires_hrg:
        HRGController.await_approval(stage.hrg_checkpoint, display_data=context.last_output)

    # 4. Execute stage
    output = stage.run(input, context)

    # 5. Validate this stage's output
    validate_output(stage, output)

    # 6. Evolve context with this stage's output
    context = context.evolve(output)

    return output, context
```

No stage MAY:
- Execute without validation of prior output (`validate_previous_output`)
- Mutate identity state without CLIP validation
- Bypass temporal continuity constraints (TemporalBuffer gates)
- Return without calling `context.evolve(output)`

### 1.2 Pipeline Execution Gates (v17.0 — all gates active)

No stage executes without ALL of the following conditions satisfied:

1. `conflict_resolver.check_consistency()` — no conflicts
2. `consistency_validator.validate_pre_stage()` — no violations
3. `hrg_controller.require_approval(stage_id)` — approved (at HRG-gated stages)
4. `pipeline_state == EXECUTING` confirmed
5. `quality_validator.validate(last_output)` — preceding artifact validated
6. `adaptive_engine.adjust_parameters()` — adjustment applied
7. `identity_manager.get_identity(character_id)` — resolved (GPU stages only)
8. `execution_guard.check_vram(required_gb)` — VRAM sufficient (GPU stages only)
9. `system_guard.execute(stage, state, gating_mode)` — wrapping active (RULE-41)
10. `authority_manager.validate(caller_level, action)` — authority passed (RULE-54)
11. `isinstance(context, Context)` — immutable context type check

**v17.0 additional gates:**
12. `composition_plan_gate.assert_present(context)` — CompositionPlan exists (image/video stages)
13. `temporal_buffer_gate.assert_ready(buffer)` — buffer.frame_count == 5 (SVI stage)
14. `svi_cfg_gate.assert_range(cfg)` — cfg ∈ [5.0, 6.0] (S-09)
15. `autoregressive_gate.assert_multiframe(init_latents)` — latents.shape[0] == 5 (S-09)

### 1.3 Pipeline Summary (v17.0)

```
Stage  Agent                              Input                       Output
───────────────────────────────────────────────────────────────────────────────────
─── PHASE 1: CPU/LLM (Narrative Intelligence) ───────────────────────────────────
       [Pre-pipeline]                     bootstrap                   All v16.0 components +
                                                                      SceneCompositionAgent,
                                                                      TemporalEngine subsystems,
                                                                      AudioQualityValidator,
                                                                      HRGController (11 checkpoints)

S-01   ScriptAgent                        idea (str)                  script.json [ScriptSchema v6.0]
         [SchemaValidationGate: ScriptSchema]
         [HRG-1: Script review]

S-02   SceneAgent + SegmentAgent          script.json                 scenes.json + segment_plan.json
         [ImmutableContextSystem initialised with 5-dimensional context]
         [HRG-2: Scene/segment plan review (NEW v17.0)]

S-03   IdentityDesignAgent                script + char_profiles      identity_design.json
         [SchemaValidationGate: IdentityDesignSchema]
         [HRG-3: Identity/environment design review]

S-04   SceneCompositionAgent (NEW v17.0)  dialogue + emotion +        composition_plan.json
         [CompositionPlanValidator: all 6 fields required]            [CompositionPlanSchema v6.0]
         [HRG-4: Composition review (NEW v17.0)]

─── PHASE 2: GPU (Visual Grounding) ─────────────────────────────────────────────
S-05   BaseImageAgent                     identity_design +           images/base/ (6 images)
         [CompositionPlanGate: plan present]                          composition_plan
         [LoRAConditionalGate: assert lora_not_loaded]
         [CLIPValidator × 6: ≥ 0.93 per image; char_identity_ref set]
         [HRG-5: Base image review]

S-06A  MultiAngleAgent                    base images                 images/angles/ (5–8 variants)
S-06B  ImageMergeAgent                    6A output + env images      images/composed/
S-06C  SceneExpansionAgent                6B output + motion_intent   images/expanded/
         + composition_plan
         [Each sub-stage: LoRAConditionalGate; CLIPValidator ≥ 0.93]
         [HRG-6: Identity reinforcement review]

S-07   ImageRefinementAgent               images/expanded/            images/refined/
         [LoRAManager.assert_unloaded()]
         [CLIPValidator: score ≥ 0.93 AND drift ≤ 0.02]
         [char_identity_ref FROZEN in ImmutableContext]
         [HRG-7: Polished image review]

─── PHASE 2B: Hard GPU Cleanup ──────────────────────────────────────────────────
       model_manager.unload_all()    FLUX + Z-Image + LoRA all gone
       gc + cuda_empty + sleep(3)    assert free_ratio ≥ 0.90

─── PHASE 3: GPU (Motion + Continuity) ───────────────────────────────────────────
S-08   VideoSegmentGenerator (Wan2.2)     refined images +            segment_1.mp4
         composition_plan
         [TemporalBufferManager.init(segment_1)]
         [CLIPValidator: keyframe ≥ 0.93]

S-09   TemporalEngine (SVI Pro 2)        TemporalBuffer + segment     segments[2..N].mp4
         plans + composition_plan
         [TemporalBufferGate: assert buffer.frames == 5]
         [SVICFGGate: assert cfg ∈ [5.0, 6.0]]
         [AutoregressiveGate: assert latents.shape[0] == 5]
         [SVIScheduler: noise-aware LoRA per timestep]
         [MotionStateTracker: estimate per segment]
         [CLIPValidator: identity per segment ≥ 0.93 (RULE-89)]
         [TemporalBufferManager.update per segment]
         [context.evolve per segment]

S-10   ContinuityValidationAgent          all video segments          continuity_report_{scene_id}.json
         [ContinuityGate: score ≥ 0.90]
         [HRG-8: Motion QA review (with identity-per-segment breakdown)]

─── PHASE 4: Audio Realism ──────────────────────────────────────────────────────
S-11   DialogueAgent                      script + segment_plan       audio/dialogue/
         [AudioTimingGate: |audio_dur - seg_dur| ≤ 0.10s]
         [HRG-9: Voice QA review]

S-12   LipSyncAgent                       video segments + dialogue   synced/*.mp4
         [CLIPValidator: identity_delta ≤ 0.03 per segment (RULE-89, RULE-97)]
         [PhonemeAlignmentGate: score ≥ 0.80]
         [IdentityStateTracker.update per synced segment]
         [HRG-10: Lip sync QA review]

S-13   AmbientAudioAgent                  identity_design             audio/ambient/
S-14   MusicAgent                         script.json                 audio/music/

S-15   AudioMixingAgent                   dialogue + ambient + music  audio/mixed/
         [AudioPriorityGate: Dialogue > Ambient > Music]
         [AudioQualityValidator: SNR ≥ 10 dB; peaks ≤ 0 dBFS (RULE-99)]
         [CrossModalAlignmentValidator: video ↔ audio duration]
         [HRG-11: Final audio QA review]

─── PHASE 5: Finalization ───────────────────────────────────────────────────────
S-16a  AssemblyAgent                      synced video + mixed audio  final_video.mp4
S-16b  ExportAgent                        final_video.mp4             /workspace/output/{job}/{scene}/
S-16c  QualityAgent                       all artifacts               pipeline_report.json +
                                                                      sla_summary.json +
                                                                      adaptive_state.json
```

---

## 2. Stage Registry (v17.0)

| Stage | Agent | Phase | HRG Checkpoint | VRAM Model(s) |
|---|---|---|---|---|
| S-01 | ScriptAgent | 1-CPU | HRG-1 | Qwen2.5-14B (4-bit) |
| S-02 | SceneAgent + SegmentAgent | 1-CPU | HRG-2 (NEW) | Qwen2.5-14B |
| S-03 | IdentityDesignAgent | 1-CPU | HRG-3 | Qwen (structured) |
| S-04 | SceneCompositionAgent (NEW) | 1-CPU | HRG-4 (NEW) | Qwen (structured) |
| S-05 | BaseImageAgent | 2-GPU | HRG-5 | FLUX.2-klein 4B |
| S-06A | MultiAngleAgent | 2-GPU | — | FLUX.2-klein 4B + (LoRA) |
| S-06B | ImageMergeAgent | 2-GPU | — | FLUX.2-klein 4B + (LoRA) |
| S-06C | SceneExpansionAgent | 2-GPU | HRG-6 (group) | FLUX.2-klein 4B + (LoRA) |
| S-07 | ImageRefinementAgent | 2-GPU | HRG-7 | Z-Image-Turbo |
| — | GPU Cleanup | 2B | — | ALL UNLOADED |
| S-08 | VideoSegmentGenerator (Wan2.2) | 3-GPU | — | Wan2.2-I2V-A14B-FP8 |
| S-09 | TemporalEngine (SVI Pro 2) | 3-GPU | — | SVI Pro 2 |
| S-10 | ContinuityValidationAgent | 3-CPU | HRG-8 | — |
| S-11 | DialogueAgent | 4-CPU/GPU | HRG-9 | CosyVoice3-0.5B |
| S-12 | LipSyncAgent | 4-GPU | HRG-10 | LatentSync-1.6 |
| S-13 | AmbientAudioAgent | 4-CPU | — | MMAudio |
| S-14 | MusicAgent | 4-CPU/GPU | — | MusicGen-medium |
| S-15 | AudioMixingAgent | 4-CPU | HRG-11 | — (pydub/torchaudio) |
| S-16a | AssemblyAgent | 5-CPU | — | — |
| S-16b | ExportAgent | 5-CPU | — | — |
| S-16c | QualityAgent | 5-CPU | — | — |

---

## 3. Stage Specifications

### 3.1 S-01: ScriptAgent (unchanged from v16.0 except schema_version)

```
INPUT:  idea: str
MODEL:  Qwen2.5-14B-Instruct-bnb-4bit
OUTPUT: script.json [ScriptSchema v6.0]
VALIDATION: SchemaValidationGate
SLA: ≤ 30 seconds
HRG-1: Script review; approve | edit | regenerate
GATE: HRG-1 required before S-02
```

### 3.2 S-02: SceneAgent + SegmentAgent (v17.0 — adds HRG-2)

```
execute_stage(S-02, input=script.json, context)
authority_manager.validate(SCENE_LEVEL, "plan_scenes")

INPUT:  script.json
OUTPUT: scenes.json (scene 10–30s) + segment_plan.json (segments 3–5s avg 4s)

IMMUTABLE CONTEXT — 5-dimensional initialisation:
  context = context_factory.create_initial({
    identity_state: IdentityState(embedding_vector=None, drift_score=0.0, history=[]),
    motion_state:   MotionState(velocity_vector=None, direction="stationary", magnitude=0.0),
    camera_state:   CameraState(angle=None, motion=None),
    lighting_state: LightingState(style=None),
    temporal_state: TemporalState(buffer=None, segment_index=0)
  })

VALIDATION:
  for each scene: assert 10 <= scene.duration <= 30
  for each segment: assert 3 <= (end - start) <= 5
  assert contiguous_coverage(segments, scene)

SLA: ≤ 60 seconds

HRG-2 (NEW v17.0):
  display: scenes.json (scene list with durations, beats) + segment_plan.json breakdown
  actions: approve | edit scene durations | trigger replanning
  gate: HRG-2 approval REQUIRED before S-03 begins
```

### 3.3 S-03: IdentityDesignAgent (updated — now gates on HRG-3 not HRG-2)

```
execute_stage(S-03, input=script.json + char_profiles, context)

INPUT:  script.json + character_profiles
MODEL:  Qwen2.5-14B (structured output mode)
OUTPUT: identity_design.json [IdentityDesignSchema v6.0]
  fields: character_identity, environment_description, reference_strategy (REQUIRED)

VALIDATION:
  assert identity_design["reference_strategy"] is not None and len > 0
  SchemaValidationGate → IdentityDesignSchema

SLA: ≤ 60 seconds

HRG-3 (was HRG-2 in v16.0):
  display: identity_design.json; reference_strategy highlighted
  actions: approve | edit JSON | trigger redesign
  gate: HRG-3 approval REQUIRED before S-04 begins

context = context.evolve({
  identity_state: IdentityState(embedding_vector=None, ...),  # set after S-07
  lighting_state: LightingState(style=derived_from_environment)
})
```

### 3.4 S-04: SceneCompositionAgent (NEW v17.0)

```
execute_stage(S-04, input=script.json + identity_design.json, context)
authority_manager.validate(COMPOSITION_LEVEL, "compose_scene")

INPUT:
  scene_data = {
    dialogue:      script[scene_id].dialogue,
    emotion:       script[scene_id].emotion,
    motion_intent: script[scene_id].motion_intent,
    characters:    identity_design.character_identity,
    environment:   identity_design.environment_description
  }

MODEL:  Qwen2.5-14B (structured output mode)

OUTPUT: composition_plan_{scene_id}.json [CompositionPlanSchema v6.0]
  camera_angle, camera_motion, character_positions,
  focus_subject, lighting_style, motion_vector  ← ALL 6 REQUIRED

VALIDATION:
  for attempt in range(COMPOSITION_MAX_RETRIES):
    plan = qwen.generate_structured(scene_data, CompositionPlanSchema)
    if CompositionPlanValidator.validate(plan):
      break
    if attempt == COMPOSITION_MAX_RETRIES - 1:
      raise CompositionPlanValidationError(scene_id, reason="all_fields_required")

STORAGE: composition/{job_id}/{scene_id}/composition_plan_{scene_id}.json
SLA: ≤ 15 seconds

context = context.evolve({
  camera_state:   CameraState(angle=plan.camera_angle, motion=plan.camera_motion),
  lighting_state: LightingState(style=plan.lighting_style)
})

HRG-4 (NEW v17.0):
  display: CompositionPlan all 6 fields; each field editable
  actions: approve | edit camera_angle | edit motion_vector | trigger recompose
  gate: HRG-4 approval REQUIRED before any image generation (S-05) begins (RULE-88)
```

### 3.5 S-05: BaseImageAgent (v17.0 — CompositionPlan integrated)

```
execute_stage(S-05, input=identity_design + composition_plan, context)
authority_manager.validate(IMAGE_GEN_LEVEL, "generate_base_images")

PRE-GENERATION GATES:
  composition_plan_gate.assert_present(context)   ← RULE-88: CompositionPlan required
  lora_manager.assert_unloaded()                   ← RULE-91: no LoRA in base generation
  assert context.rebuild_prompt == False

INPUT:  identity_design.json [approved at HRG-3] + composition_plan [approved at HRG-4]

GENERATION:
  for i in range(BASE_IMAGE_COUNT):  # 6 images
    prompt = prompt_builder.build_base_prompt(
      identity_design,
      composition_plan,         ← camera_angle, lighting_style, character_positions
      neutral_pose=True
    )
    image = flux_wrapper.generate(prompt=prompt, lora=None)

POST-GENERATION:
  for i, image in enumerate(base_images):
    clip_score = clip_validator.score(image, char_identity_ref_candidate)
    if clip_score >= CLIP_IDENTITY_THRESHOLD and i == 0:
      # Set char_identity_ref to best-scored image (updated as we go)
      best_ref_candidate = (clip_score, image)
    log_clip_validation(stage="stage_5", image_id=i, score=clip_score)

  # Freeze char_identity_ref after all 6 images validated
  # (final freeze happens at S-07 completion; preliminary reference set here)

STORAGE: images/{job_id}/{scene_id}/base/base_{i}.png
SLA: ≤ 120 seconds total

context = context.evolve({
  identity_state: IdentityState(
    embedding_vector=clip_encoder(best_base_image),  # preliminary; frozen at S-07
    drift_score=0.0,
    history=[]
  )
})

HRG-5 (was HRG-3):
  display: 6 base images with CLIP scores; upload widget
  actions: approve | upload replacement | trigger regeneration
  gate: HRG-5 approval REQUIRED before S-06A begins
```

### 3.6 S-06A, S-06B, S-06C: IdentityReinforcementLoop (v17.0 — renamed, CompositionPlan integrated)

```
Sub-stage execution pattern (identical for 6A, 6B, 6C):

execute_stage(S-06{X}, input=prev_output + composition_plan, context)

PRE-GENERATION GATES:
  composition_plan_gate.assert_present(context)   ← RULE-88
  lora_manager.assert_state(editing_required)      ← SEC-055

STAGE 6A (MultiAngleAgent):
  input: base images + composition_plan
  output: 5–8 angle variants, expression variants, pose variants (RULE-94)
  lora_weight_default: LORA_DEFAULT_5A = 0.50
  path: images/{job_id}/{scene_id}/angles/

STAGE 6B (ImageMergeAgent):
  input: 6A output + environment images + composition_plan
  output: identity-stabilised master image (character + environment unified)
  lora_weight_default: LORA_DEFAULT_5B = 0.55
  path: images/{job_id}/{scene_id}/composed/

STAGE 6C (SceneExpansionAgent):
  input: 6B output + composition_plan (character_positions, focus_subject)
  output: motion-aligned scene shots bound to CompositionPlan
  lora_weight_default: LORA_DEFAULT_5C = 0.60
  path: images/{job_id}/{scene_id}/expanded/

POST-GENERATION (each sub-stage):
  for each output image:
    clip_score = clip_validator.score(image, char_identity_ref)
    identity_state_tracker.update(IS, CLIP_encoder(image))
    if clip_score < CLIP_IDENTITY_THRESHOLD:
      → retry sub-stage (up to 3 retries)

LORA UNLOAD AFTER 6C:
  lora_manager.unload()
  lora_manager.assert_unloaded()
  tracer.log({"event": "lora_unloaded_after_6c", "scene_id": scene_id})

SLA: ≤ 90 seconds per sub-stage

context = context.evolve({
  identity_state: identity_state_tracker.current_state
})

HRG-6 (was HRG-4):
  display: tabbed view: 6A | 6B | 6C images with CLIP scores per tab
  actions: approve | upload replacement | trigger sub-stage redo
  gate: HRG-6 approval REQUIRED before S-07 begins
```

### 3.7 S-07: ImageRefinementAgent (v17.0 — char_identity_ref frozen here)

```
execute_stage(S-07, input=images/expanded/, context)

INPUT:  images/expanded/ [6C output, approved at HRG-6]
MODEL:  Tongyi-MAI/Z-Image-Turbo

PRE-GENERATION GATES:
  lora_manager.assert_unloaded()   ← LoRA MUST be gone
  denoise_param = clamp(params.denoise, DENOISE_MIN, DENOISE_MAX)
  cfg_param = CFG_REFINEMENT  # = 5.0

GENERATION:
  best_clip_score = 0.0
  best_image = None

  for each input_image:
    clip_before = clip_validator.score(input_image, context.identity_state.embedding_vector)
    refined_image = z_image_turbo.refine(input_image, denoise=denoise_param, cfg=cfg_param)

    clip_after = clip_validator.score(refined_image, context.identity_state.embedding_vector)
    drift = abs(clip_after - clip_before)
    log_identity_drift(stage="stage_7", clip_before, clip_after, drift)

    if drift > CLIP_DRIFT_THRESHOLD:
      → retry with denoise -= DENOISE_STEP_DOWN
    if clip_after < CLIP_IDENTITY_THRESHOLD:
      → retry (up to 3 total)

    if clip_after > best_clip_score:
      best_clip_score = clip_after
      best_image = refined_image

  # FREEZE char_identity_ref from best refined image
  char_identity_ref = CLIP_encoder(best_image)
  context = context.evolve({
    identity_state: IdentityState(
      embedding_vector=char_identity_ref,  ← FROZEN; never changes after this
      drift_score=context.identity_state.drift_score,
      history=context.identity_state.history
    )
  })

STORAGE: images/{job_id}/{scene_id}/refined/
SLA: ≤ 30 seconds per image

HRG-7 (was HRG-5):
  display: before/after image pair + CLIP scores + drift badge
  actions: approve | trigger re-refinement (lower denoise)
  gate: HRG-7 approval REQUIRED before Phase 2B cleanup begins
  NOTE: On HRG-7 approval, char_identity_ref is locked and written to ImmutableContext
```

### 3.8 Phase 2B: Hard GPU Cleanup (unchanged from v16.0)

```
model_manager.unload_all()   ← FLUX + Z-Image + LoRA ALL UNLOADED
gc.collect()
torch.cuda.empty_cache()
time.sleep(3)
free_ratio = get_free_vram_ratio()
assert free_ratio >= 0.90
tracer.log({"event": "phase_2b_cleanup_complete", "free_ratio": free_ratio})
```

### 3.9 S-08: VideoSegmentGenerator / Wan2.2 (NEW v17.0 specification)

```
execute_stage(S-08, input=refined_images + composition_plan, context)
authority_manager.validate(VIDEO_GEN_LEVEL, "generate_segment_1")

PRE-GENERATION GATES:
  composition_plan_gate.assert_present(context)    ← RULE-88
  assert char_identity_ref is not None              ← frozen at HRG-7

INPUT:
  init_image        = load_refined_image(scene_id)  ← best refined image from S-07
  composition_plan  = context.camera_state + context.motion_state

GENERATION:
  segment_1 = wan_wrapper.generate(
    init_image    = init_image,
    prompt        = compose_video_prompt(
      scene_plan[0], composition_plan, context.identity_state
    ),
    motion_params = {
      "camera_motion": context.camera_state.motion,
      "motion_vector": segment_plan[0].motion_vector or composition_plan.motion_vector
    },
    cfg   = 7.0,    ← Wan2.2 uses different CFG range than SVI
    steps = 30,
    seed  = scene_seed
  )

POST-GENERATION:
  clip_score = clip_validator.score(segment_1.keyframe, char_identity_ref)
  assert clip_score >= CLIP_IDENTITY_THRESHOLD, "Segment_1 identity failed"
  identity_state_tracker.update(IS, CLIP_encoder(segment_1.keyframe))

  # CRITICAL: Initialize TemporalBuffer from Segment_1
  buffer = TemporalBufferManager.init(segment_1)
  assert buffer.frames.shape[0] == TEMPORAL_BUFFER_SIZE  # = 5

context = context.evolve({
  temporal_state: TemporalState(buffer=buffer, segment_index=1)
})

STORAGE: video/{job_id}/{scene_id}/segment_001.mp4
```

### 3.10 S-09: TemporalEngine — Temporal Loop Controller (NOT A STAGE) (v17.2 HARDENED)

```
DEFINITION (v17.2 — AUTHORITATIVE):
  S-09 is NOT a pipeline stage in the traditional sense.
  S-09 is a CONTROL LOOP CONTROLLER responsible for sequential segment generation.
  It receives a TemporalBuffer and a list of segment plans, and generates segments
  one at a time in autoregressive order.

execute_stage(S-09, input=TemporalBuffer + segment_plans, context)
authority_manager.validate(TEMPORAL_LEVEL, "generate_temporal_segments")

TEMPORAL EXECUTION CONTRACT v17.2 — S-09 HARD RULES:

  RULE: S-09 is NOT a single execution step.
  RULE: S-09 is NOT a batch processor.
  RULE: S-09 input MUST NOT be a pre-generated list of segments.
  RULE: S-09 output MUST be generated sequentially — one segment per loop iteration.
  RULE: S-09 is a CONTROL LOOP that:
    1. Receives TemporalBuffer (5 frames from Segment_1 or prior segment)
    2. Generates next segment using buffer as multi-frame conditioning input
    3. Validates the segment (CLIPValidator + ContinuityValidator + TemporalConsistencyValidator)
    4. Updates TemporalBuffer from new segment
    5. Repeats until all segments for this scene are generated

  FORBIDDEN (these violate S-09 contract — execution SHALL halt if attempted):
    generate_all_segments(scene_plan)     ← batch generation; FORBIDDEN
    merge_segments(seg_a, seg_b)          ← post-hoc merge; FORBIDDEN
    SVI.generate(init_image=img)          ← single-image; FORBIDDEN for n >= 2
    [plan for plan in scene_plan]         ← list comprehension; FORBIDDEN (does not update buffer)
    asyncio.gather(*tasks)                ← parallel generation; FORBIDDEN

  HARD RULE: S-09 MUST NOT accept multiple segments as input.
  HARD RULE: S-09 MUST NOT merge independently generated segments.
  HARD RULE: S-09 MUST generate each segment from the previous buffer.
  HARD RULE: S-09 MUST operate as a loop controller — one segment per iteration.
  HARD RULE: S-09 MUST operate sequentially — one segment at a time.

PRECONDITION:
  buffer = context.temporal_state.buffer
  assert buffer is not None, "Buffer must be initialized from S-08"
  assert buffer.frames.shape[0] == TEMPORAL_BUFFER_SIZE   ← RULE-86

AUTOREGRESSIVE LOOP:
  segments = [segment_1]  ← from S-08

  for n, segment_plan in enumerate(scene_plan[1:], start=2):

    # Gate 1: Buffer integrity
    temporal_buffer_gate.assert_ready(buffer)    ← frame_count == 5 (RULE-86)

    # Gate 2: Motion state estimation
    motion_state = MotionStateTracker.estimate(buffer.frames)
    context = context.evolve({motion_state: motion_state})
    log_motion_state(n, motion_state)

    # Gate 3: Multi-frame latent encoding
    latents = TemporalBufferManager.encode(buffer)  # shape: (5, C', H', W')
    autoregressive_gate.assert_multiframe(latents)  ← latents.shape[0] == 5

    # Gate 4: CFG validation
    cfg = context.adaptive_params.get("svi_cfg", SVI_CFG_DEFAULT)
    svi_cfg_gate.assert_range(cfg)                 ← cfg ∈ [5.0, 6.0] (RULE-86)

    # Gate 5: Steps determination
    steps = STEPS_CRITICAL if segment_plan.is_critical else STEPS_STANDARD

    # Retry loop for this segment
    for attempt in range(TEMPORAL_MAX_RETRIES_PER_SEGMENT):

      segment_n = svi_wrapper.generate(
        init_latents      = latents,          ← MULTI-FRAME; NOT single-image
        prompt            = compose_temporal_prompt(segment_plan, motion_state, context),
        lora_scheduler    = SVIScheduler,     ← noise-aware; static weight FORBIDDEN
        cfg               = cfg,
        steps             = steps,
        temporal_attention = True
      )

      # Identity validation per segment (RULE-89)
      clip_score = clip_validator.score(segment_n.keyframe, char_identity_ref)
      log_clip_validation(stage="stage_9", segment_id=n, score=clip_score)

      # Per-segment continuity
      cont = ContinuityValidator.score_segment(buffer.frames[-1], segment_n.frames[0])

      if clip_score >= CLIP_IDENTITY_THRESHOLD and cont >= SEGMENT_CONTINUITY_MIN:
        identity_state_tracker.update(IS, CLIP_encoder(segment_n.keyframe))
        break

      # Adjust and retry
      adjust_temporal_params(attempt)

    else:
      raise TemporalSegmentFailureError(scene_id, segment_id=n)

    # Update buffer (RULE-86)
    buffer = TemporalBufferManager.update(buffer, segment_n)
    assert buffer.frames.shape[0] == TEMPORAL_BUFFER_SIZE

    segments.append(segment_n)
    context = context.evolve({
      temporal_state: TemporalState(buffer=buffer, segment_index=n)
    })

    log_temporal_buffer_update(n, buffer)

STORAGE: video/{job_id}/{scene_id}/segment_{n:03d}.mp4
SLA: ≤ 120 seconds per standard segment; ≤ 300 seconds per critical segment
```

### 3.10.1 S-09 Explicit Execution Flow (v17.1 — Canonical)

```
S-08 completes → Segment_1 generated by Wan2.2
                 ↓
                 buffer = TemporalBufferManager.init(Segment_1)
                 assert len(buffer.frames) == 5
                 ↓
S-09 LOOP START:
  ┌──────────────────────────────────────────────────────────────────┐
  │  for segment_plan in scene_plan[1:]:                            │
  │                                                                  │
  │    assert len(buffer.frames) == 5         ← gate before gen     │
  │    latents = encode(buffer)               ← 5-frame tensor      │
  │    assert latents.shape[0] == 5           ← gate before SVI     │
  │    cfg = clamp(cfg, 5.0, 6.0)            ← CFG gate            │
  │    segment_n = SVI.generate(init_latents=latents, ...)          │
  │    validate_segment(segment_n)            ← CLIP + continuity   │
  │    buffer = update(buffer, segment_n)     ← rolling update      │
  │    assert len(buffer.frames) == 5         ← gate after update   │
  │    context = context.evolve(...)                                 │
  │                                                                  │
  └──────────────────────────────────────────────────────────────────┘
  ↓
  return all_segments → S-10
```

### 3.10.2 Global Failure & Retry Model (NEW v17.2)

```
GLOBAL FAILURE HANDLING MODEL v17.2:

Every pipeline stage MUST implement the following retry pattern:

  for attempt in range(MAX_RETRIES_FOR_STAGE):
      output = run_stage(input)
      validation_result = validate_output(output)

      if validation_result.passed:
          proceed_to_next_stage(output)
          break
      else:
          adjust_parameters(attempt)
          retry()

  if all_retries_exhausted:
      raise StageFailureError(stage_id=stage.name, best_result=best_output)

TEMPORAL FAILURE SPECIFIC RULES:
  If continuity_score < SEGMENT_CONTINUITY_MIN:
    → Adjust CFG (increase by 0.1, clamped to SVI_CFG_MAX)
    → Adjust motion vector parameters
    → Retry generation from same buffer state
    → Max retries: TEMPORAL_MAX_RETRIES_PER_SEGMENT (= 3)

  If CLIP score < CLIP_IDENTITY_THRESHOLD (0.93):
    → Increase LoRA weight for next attempt (if applicable)
    → Regenerate segment from same buffer
    → Max retries: TEMPORAL_MAX_RETRIES_PER_SEGMENT (= 3)

  If cumulative identity drift > IDENTITY_CUMULATIVE_DRIFT_THRESHOLD (0.15):
    → reset identity state after successful phase regeneration
    → Trigger full phase regeneration from S-08
    → Max full phase regenerations: IDENTITY_MAX_PHASE_REGENERATIONS (= 1)

AUDIO FAILURE SPECIFIC RULES:
  If SNR < MIN_SNR_DB (10 dB):
    → Rebalance audio levels (increase dialogue gain, reduce ambient/music)
    → Regenerate audio mix
    → Max retries: AUDIO_QUALITY_MAX_RETRIES (= 3)

  If peak_db > 0.0 (clipping detected):
    → Apply normalization (target: -1 dBFS)
    → Re-validate after normalization
    → Max retries: AUDIO_QUALITY_MAX_RETRIES (= 3)
```

---

### 3.11 S-10: ContinuityValidationAgent (v17.0 — identity-per-segment breakdown)

```
execute_stage(S-10, input=all video segments, context)

INPUT:  all video segments [from S-08 + S-09]
OUTPUT: continuity_report_{scene_id}.json [ContinuityReport v6.0]

SCORING:
  S_motion   = motion_smoothness_score(segments)
  S_lighting = lighting_consistency_score(segments)
  S_identity = identity_consistency_score(segments, char_identity_ref)
  S_continuity = 0.40*S_motion + 0.30*S_lighting + 0.30*S_identity

  identity_per_segment = [
    clip_validator.score(seg.keyframe, char_identity_ref)
    for seg in segments
  ]

VALIDATION:
  report = ContinuityReport(
    scene_id=scene_id,
    continuity_score=S_continuity,
    motion_smoothness=S_motion,
    lighting_consistency=S_lighting,
    identity_consistency=S_identity,
    identity_per_segment=identity_per_segment,   ← NEW v17.0 field
    passed=S_continuity >= CONTINUITY_THRESHOLD
  )
  storage.write(f"continuity_report_{scene_id}.json", report)

  if not report.passed:
    → diagnose failure
    → scene_regen_engine.regenerate(scene_id, diagnosis)
    → re-run S-08 + S-09 + S-10 for this scene (max CONTINUITY_MAX_SCENE_REGEN = 2)

SLA: ≤ 15 seconds

HRG-8 (was HRG-6):
  display: video segments + continuity score breakdown + identity-per-segment chart
  actions: approve | trigger scene regeneration
  gate: HRG-8 approval REQUIRED before Phase 4 audio begins
```

### 3.12 S-11: DialogueAgent (updated numbering; logic unchanged from v16.0)

```
execute_stage(S-11, input=script + segment_plan, context)
[All v16.0 S-09 logic retained]

SLA: timing error ≤ 0.1s per segment

HRG-9 (was HRG-7):
  display: audio per segment + timing alignment Gantt chart
  actions: approve | adjust emotion/pacing | trigger regeneration
  gate: HRG-9 approval REQUIRED before S-12 begins
```

### 3.13 S-12: LipSyncAgent (v17.0 — IdentityState update added)

```
execute_stage(S-12, input=video segments + dialogue audio, context)

PRECONDITION:
  assert dialogue_audio_ready(all_segments)  ← audio MUST exist before lip sync

INPUT:  video segments [from S-09] + dialogue audio [from S-11, approved at HRG-9]
MODEL:  ByteDance/LatentSync-1.6

GENERATION:
  for each segment:
    synced_video = latentsync.sync(
      video=video_segment[i],
      audio=dialogue_audio[i],
      sync_strength=SYNC_STRENGTH_DEFAULT
    )

POST-GENERATION VALIDATION (RULE-89 — identity validation after lip sync):
  phoneme_score = latentsync.get_alignment_score(synced_video)
  if phoneme_score < PHONEME_ALIGNMENT_THRESHOLD:
    → retry with different seed (up to 3 retries)

  clip_synced = clip_validator.score(synced_frame, char_identity_ref)  ← RULE-89
  clip_stage7 = clip_records_stage7[segment_id]
  delta = abs(clip_synced - clip_stage7)
  if delta > LIPSYNC_IDENTITY_DELTA_THRESHOLD:  # > 0.03; RULE-97
    → retry with reduced sync_strength

  # Update IdentityState (cumulative drift tracking)
  identity_state_tracker.update(IS, CLIP_encoder(synced_frame))

context = context.evolve({
  identity_state: identity_state_tracker.current_state
})

STORAGE: synced/{job_id}/{scene_id}/synced_{seg_id}.mp4
SLA: ≤ 60 seconds per segment

HRG-10 (was HRG-8):
  display: lip-synced video + phoneme alignment score + identity_delta per segment
  actions: approve | trigger re-sync | upload alternative audio
  gate: HRG-10 approval REQUIRED before S-13 begins
```

### 3.14 S-13, S-14: Ambient & Music (updated numbering; logic unchanged from v16.0)

```
S-13: AmbientAudioAgent — MMAudio
S-14: MusicAgent — MusicGen-medium
[All v16.0 logic retained; no HRG checkpoint; reviewed in HRG-11]
```

### 3.15 S-15: AudioMixingAgent (v17.0 — AudioQualityValidator + CrossModal added)

```
execute_stage(S-15, input=dialogue + ambient + music, context)

MIXING: [All v16.0 mixing logic retained]
  mixer.mix(dialogue, ambient, music)
  AudioPriorityGate: assert dialogue_db > ambient_db > music_db

POST-MIXING VALIDATION (NEW v17.0):

  # Audio Quality Validation (RULE-99)
  snr_db  = audio_quality_validator.compute_snr(mixed)
  peak_db = audio_quality_validator.compute_peak_db(mixed)

  if snr_db < MIN_SNR_DB:
    for attempt in range(AUDIO_QUALITY_MAX_RETRIES):
      mixed = mixer.remix_with_boost(dialogue_boost=+3.0)
      snr_db = audio_quality_validator.compute_snr(mixed)
      if snr_db >= MIN_SNR_DB: break
    if snr_db < MIN_SNR_DB:
      tracer.log({"event": "snr_below_threshold_after_retries", ...})

  if peak_db > MAX_PEAK_DBFS:
    mixed = audio_quality_validator.normalize(mixed, target_peak=-(HEADROOM_DB))
    peak_db = audio_quality_validator.compute_peak_db(mixed)

  assert snr_db >= MIN_SNR_DB         ← RULE-99
  assert peak_db <= MAX_PEAK_DBFS     ← RULE-99

  storage.write("audio_quality_log.json", AudioQualityRecord(snr_db, peak_db, ...))

  # Cross-Modal Alignment Validation (NEW v17.0)
  cross_modal_validator.validate_alignment(video_segments, dialogue_segments)
  storage.write("cross_modal_alignment_log.json", CrossModalAlignmentRecord(...))

STORAGE: audio/{job_id}/{scene_id}/mixed/mixed_{scene_id}.wav
SLA: ≤ 20 seconds

HRG-11 (was HRG-9):
  display: full scene video + mixed audio + SNR badge + clipping status + level meters
  actions: approve (→ triggers S-16 export) | trigger remix | adjust levels
  gate: HRG-11 approval REQUIRED before S-16a assembly begins
  on_approve: pipeline_state = APPROVED_FOR_EXPORT
```

### 3.16 S-16a through S-16c: Assembly, Export, Quality (updated numbering)

```
S-16a: AssemblyAgent [was S-14]
  input: synced video segments + mixed audio
  output: final_video.mp4 per scene
  method: ffmpeg merge

S-16b: ExportAgent [was S-15]
  input: final_video.mp4
  output: /workspace/output/{job_id}/{scene_id}/final_video.mp4
  also writes: pipeline_report.json, sla_summary.json, adaptive_state.json
  post-export: scene locked (immutable)
  SLA: ≤ 30 seconds

S-16c: QualityAgent [was S-16]
  generates PipelineReport:
    includes all v16.0 fields + v17.0 additions:
      composition_plan_summary, temporal_buffer_health_log,
      motion_state_summary, identity_state_final,
      audio_quality_summary, cross_modal_alignment_summary
  final adaptive_memory.save()
  hrg_controller.save_log()  ← 11-checkpoint full log
  pod STOP signal
```

---

## 4. Director Orchestration Logic (v17.0)

All v16.0 MasterOrchestrator logic retained. v17.0 updates:

```python
class MasterOrchestrator:

    STAGE_ORDER = [
        "S-01_script",
        "S-02_scene_segment",
        "S-03_identity",
        "S-04_composition",        ← NEW v17.0
        "S-05_base_image",
        "S-06A_multi_angle",
        "S-06B_image_merge",
        "S-06C_scene_expansion",
        "S-07_refinement",
        "PHASE_2B_cleanup",
        "S-08_wan_segment_1",      ← NEW separation from temporal engine
        "S-09_temporal_engine",    ← NEW: TemporalEngine subsystem
        "S-10_continuity_validation",
        "S-11_dialogue",
        "S-12_lip_sync",
        "S-13_ambient",
        "S-14_music",
        "S-15_audio_mix",
        "S-16a_assembly",
        "S-16b_export",
        "S-16c_quality"
    ]

    HRG_GATES = {
        "S-01_script":           "HRG-1",
        "S-02_scene_segment":    "HRG-2",    ← NEW v17.0
        "S-03_identity":         "HRG-3",
        "S-04_composition":      "HRG-4",    ← NEW v17.0
        "S-05_base_image":       "HRG-5",
        "S-06C_scene_expansion": "HRG-6",
        "S-07_refinement":       "HRG-7",
        "S-10_continuity_validation": "HRG-8",
        "S-11_dialogue":         "HRG-9",
        "S-12_lip_sync":         "HRG-10",
        "S-15_audio_mix":        "HRG-11",
    }

    def _execute_stage(self, stage_id: str, job: Job) -> Any:
        """
        Implements SYSTEM DIRECTIVE v17 execute_stage() contract.
        """
        stage = self.stage_registry[stage_id]
        input_data = self.stage_input_resolver.resolve(stage_id, job)

        # SYSTEM DIRECTIVE v17 contract
        output, new_context = execute_stage(
            stage=stage,
            input=input_data,
            context=job.context
        )

        # Update job context
        job.context = new_context

        # Post-stage adaptive update (video stages)
        if stage_id == "S-09_temporal_engine":
            calibration_engine.update(adaptive_memory, scene_metrics)
            performance_learner.record(adaptive_memory, job.context, success=True)
            adaptive_memory.save()

        return output
```

---

## 5. Inter-Stage Data Contracts (v17.0)

All v16.0 data contracts retained. v17.0 additions:

| From Stage | To Stage | Data | Schema |
|---|---|---|---|
| S-03 | S-04 | identity_design.json | IdentityDesignSchema v6.0 |
| S-04 | S-05, S-06A/B/C, S-08, S-09 | composition_plan.json | CompositionPlanSchema v6.0 |
| S-04 | ImmutableContext | camera_state, lighting_state | CameraState, LightingState |
| S-07 | S-08, S-09, S-12 (ref) | char_identity_ref (frozen) | Tensor (CLIP embedding) |
| S-08 | S-09 | segment_1.mp4 + TemporalBuffer | mp4 + TemporalBufferRecord |
| S-09 | S-10, S-12, S-16a | video segments 2..N | mp4 file paths + IdentityStateRecord |
| S-09 | ImmutableContext | temporal_state, motion_state | TemporalState, MotionState |
| S-10 | S-11 (gate) | continuity_report (with identity_per_segment) | ContinuityReport v6.0 |
| S-15 | S-16a | mixed audio + AudioQualityRecord | wav + AudioQualityRecord v6.0 |

---

## 6. Error Handling Flows (v17.0 additions)

All v16.0 error flows retained (§6.1–§6.14). v17.0 additions:

### 6.15 CompositionPlan Missing Flow (NEW v17.0)

```
CompositionPlanGate raises CompositionPlanMissingError(stage, scene_id):

  tracer.log({"event": "composition_plan_missing", "stage": stage, "severity": "CRITICAL"})
  raise CriticalPipelineError(
    stage=stage,
    cause="composition_plan_missing",
    message=f"CompositionPlan required at {stage} but not found in context"
  )
  ← Pipeline halts; requires SceneCompositionAgent to run for this scene
```

### 6.16 TemporalBuffer Error Flow (NEW v17.0)

```
TemporalBufferGate raises TemporalBufferError(scene_id, frame_count, required):

  tracer.log({"event": "temporal_buffer_error", "frame_count": frame_count, "severity": "CRITICAL"})
  raise CriticalPipelineError(
    stage="S-09_temporal_engine",
    cause="temporal_buffer_insufficient",
    message=f"Buffer has {frame_count} frames; {required} required"
  )
  ← Pipeline halts; S-08 must regenerate Segment_1 and reinitialize buffer
```

### 6.17 SVI CFG Violation Flow (NEW v17.0)

```
SVICFGGate raises SVICFGViolationError(cfg, min, max):

  tracer.log({"event": "svi_cfg_violation", "cfg": cfg, "severity": "CRITICAL"})
  raise CriticalPipelineError(
    stage="S-09_temporal_engine",
    cause="svi_cfg_out_of_range",
    message=f"SVI CFG {cfg} outside [{min}, {max}]"
  )
  ← Pipeline halts; code fix required before restart
```

### 6.18 Audio Quality Failure Flow (NEW v17.0)

```
AudioQualityValidator raises AudioQualityError(snr_db, peak_db, scene_id):

  for attempt in range(AUDIO_QUALITY_MAX_RETRIES):
    if snr_db < MIN_SNR_DB:
      mixed = audio_mixing_agent.remix_with_boost(dialogue_boost=+3.0)
    elif peak_db > MAX_PEAK_DBFS:
      mixed = audio_quality_validator.normalize(mixed)

    snr_db, peak_db = audio_quality_validator.validate(mixed)

    if snr_db >= MIN_SNR_DB and peak_db <= MAX_PEAK_DBFS:
      return mixed  ← success

  # After retries: accept best result; flag in HRG-11
  tracer.log({"event": "audio_quality_below_threshold_after_retries", "scene_id": scene_id})
  # HRG-11 shows orange warning badge; human decides
```

### 6.19 Cumulative Identity Drift Flow (NEW v17.0)

```
IdentityStateTracker raises IdentityCumulativeDriftError(scene_id, cumulative_drift, threshold):

  diagnosis = identify_drift_source(IS.history)
  # diagnosis: "image_phase" | "video_phase" | "lipsync_phase"

  if regen_count < IDENTITY_MAX_PHASE_REGENERATIONS:  # = 1
    regen_count += 1
    IS.drift_score = 0.0
    IS.history = []

    if diagnosis == "video_phase":
      scene_regen_engine.regenerate_from(start_stage="S-08", scene_id=scene_id)
    elif diagnosis == "lipsync_phase":
      lip_sync_agent.regenerate_all(scene_id=scene_id, reduce_sync_strength=True)
    elif diagnosis == "image_phase":
      scene_regen_engine.regenerate_from(start_stage="S-05", scene_id=scene_id)
  else:
    raise SceneHaltError(scene_id, reason="identity_drift_after_max_regen")
```

---

## 7. Resume Flow (v17.0 additions)

All v16.0 resume logic retained. v17.0 additions:

```
Resume after TemporalBuffer error:
  Determine which segment failed (from session_state)
  Re-run S-08 if Segment_1 was the issue
  Re-run S-09 from failed segment N if buffer was valid for 1..N-1

Resume after CompositionPlan error:
  Re-run S-04 for the affected scene
  Re-display HRG-4 with new plan
  Continue from S-05 after approval

Resume after audio quality failure halt:
  Restore mixed audio from last valid attempt
  Display HRG-11 with quality warning badges
  Human decides: accept suboptimal or trigger re-mix
```

---

## 8. Pod Stop Flow (v17.0 additions)

All v16.0 pod stop logic retained. v17.0 additions:

```
# v17.0 additions to pod stop sequence:
temporal_buffer_manager.clear_all()          ← release TemporalBuffer tensors from GPU/CPU
motion_state_tracker.save_state(job_id)      ← persist final MotionState
identity_state_tracker.save_state(job_id)    ← persist final IdentityState (drift history)
audio_quality_validator.save_log(job_id)     ← persist AudioQualityRecord per scene
storage.write_composition_plans(job_id)      ← persist all CompositionPlans
storage.write_cross_modal_logs(job_id)       ← persist CrossModalAlignmentRecords
hrg_controller.save_log()                    ← 11-checkpoint full decision log
```

---

## 9. Rollback Plans Per Phase (v17.0 additions)

All v16.0 rollback plans retained. v17.0 additions:

```
Scene Composition (S-04) rollback:
  Snapshot: pre_composition
  Rollback trigger: CompositionPlan schema failures after 3 retries
  Rollback scope: use simplified CompositionPlan (default values per field)
  Note: HRG-4 shown with "fallback composition" warning; human confirms

S-08 / S-09 (Temporal) rollback:
  Snapshot: pre_temporal_generation
  Rollback trigger: TemporalBuffer corruption or SVI failure on all segments
  Rollback scope: regenerate Segment_1 (Wan2.2); reset buffer; retry S-09 from Segment_2
  Note: HRG-8 shown with regeneration note

Audio Quality rollback:
  Snapshot: pre_audio_mixing
  Rollback trigger: SNR cannot reach 10 dB after 3 remix attempts
  Rollback scope: accept best-SNR result; proceed with HRG-11 quality warning badge
  Note: Human makes final decision at HRG-11
```

---

## 10. Full Execution Sequence Diagram (v17.0)

```
BOOTSTRAP
  ├─ [All v16.0 bootstrap steps retained]
  ├─ SceneCompositionAgent.initialize()
  ├─ TemporalBufferManager.initialize()     ← buffer cleared
  ├─ SVIScheduler.initialize()              ← noise thresholds configured
  ├─ MotionStateTracker.initialize()
  ├─ AudioQualityValidator.initialize()
  └─ HRGController.initialize(11 checkpoints)  ← expanded from 9

PHASE 1: CPU/LLM (Narrative Intelligence)
  ├─ S-01: ScriptAgent → [SchemaGate] → [HRG-1]
  ├─ S-02: SceneAgent + SegmentAgent → [5D Context init] → [HRG-2 NEW]
  ├─ S-03: IdentityDesignAgent → [SchemaGate] → [HRG-3]
  └─ S-04: SceneCompositionAgent → [CompositionPlanValidator] → [HRG-4 NEW]

PHASE 2: GPU (Visual Grounding)
  ├─ S-05: BaseImageAgent (FLUX, no LoRA, CompositionPlan)
  │   └─ [CompositionPlanGate] → [LoRAGate] → [CLIPValidator × 6] → [HRG-5]
  │
  ├─ S-06A: MultiAngleAgent (5–8 variants; CompositionPlan)
  │   └─ [LoRAGate] → [CLIPValidator]
  ├─ S-06B: ImageMergeAgent (identity master)
  │   └─ [LoRAGate] → [CLIPValidator]
  ├─ S-06C: SceneExpansionAgent (CompositionPlan bound)
  │   └─ [LoRAGate] → [CLIPValidator] → [LoRAManager.unload()] → [HRG-6]
  │
  └─ S-07: ImageRefinementAgent (Z-Image-Turbo, no LoRA)
      └─ [CLIPValidator: score + drift] → [char_identity_ref FROZEN] → [HRG-7]

PHASE 2B: Hard GPU Cleanup
  └─ unload ALL image models → gc → cuda_empty → assert free ≥ 0.90

PHASE 3: GPU (Motion + Continuity)
  ├─ S-08: VideoSegmentGenerator (Wan2.2)
  │   └─ [CompositionPlanGate] → Segment_1 → [CLIPValidator]
  │       → [TemporalBufferManager.init] → buffer.frame_count=5
  │
  ├─ S-09: TemporalEngine (SVI — Autoregressive)
  │   └─ for each segment n ≥ 2:
  │       [TemporalBufferGate: frame_count==5]
  │       → [MotionStateTracker.estimate]
  │       → [TemporalBufferManager.encode → latents.shape[0]==5]
  │       → [SVICFGGate: cfg ∈ [5.0,6.0]]
  │       → [SVIScheduler: noise-aware LoRA 0.6/0.5/0.4]
  │       → SVI.generate(init_latents=latents)
  │       → [CLIPValidator: identity per segment ≥ 0.93 (RULE-89)]
  │       → [TemporalBufferManager.update] → context.evolve
  │
  └─ S-10: ContinuityValidationAgent
      └─ [ContinuityGate ≥ 0.90 + identity_per_segment] → [HRG-8]

PHASE 4: Audio Realism
  ├─ S-11: DialogueAgent (CosyVoice3)
  │   └─ [TimingGate ±0.1s] → [HRG-9]
  │
  ├─ S-12: LipSyncAgent (LatentSync-1.6)
  │   └─ [PhonemeGate ≥ 0.80] → [CLIPValidator: delta ≤ 0.03 (RULE-89)]
  │       → [IdentityStateTracker.update] → [HRG-10]
  │
  ├─ S-13: AmbientAudioAgent (MMAudio)
  ├─ S-14: MusicAgent (MusicGen-medium)
  └─ S-15: AudioMixingAgent (pydub/torchaudio)
      └─ [AudioPriorityGate: D > A > M]
          → [AudioQualityValidator: SNR ≥ 10 dB; peaks ≤ 0 dBFS (RULE-99)]
          → [CrossModalAlignmentValidator]
          → [HRG-11]

PHASE 5: Finalization
  ├─ S-16a: AssemblyAgent → final_video.mp4
  ├─ S-16b: ExportAgent → /workspace/output/{job_id}/{scene_id}/
  └─ S-16c: QualityAgent → pipeline_report + sla_summary + adaptive_state
       ├─ adaptive_memory.save()
       ├─ hrg_controller.save_log()  ← 11-checkpoint log
       ├─ identity_state_tracker.save_final_state()
       ├─ temporal_buffer_manager.clear_all()
       └─ State: COMPLETE → pod STOP
```
