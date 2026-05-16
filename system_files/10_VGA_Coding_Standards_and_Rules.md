# VGA Coding Standards and Rules
**Project:** Video Generation Automation (VGA) — Motivation System
**Version:** 17.2.0
**Status:** Authoritative Reference
**Audience:** Claude Code Agent, All Engineers

---

## Overview

This document is the complete, authoritative set of coding rules for the VGA system. **Every rule is MANDATORY. No rule may be selectively applied.**

**Retained from v15.0 (CGRL-01 through CGRL-64):** All rules unchanged.
**Retained from v16.0 (CGRL-65 through CGRL-84):** All rules unchanged.
**New in v17.0 (CGRL-85 through CGRL-100):** 16 new rules covering SceneCompositionAgent, TemporalEngine, temporal buffer integrity, SVI scheduling, motion state, identity state tracking, audio quality, cross-modal alignment, HRG checkpoint count, and the execute_stage() contract.

---

## CGRL-01 through CGRL-64 (Retained from v15.0 — unchanged)

All rules CGRL-01 through CGRL-64 are retained without modification.

---

## CGRL-65 through CGRL-84 (Retained from v16.0 — unchanged)

All rules CGRL-65 through CGRL-84 are retained without modification.

---

## CGRL-85: execute_stage() Contract Is Mandatory

```
Rule: Every pipeline stage MUST be executed via the execute_stage() function
      defined in master_orchestrator.py. Direct agent.run() calls outside
      execute_stage() are FORBIDDEN.

Rationale: SYSTEM DIRECTIVE v17. Ensures SystemGuard wrapping, predecessor
           output validation, HRG gating, output validation, and context.evolve()
           are applied uniformly across all 20 pipeline stages.

CORRECT:
  output, context = execute_stage(stage, input_data, context)

INCORRECT:
  output = agent.run(input_data)  ← missing SystemGuard, predecessor check, context.evolve
  context = context.evolve(output)

Test: chaos/test_execute_stage_contract.py
```

---

## CGRL-86: CompositionPlan Must Exist Before Image or Video Generation

```
Rule: No image generation agent (S-05, S-06A, S-06B, S-06C) and no video
      generation agent (S-08, S-09) may execute without a validated
      CompositionPlan present in ImmutableContext. This is enforced by
      CompositionPlanValidator.assert_in_context(context).

Rationale: RULE-88. Every visual stage requires explicit visual directives.
           Unguided generation produces inconsistent visual language.

CORRECT:
  composition_validator.assert_in_context(context)   ← raises if missing
  output = generate(input, context)

INCORRECT:
  output = generate(input, context)  ← no composition gate
  # or:
  output = generate(input, context, composition_plan=None)  ← None is FORBIDDEN

Raises: CompositionPlanValidationError
Test: chaos/test_composition_plan_missing.py
```

---

## CGRL-87: TemporalBuffer Must Always Contain Exactly 5 Frames

```
Rule: TemporalBuffer.frames.shape[0] MUST equal TEMPORAL_BUFFER_SIZE (= 5) at
      all entry and exit points of TemporalBufferManager. Any other frame count
      is a hard error that halts the pipeline.

Rationale: RULE-86. 5-frame rolling buffer is the architectural contract.
           Fewer frames degrades temporal continuity; more frames is undefined.

CORRECT:
  buffer = TemporalBufferManager.init(segment_1)
  assert buffer.frames.shape[0] == 5

INCORRECT:
  buffer.frames = segment_1.video_tensor  ← raw tensor; shape unknown
  # or:
  buffer.frames = frames[:3]              ← 3 frames is FORBIDDEN

Raises: TemporalBufferError
Test: chaos/test_temporal_buffer_error.py
```

---

## CGRL-88: SVI Conditioning Must Use Multi-Frame Latents (5-Frame)

```
Rule: When calling svi_wrapper.generate(), the init_latents argument MUST be
      a Tensor of shape (5, C', H', W'). Single-image latent conditioning is
      FORBIDDEN for SVI. This is the autoregressive architecture contract.

Rationale: RULE-87. SVI Pro 2 requires temporal context from prior frames.
           Single-image conditioning collapses to static frame generation.

CORRECT:
  latents = temporal_buffer_manager.encode(buffer)   # shape: (5, C', H', W')
  assert latents.shape[0] == 5
  segment_n = svi.generate(init_latents=latents, ...)

INCORRECT:
  latent = vae.encode(last_frame)                    ← shape: (C', H', W')
  segment_n = svi.generate(init_latents=latent, ...) ← single-frame FORBIDDEN

Raises: AutoregressiveViolationError
Test: chaos/test_autoregressive_gate.py
```

---

## CGRL-89: SVIScheduler Must Be Used for Every SVI Call

```
Rule: Every svi_wrapper.generate() call MUST receive the SVIScheduler instance
      as the lora_scheduler argument. Static LoRA weight assignment (passing a
      float) is FORBIDDEN. The scheduler applies different weights at different
      diffusion timesteps.

Rationale: RULE-86. Noise-aware LoRA scheduling prevents identity drift at
           high-noise timesteps and preserves detail at low-noise timesteps.

CORRECT:
  segment_n = svi.generate(
    init_latents=latents,
    lora_scheduler=self.scheduler,   ← SVIScheduler instance
    cfg=5.5,
    steps=30
  )

INCORRECT:
  segment_n = svi.generate(
    init_latents=latents,
    lora_weight=0.5,                 ← static weight; FORBIDDEN
    cfg=5.5,
    steps=30
  )

Implementation: svi_wrapper.py must call scheduler.apply_lora(timestep) at
               each diffusion step; not store a single weight from the call site.
Test: unit/test_svi_scheduler.py
```

---

## CGRL-90: SVI CFG Must Stay Within [5.0, 6.0]

```
Rule: The guidance scale (CFG) for SVI Pro 2 MUST be in [SVI_CFG_MIN, SVI_CFG_MAX]
      = [5.0, 6.0]. Out-of-range CFG MUST raise SVICFGViolationError; silent
      clamping is FORBIDDEN.

Rationale: RULE-86. cfg < 5.0: identity/structure drift. cfg > 6.0: color
           banding artifacts. Empirically confirmed on RTX 4090 with SVI Pro 2.

CORRECT:
  cfg = self.scheduler.assert_cfg_valid(cfg)   ← raises if out of range
  segment = svi.generate(cfg=cfg, ...)

INCORRECT:
  cfg = max(5.0, min(6.0, requested_cfg))      ← silent clamp; FORBIDDEN
  segment = svi.generate(cfg=cfg, ...)

Raises: SVICFGViolationError
Test: chaos/test_svi_cfg_violation.py
```

---

## CGRL-91: Segments Must Be Generated Sequentially

```
Rule: Segment[n] MUST NOT be generated before Segment[n-1] is complete.
      Parallel segment generation is FORBIDDEN. The autoregressive contract
      requires buffer from Segment[n-1] before generating Segment[n].

Rationale: RULE-87. Autoregressive Invariant I4. Each segment's buffer
           depends on the previous segment's output.

CORRECT:
  for n, plan in enumerate(plans[1:], start=2):
    segment_n = generate_with_retry(buffer, plan)  ← sequential
    buffer = buffer_manager.update(buffer, segment_n)

INCORRECT:
  futures = [executor.submit(generate, plan) for plan in plans]  ← parallel; FORBIDDEN
  segments = [f.result() for f in futures]

Test: integration/test_temporal_engine.py::test_sequential_ordering
```

---

## CGRL-92: context.evolve() Must Be Called After Every Segment

```
Rule: After every segment generated by TemporalEngine, context.evolve() MUST
      be called with the updated temporal_state (buffer + segment_index).
      The autoregressive loop must not proceed to Segment[n+1] with stale context.

Rationale: Autoregressive Invariant I5. Stale context causes motion_state and
           identity_state to diverge from actual pipeline state.

CORRECT:
  buffer = buffer_manager.update(buffer, segment_n)
  context = context.evolve({
    "temporal_state": TemporalState(buffer=buffer, segment_index=n)
  })
  # Only now proceed to Segment[n+1]

INCORRECT:
  buffer = buffer_manager.update(buffer, segment_n)
  # ← context not evolved; next iteration uses stale context

Test: unit/test_immutable_context.py::test_temporal_state_evolution
```

---

## CGRL-93: MotionStateTracker Must Be Called Before Every SVI Generation

```
Rule: MotionStateTracker.estimate(buffer.frames) MUST be called before each
      SVI generation call. The result MUST be used in the segment prompt and
      stored via context.evolve(). Results MUST NOT be cached across segments.

Rationale: Each segment may have different motion; stale motion state causes
           motion reset artifacts between segments (RULE-87 Invariant I1).

CORRECT:
  for n, plan in enumerate(plans[1:], start=2):
    motion_state = motion_tracker.estimate(buffer.frames)   ← called every segment
    context = context.evolve({"motion_state": motion_state})
    segment_n = svi.generate(prompt=build_prompt(plan, motion_state), ...)

INCORRECT:
  motion_state = motion_tracker.estimate(initial_buffer.frames)  ← called once; cached
  for n, plan in enumerate(plans[1:], start=2):
    segment_n = svi.generate(prompt=build_prompt(plan, motion_state), ...)  ← stale

Test: unit/test_motion_state_tracker.py::test_per_segment_estimation
```

---

## CGRL-94: Identity Validation Must Occur at Every Phase Boundary

```
Rule: CLIPValidator.score(frame, char_identity_ref) MUST be called after:
      - Each base image (S-05)
      - Each image editing sub-stage output (S-06A, S-06B, S-06C)
      - Each refined image (S-07)
      - Each SVI segment keyframe (S-09)
      - Each lip-synced segment (S-12)
      Skipping any of these validation points is a RULE-89 violation.

Rationale: Cross-phase identity validation is the primary mechanism preventing
           character drift. Any unvalidated transition may allow silent drift.

CORRECT:
  for i, image in enumerate(images):
    clip_score = clip_validator.score(image, char_identity_ref)
    assert clip_score >= CLIP_IDENTITY_THRESHOLD, f"Stage {stage_id} image {i}"
    identity_tracker.update(char_identity_ref, image, stage_id)

INCORRECT:
  # skipping validation between S-06C and S-07
  image_07 = refine(image_06c)
  # No clip_validator call between the two stages ← RULE-89 violation

Test: integration/test_identity_cross_phase.py
```

---

## CGRL-95: char_identity_ref Must Be Frozen After S-07

```
Rule: After `image_refinement_agent.py` freezes `char_identity_ref` in
      ImmutableContext, no agent may recompute or replace it. All downstream
      CLIPValidator calls MUST use the frozen reference. Identity reference
      mutation detection via hash comparison is MANDATORY in CLIPValidator.

Rationale: RULE-95 (cross-stage identity propagation). Recomputing the
           reference mid-pipeline causes identity drift measurements to
           reference a different baseline, invalidating all prior tracking.

CORRECT:
  # In image_refinement_agent.py (S-07):
  char_identity_ref = clip_encoder(best_refined_image)
  context = context.evolve({"identity_state": IdentityState(embedding_vector=char_identity_ref, ...)})

  # In clip_validator.py — hash guard:
  current_hash = hash(identity_ref.numpy().tobytes())
  assert current_hash == self._frozen_ref_hash, "Identity reference mutated — CRITICAL"

INCORRECT:
  # In temporal_engine.py:
  char_identity_ref = clip_encoder(segment_n.keyframe)  ← recomputing; FORBIDDEN

Raises: IdentityReferenceCorruptionError
Test: chaos/test_identity_reference_immutability.py
```

---

## CGRL-96: IdentityStateTracker Must Be Called After Every CLIPValidator Check

```
Rule: Every CLIPValidator.score() call that checks identity MUST be followed
      by IdentityStateTracker.update(char_identity_ref, frame, stage_id).
      CLIPValidator and IdentityStateTracker are inseparable in any identity-
      checking context.

Rationale: Cumulative drift tracking requires every per-stage measurement.
           Skipping IdentityStateTracker.update() at any phase produces an
           incomplete drift history and may fail to detect cumulative drift.

CORRECT:
  clip_score = clip_validator.score(frame, char_identity_ref)
  identity_tracker.update(char_identity_ref, frame, stage_id)   ← always paired

INCORRECT:
  clip_score = clip_validator.score(frame, char_identity_ref)
  # ← No IdentityStateTracker.update call; drift tracking gap

Test: unit/test_identity_state_tracker.py::test_paired_with_clip_validator
```

---

## CGRL-97: AudioQualityValidator Must Be Called Before Audio Storage Write

```
Rule: audio_quality_validator.validate() MUST be called after AudioMixingAgent
      produces mixed audio and BEFORE writing the mixed audio to storage.
      Mixed audio that fails SNR or peak level checks MUST be re-mixed or
      normalized before writing.

Rationale: RULE-99. Writing substandard audio to storage propagates the
           problem downstream to assembly, export, and the final product.

CORRECT:
  mixed = mixer.mix(dialogue, ambient, music)
  record = audio_quality_validator.validate(mixed, dialogue, scene_id, job_id)
  if record.peak_db > MAX_PEAK_DBFS:
    mixed = audio_quality_validator.normalize(mixed)
  storage.write(mixed_path, mixed)   ← write AFTER validation + normalization

INCORRECT:
  mixed = mixer.mix(dialogue, ambient, music)
  storage.write(mixed_path, mixed)   ← write BEFORE validation; FORBIDDEN

Test: unit/test_audio_quality_validator.py::test_called_before_storage
```

---

## CGRL-98: CrossModalAlignmentValidator Must Be Called Before HRG-11

```
Rule: cross_modal_alignment_validator.validate_alignment() MUST be called
      by AudioMixingAgent after mixing completes and BEFORE HRG-11 display.
      The alignment report MUST be included in HRG-11 display data.

Rationale: RULE-96. Misaligned video and audio discovered after HRG-11
           approval is very costly. Cross-modal alignment must be surfaced
           to the human reviewer at HRG-11.

CORRECT:
  alignment_report = cross_modal_validator.validate_alignment(
    video_segments, audio_segments, scene_id
  )
  storage.write_cross_modal_alignment(scene_id, alignment_report)
  # HRG-11 display includes alignment_report.all_passed badge

INCORRECT:
  # Skip cross-modal validation; show only SNR/peak at HRG-11

Test: integration/test_cross_modal_alignment.py
```

---

## CGRL-99: SceneCompositionAgent Must Retry on Schema Validation Failure

```
Rule: SceneCompositionAgent MUST retry up to COMPOSITION_MAX_RETRIES times
      (default: 3) when CompositionPlanValidator.validate() raises. If all
      retries are exhausted, it MUST raise CompositionPlanValidationError.
      The agent MUST NOT proceed with a partially-filled plan.

Rationale: Qwen may occasionally produce malformed JSON or omit optional-
           looking fields that are mandatory. Retry with lower temperature
           typically resolves the issue.

CORRECT:
  for attempt in range(COMPOSITION_MAX_RETRIES):
    plan = model.generate_structured(prompt)
    if composition_validator.validate(plan):
      return plan
    # Adjust temperature; continue loop
  raise CompositionPlanValidationError(scene_id, reason="all retries failed")

INCORRECT:
  plan = model.generate_structured(prompt)
  return plan  ← no validation; missing fields will cause downstream failures

Raises: CompositionPlanValidationError after all retries
Test: unit/test_scene_composition_agent.py::test_retry_on_validation_failure
```

---

## CGRL-100: HRGController Checkpoint Count Is Authoritative

```
Rule: The number of valid HRG checkpoints is EXACTLY 11 (HRG-1 through HRG-11).
      This is a hard-coded architectural constant (HRG_CHECKPOINT_COUNT = 11).
      No code may add checkpoints dynamically. No code may reference checkpoints
      outside this range (e.g., HRG-0 or HRG-12).

Rationale: HRG checkpoints map to specific pipeline gates. Dynamic addition
           would violate the stage-to-checkpoint mapping in MasterOrchestrator.
           The 11-checkpoint layout is the authoritative v17.0 architecture.

CORRECT:
  VALID_CHECKPOINTS = {f"HRG-{i}" for i in range(1, 12)}  # 11 checkpoints
  assert checkpoint in VALID_CHECKPOINTS

INCORRECT:
  VALID_CHECKPOINTS = {f"HRG-{i}" for i in range(1, 13)}  ← 12 checkpoints; FORBIDDEN
  hrg_controller.add_checkpoint("HRG-12")                  ← dynamic addition; FORBIDDEN

Test: unit/test_hrg_controller.py::test_checkpoint_count_invariant

---

RULE SUMMARY (v17.0 complete — CGRL-01 through CGRL-100):

v15.0 rules:   CGRL-01 through CGRL-64
v16.0 rules:   CGRL-65 through CGRL-84
v17.0 rules:   CGRL-85 through CGRL-100

  CGRL-85: execute_stage() contract mandatory for all stages
  CGRL-86: CompositionPlan gate before all image/video generation
  CGRL-87: TemporalBuffer always exactly 5 frames
  CGRL-88: SVI conditioning uses multi-frame latents (5-frame, not single-image)
  CGRL-89: SVIScheduler mandatory for every SVI call
  CGRL-90: SVI CFG within [5.0, 6.0]; no silent clamping
  CGRL-91: Segments generated sequentially; no parallel generation
  CGRL-92: context.evolve() called after every segment in TemporalEngine loop
  CGRL-93: MotionStateTracker called before every SVI generation; no caching
  CGRL-94: Identity validation at every phase boundary (RULE-89)
  CGRL-95: char_identity_ref frozen after S-07; hash guard in CLIPValidator
  CGRL-96: IdentityStateTracker always paired with CLIPValidator
  CGRL-97: AudioQualityValidator called before audio storage write
  CGRL-98: CrossModalAlignmentValidator called before HRG-11
  CGRL-99: SceneCompositionAgent retries up to COMPOSITION_MAX_RETRIES
  CGRL-100: HRGController checkpoint count is exactly 11; no dynamic addition
```

---

## CGRL-101: TemporalEngine MUST Use Explicit Loop (v17.1 — CGS-17 Enforcement)

```
Rule: The implementation of temporal_engine.py MUST use an explicit for loop
      for all segment generation beyond Segment_1. This rule formalizes
      CGS-17 at the coding standards level.

Rationale: The autoregressive contract (TEMPORAL EXECUTION CONTRACT v17.1)
           can ONLY be correctly honored by a sequential loop that updates
           the TemporalBuffer between each segment generation. Any non-loop
           pattern silently breaks temporal continuity.

CORRECT:
  for n, plan in enumerate(segment_plans[1:], start=2):
      assert buffer.frames.shape[0] == TEMPORAL_BUFFER_SIZE
      latents = self.buffer_manager.encode(buffer)
      segment_n = self.svi.generate(init_latents=latents, ...)
      validate_segment(segment_n)
      buffer = self.buffer_manager.update(buffer, segment_n)
      assert buffer.frames.shape[0] == TEMPORAL_BUFFER_SIZE
      segments.append(segment_n)

FORBIDDEN:
  [self.svi.generate(buffer, p) for p in segment_plans[1:]]  ← comprehension; no buffer update
  asyncio.gather(*[gen(p) for p in plans])                   ← parallel; violates contract
  self.svi.generate_all(segment_plans, init_image)            ← batch call; FORBIDDEN
  segments = merge(independently_generated_segments)          ← post-hoc merge; FORBIDDEN

ENFORCEMENT: tests/regression/test_temporal_regression_guard.py performs
             AST-level verification of the loop structure on every CI run.

Test: tests/integration/test_temporal_loop_integration.py (all 9 cases)
      tests/regression/test_temporal_regression_guard.py
```

---

## CGRL-102: Universal Segment Validation — All Three Validators Required (v17.1)

```
Rule: Every generated video segment MUST pass ALL THREE validators before
      it may be appended to the segment list or used to update TemporalBuffer.
      Passing two-of-three is NOT sufficient.

The three required validators:

  1. CLIPValidator.score(segment.keyframe, char_identity_ref) >= CLIP_IDENTITY_THRESHOLD
     → identity preserved across the generated frame [RULE-89]

  2. ContinuityValidator.score_segment(buffer.last_frame, segment.first_frame)
     >= SEGMENT_CONTINUITY_MIN
     → no visual discontinuity at segment boundary

  3. TemporalConsistencyValidator.assert_buffer_valid(buffer)
     → buffer size == 5; frame resolution consistent; device == cpu

CORRECT:
  clip_ok   = clip_validator.score(seg.keyframe, char_identity_ref) >= threshold
  cont_ok   = continuity_validator.score_segment(buf.last_frame, seg.first_frame) >= min_score
  buffer_ok = temporal_consistency_validator.assert_buffer_valid(buffer)

  if not (clip_ok and cont_ok and buffer_ok):
      segment_n = retry_controller.retry(...)

FORBIDDEN:
  if clip_ok:  # skipping continuity or buffer validation
      segments.append(segment_n)  ← FORBIDDEN; partial validation

RATIONALE: All three failures represent distinct failure modes (identity drift,
           visual jump cut, buffer corruption). Each requires independent detection.

Test: tests/integration/test_temporal_loop_integration.py::test_continuity_score_above_threshold
      tests/integration/test_identity_cross_phase.py
```

---

## CGRL-103: TemporalBuffer Device Discipline (v17.1)

```
Rule: TemporalBuffer frames MUST be CPU tensors at all times EXCEPT during
      TemporalBufferManager.encode(). This is the Buffer Device Rule from
      TEMPORAL EXECUTION CONTRACT v17.1.

CORRECT:
  buffer = buffer_manager.init(segment_1)  # CPU
  # ... between segments: buffer.frames.device == "cpu"
  latents = buffer_manager.encode(buffer)  # GPU ONLY inside encode()
  # latents returned as CPU tensor; buffer remains CPU

FORBIDDEN:
  buffer.frames = buffer.frames.to("cuda")  # ← outside encode(); FORBIDDEN
  segment_n = svi.generate(init_latents=buffer.frames.cuda())  # ← bypasses encode(); FORBIDDEN

ENFORCEMENT: CGRL-103 is checked by TemporalBufferRecord.device_at_log_time validator
             (must equal "cpu" at every log point)

Test: tests/integration/test_temporal_loop_integration.py::test_buffer_is_cpu_resident_between_segments
```

---

## RULE SUMMARY UPDATE (v17.1)

```
v17.1 additions:
  CGRL-101: TemporalEngine explicit for loop mandatory (CGS-17 at code level)
  CGRL-102: All three segment validators required (CLIP + continuity + buffer)
  CGRL-103: TemporalBuffer CPU-residency discipline between segments

Complete rule set:
  v15.0: CGRL-01 through CGRL-64
  v16.0: CGRL-65 through CGRL-84
  v17.0: CGRL-85 through CGRL-100
  v17.1: CGRL-101 through CGRL-103
```



---

## v17.2 Coding Standards Additions (NEW)

### CGRL-101: Temporal Authority Guard — Mandatory Integration

Every call to SVI, TemporalBuffer.update(), and segment iteration control MUST be
guarded by `TemporalAuthorityGuard`. No exceptions. See CGS-19.

```python
# REQUIRED at top of temporal_engine.py generate_scene():
TemporalAuthorityGuard.guard_segment_iteration(self.__class__.__qualname__)

# REQUIRED at top of svi_wrapper.py generate():
TemporalAuthorityGuard.guard_svi_invoke(caller_qualname)

# REQUIRED at top of temporal_buffer_manager.py update():
TemporalAuthorityGuard.guard_buffer_update(caller_qualname)
```

### CGRL-102: SVI Generation Lineage — All Fields Required

Every `SVIGenerationRecord` logged MUST include lineage fields. See CGS-18.

```python
# FORBIDDEN (no lineage):
record = SVIGenerationRecord(segment_id=seg_id, cfg=5.5, ...)

# REQUIRED (with lineage):
record = SVIGenerationRecord(
    segment_id=seg_id,
    previous_segment_id=segment_n_minus_1.id,
    source_buffer_frame_ids=[f.id for f in buffer.frames],  # exactly 5
    buffer_timestamp_range=(buffer.timestamps[0], buffer.timestamps[-1]),
    generation_mode="autoregressive",
    cfg=5.5, ...
)
```

### CGRL-103: System Certification — Final Gate

`SystemCertificationValidator.certify()` MUST be called in `quality_agent.py`
before any PipelineReport is written. See CGS-20.

```python
# FORBIDDEN (no certification):
storage.write_pipeline_report(report)

# REQUIRED:
cert = SystemCertificationValidator(tracer).certify(report_dict)
report.certified = cert["certified"]
storage.write_pipeline_report(report)
```

### CGRL-104: Cross-Modal Validation — Unified Contract

All cross-modal validation MUST use `CrossModalValidationUnified` for per-segment
validation. Fragmented individual checks are deprecated.

```python
# DEPRECATED pattern:
check_duration_alignment(video, audio)
check_phoneme_score(synced)
check_identity_delta(frame)

# REQUIRED pattern:
record = CrossModalValidationUnified(clip_validator, tracer).validate(
    scene_id, segment_id, video, audio, synced_frame,
    frame_before, frame_after, char_identity_ref,
    continuity_score, phoneme_score
)
storage.write_cross_modal_validation(job_id, scene_id, record)
```
