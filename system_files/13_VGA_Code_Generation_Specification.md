# VGA Code Generation Specification
**Project:** Video Generation Automation (VGA) — Motivation System
**Version:** 17.2.0
**Status:** Authoritative Reference
**Audience:** Claude Code Agent

---

## Overview

This document is the authoritative set of instructions for Claude Code when generating implementation files for the VGA v17.0 system. **Every instruction here is MANDATORY. Claude Code must follow all instructions in this document without exception.**

**Retained from v15.0 (CGS-01 through CGS-06):** All code generation rules unchanged.
**Retained from v16.0 (CGS-07 through CGS-08):** All v16.0 rules unchanged.
**New in v17.0 (CGS-09 through CGS-16):** 8 new rules covering the v17.0 architecture additions.
**New in v17.1 (CGS-17):** Critical enforcement rule — TemporalEngine MUST be implemented as an explicit loop; any single-call or batch pattern is FORBIDDEN.
**New in v17.2 (CGS-18 through CGS-20):** Three new rules covering: SVI generation lineage traceability, TemporalEngine authority guard integration, and system certification validator integration.

---

## CGS-01 through CGS-06 (Retained from v15.0 — unchanged)

All rules CGS-01 through CGS-06 are retained without modification.

---

## CGS-07 through CGS-08 (Retained from v16.0 — unchanged)

All rules CGS-07 through CGS-08 are retained without modification.

---

## CGS-09: Generate All v17.0 Files in Phase 12 Order

```
Rule: Claude Code MUST generate all v17.0 files in the exact order specified in
      Doc 11 Phase 12 (Development Sequence). The dependency chain is strict:
      foundation files MUST exist before files that import from them.

Phase 12 mandatory generation order:
  1. settings.py (v17.0 constants appended)
  2. schemas.py (v17.0 schemas appended)
  3. exceptions.py (v17.0 exceptions appended)
  4. state/immutable_context.py (5-dim update)
  5. state/context_factory.py (5-dim init)
  6. core/storage.py (v17.0 paths)
  7. validation/composition_validator.py
  8. validation/audio_quality_validator.py
  9. validation/cross_modal_alignment_validator.py
  10. config/prompts/composition_prompts.py
  11. temporal/temporal_buffer_manager.py
  12. temporal/svi_scheduler.py
  13. temporal/motion_state_tracker.py
  14. temporal/temporal_retry_controller.py
  15. models/wrappers/svi_wrapper.py (update)
  16. identity/identity_state_tracker.py  ← BEFORE temporal_engine
  17. temporal/temporal_engine.py
  18. agents/scene_composition_agent.py
  19. agents/video_segment_generator.py
  20. agents/base_image_agent.py (update)
  21. agents/scene_expansion_agent.py (update)
  22. agents/image_refinement_agent.py (update)
  23. agents/continuity_validation_agent.py (update)
  24. agents/lip_sync_agent.py (update)
  25. agents/audio_mixing_agent.py (update)
  26. agents/quality_agent.py (update)
  27. core/hrg_controller.py (update)
  28. core/master_orchestrator.py (update)
  29. api/routes/temporal.py
  30. api/routes/identity.py
  31. api/routes/audio.py
  32. api/routes/composition.py
  33. api/routes/hrg.py (update)
  34. api/main.py (update)
  35. ui/components/hrg_panels/hrg_2_scene_plan.py
  36. ui/components/hrg_panels/hrg_4_composition.py
  37. ui/components/temporal_engine_panel.py
  38. hrg panel updates (8, 10, 11)
  39. hrg panel renumbering (3, 5, 6, 7, 9)
  40. core/schema_migrations.py (update)
  41. bootstrap.py (steps 6Z-p through 6Z-z)
  42. all v17.0 tests

VERIFICATION: After generating each file, Claude Code must confirm that
  all imports in the generated file resolve to previously-generated files
  in this sequence or to v16.0/earlier files already on disk.
```

---

## CGS-10: Temporal Engine — Multi-Frame Latent Enforcement

```
Rule: When generating svi_wrapper.py, temporal_buffer_manager.py, and
      temporal_engine.py, Claude Code MUST enforce the multi-frame latent
      contract at the function signature level, not merely via runtime assertion.

REQUIRED: The `generate()` method of SVIWrapper MUST have:
  def generate(
    self,
    init_latents: torch.Tensor,   ← type-annotated; NOT init_image
    lora_scheduler: "SVIScheduler",  ← NOT lora_weight: float
    cfg: float,
    steps: int,
    temporal_attention: bool = True,
    ...
  ) -> "VideoSegment": ...

FORBIDDEN signature patterns:
  def generate(self, init_image: PIL.Image, ...)    ← single image; FORBIDDEN
  def generate(self, ..., lora_weight: float, ...)  ← static weight; FORBIDDEN

The function docstring MUST state:
  "init_latents MUST be shape (5, C', H', W'). Single-frame tensor is FORBIDDEN."
  "lora_scheduler MUST be SVIScheduler instance. Static float weight is FORBIDDEN."

Runtime assertion REQUIRED inside the function body:
  if init_latents.shape[0] != TEMPORAL_BUFFER_SIZE:
    raise AutoregressiveViolationError(...)
```

---

## CGS-11: SceneCompositionAgent — Structured Output Pattern

```
Rule: When generating scene_composition_agent.py, Claude Code MUST use
      Qwen structured output mode with explicit JSON schema enforcement.
      The generation loop MUST include retry logic with COMPOSITION_MAX_RETRIES.

REQUIRED pattern:
  for attempt in range(COMPOSITION_MAX_RETRIES):
    try:
      raw = self.model_wrapper.generate(
        prompt=self.prompt_builder.build_composition_prompt(scene_data),
        temperature=0.3,    ← lower temp for structured output reliability
        max_tokens=1024
      )
      plan_dict = json.loads(raw)
      plan_dict["scene_id"] = scene_id
      plan_dict["schema_version"] = SCHEMA_VERSION
      validated = CompositionPlanSchema(**plan_dict)
      self.composition_validator.validate(validated)  ← always validate
      return validated
    except (json.JSONDecodeError, ValidationError):
      if attempt == COMPOSITION_MAX_RETRIES - 1:
        raise CompositionPlanValidationError(scene_id, reason="...")

FORBIDDEN: Returning plan without calling CompositionPlanValidator.validate()
FORBIDDEN: Proceeding with a plan that has any None or empty string field
```

---

## CGS-12: Identity State Tracker — Delta Computation Pattern

```
Rule: When generating identity_state_tracker.py, Claude Code MUST use
      cosine similarity for delta computation and cumulative addition for
      drift_score tracking. The pattern must match Doc 03 §50.2 exactly.

REQUIRED pattern:
  def update(self, char_identity_ref: torch.Tensor, new_frame, stage_id: str) -> dict:
    e_new = self.clip_encoder.encode(new_frame)
    cos_sim = torch.nn.functional.cosine_similarity(
      char_identity_ref.unsqueeze(0), e_new.unsqueeze(0)
    ).item()
    delta = 1.0 - cos_sim
    self._drift_score += delta
    self._history.append(delta)

    if self._drift_score > IDENTITY_CUMULATIVE_DRIFT_THRESHOLD:
      raise IdentityCumulativeDriftError(...)

FORBIDDEN: Using Euclidean distance instead of cosine similarity
FORBIDDEN: Using drift_score = delta (replace, not accumulate)
FORBIDDEN: Storing embedding_vector inside IdentityStateTracker
           (embedding is owned by ImmutableContext; tracker owns delta/history only)
```

---

## CGS-13: Audio Quality — SNR Computation Pattern

```
Rule: When generating audio_quality_validator.py, Claude Code MUST implement
      SNR as dialogue RMS minus background RMS in dB, and peak level as
      max sample value in dBFS. These computations must match Doc 03 §51 exactly.

REQUIRED pattern:
  def compute_snr(self, mixed: AudioSegment, dialogue: AudioSegment) -> float:
    dialogue_rms = dialogue.rms
    background_rms = max(1, mixed.rms - dialogue_rms)
    if dialogue_rms <= 0:
      return 0.0
    return 20.0 * math.log10(dialogue_rms / background_rms)

  def compute_peak_db(self, audio: AudioSegment) -> float:
    peak = audio.max
    if peak <= 0:
      return -99.0
    return 20.0 * math.log10(peak / 32768.0)  # normalised to 16-bit

REQUIRED normalization pattern:
  def normalize(self, mixed: AudioSegment, target_peak_db: float = -1.0) -> AudioSegment:
    current_peak = self.compute_peak_db(mixed)
    gain_reduction = target_peak_db - current_peak
    return mixed.apply_gain(gain_reduction)

FORBIDDEN: Using pydub's built-in normalize() without custom target
FORBIDDEN: Raising an exception in validate() on quality failure
           (validate() returns record only; caller decides action)
```

---

## CGS-14: MotionStateTracker — Optical Flow Fallback Pattern

```
Rule: When generating motion_state_tracker.py, Claude Code MUST implement
      primary optical flow via torchvision with cv2 fallback. Both paths
      MUST produce a flow tensor of shape (H, W, 2).

REQUIRED pattern:
  def _compute_optical_flow(self, frame_a: torch.Tensor, frame_b: torch.Tensor) -> torch.Tensor:
    try:
      # Primary: torchvision optical_flow (if available)
      import torchvision.models.optical_flow as of_module
      # Use RAFT or Farneback from torchvision
      flow = self._torchvision_flow(frame_a, frame_b)
    except (ImportError, AttributeError):
      # Fallback: cv2
      import cv2
      import numpy as np
      a_gray = (frame_a.mean(dim=0).numpy() * 255).astype(np.uint8)
      b_gray = (frame_b.mean(dim=0).numpy() * 255).astype(np.uint8)
      flow_np = cv2.calcOpticalFlowFarneback(a_gray, b_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
      flow = torch.from_numpy(flow_np)   # shape: (H, W, 2)
    return flow  # MUST be (H, W, 2)

The DEVIATION_LOG.md must be updated if torchvision is unavailable in production.
```

---

## CGS-15: HRG Controller — 11-Checkpoint Enforcement

```
Rule: When updating hrg_controller.py, Claude Code MUST update ONLY the
      checkpoint count (9 → 11) and add HRG-2 and HRG-4 entries.
      All existing HRG-1, HRG-3, HRG-5 through HRG-11 logic MUST be
      retained without modification to preserve their blocking behavior.

REQUIRED change:
  BEFORE: VALID_CHECKPOINTS = {f"HRG-{i}" for i in range(1, 10)}  # 9 checkpoints
  AFTER:  VALID_CHECKPOINTS = {f"HRG-{i}" for i in range(1, 12)}  # 11 checkpoints

  BEFORE: self._events = {checkpoint: threading.Event() for checkpoint in self.VALID_CHECKPOINTS}
  AFTER:  same pattern; just 11 events now

HRG-2 DISPLAY DATA requirement:
  The require_approval() call for HRG-2 receives HRG2DisplayData as display_data.
  It MUST include: scenes list, segments_per_scene dict, total_scenes, total_segments.

HRG-4 DISPLAY DATA requirement:
  The require_approval() call for HRG-4 receives HRG4DisplayData as display_data.
  It MUST include all 6 CompositionPlan fields plus validation_passed boolean.

FORBIDDEN: Reducing checkpoint count below 11
FORBIDDEN: Adding checkpoint above HRG-11 dynamically
FORBIDDEN: Modifying the blocking behavior of require_approval() (must use threading.Event.wait)
```

---

## CGS-16: execute_stage() — Mandatory Integration in All Stage Calls

```
Rule: When updating master_orchestrator.py, Claude Code MUST replace ALL
      direct agent.run() calls with execute_stage() calls. This is SYSTEM
      DIRECTIVE v17. No exceptions.

For EVERY stage in STAGE_ORDER:
  BEFORE (v16.0 pattern):
    output = self.stage_registry[stage_id].agent.run(input_data, context)
    context = context.evolve(output)

  AFTER (v17.0 pattern — SYSTEM DIRECTIVE):
    output, context = execute_stage(
      stage=self.stage_registry[stage_id],
      input_data=input_data,
      context=context
    )

The execute_stage() function body MUST include:
  1. SystemGuard.execute(stage)
  2. authority_manager.validate(...)
  3. _validate_previous_output(stage, context)  ← raises MissingPredecessorOutputError if missing
  4. hrg_controller.require_approval(...) if stage.requires_hrg
  5. output = stage.run(input_data, context)
  6. _validate_output(stage, output)
  7. context = context.evolve(output)  ← MANDATORY; must not be skipped
  8. return output, context

FORBIDDEN: Calling stage.run() directly from _execute_stage() without the wrapper
FORBIDDEN: Calling context.evolve() in _execute_stage() separately from execute_stage()
FORBIDDEN: Skipping any of the 8 steps in execute_stage()

VERIFICATION: After updating master_orchestrator.py, Claude Code MUST confirm
  that no direct agent.run() call exists outside execute_stage().
```

---

## CGS-17: TemporalEngine — Explicit Loop Implementation Mandatory (v17.1 — CRITICAL)

```
Rule: Claude Code MUST implement TemporalEngine as an explicit iterative loop.

This rule exists because the autoregressive contract can only be correctly
implemented as a sequential loop. Any other pattern silently violates the
TEMPORAL EXECUTION CONTRACT v17.1.

REQUIRED implementation structure (mandatory; no alternatives):

  def generate_scene(self, segment_plans, segment_1, context, char_identity_ref, trace_id):
      buffer = self.buffer_manager.init(segment_1)
      assert buffer.frames.shape[0] == TEMPORAL_BUFFER_SIZE  # MANDATORY assertion

      segments = [segment_1]

      for n, segment_plan in enumerate(segment_plans[1:], start=2):
          # (loop body: buffer gate → motion estimate → encode → SVI generate → validate → update)
          # See Doc 04 §3.10 for full loop body specification
          ...

      return segments, context

FORBIDDEN patterns (any of these in temporal_engine.py = CGS-17 VIOLATION):

  Pattern A — Single-call batch generation:
    segments = self.svi.generate_all_segments(scene_plan, init_image)
    FORBIDDEN: generates all segments in one call; violates autoregressive contract

  Pattern B — List comprehension segment generation:
    segments = [self.svi.generate(buffer, plan) for plan in scene_plan[1:]]
    FORBIDDEN: comprehensions do not update the buffer between iterations

  Pattern C — Pre-generation and merge:
    raw_segments = [generate_segment(plan) for plan in scene_plan]
    final = merge_segments(raw_segments)
    FORBIDDEN: post-hoc merge; violates continuity contract

  Pattern D — Parallel/async segment generation:
    tasks = [asyncio.create_task(generate(plan)) for plan in scene_plan]
    segments = await asyncio.gather(*tasks)
    FORBIDDEN: parallel execution; each segment must condition on previous

VERIFICATION: After generating temporal_engine.py, Claude Code MUST:
  1. Confirm the file contains exactly one for loop over segment_plans[1:]
  2. Confirm buffer is updated inside the loop body (not outside or after)
  3. Confirm no list comprehension generates segments
  4. Confirm no asyncio.gather or equivalent parallelism is used
  5. Confirm the assertion assert buffer.frames.shape[0] == TEMPORAL_BUFFER_SIZE
     appears both before the loop AND inside the loop body

CROSS-REFERENCE:
  - RULE-87: Autoregressive generation; multi-frame latents; sequential
  - TEMPORAL EXECUTION CONTRACT v17.1 Constraint C-3 (batch FORBIDDEN)
  - TEMPORAL EXECUTION CONTRACT v17.1 Constraint C-4 (post-hoc merge FORBIDDEN)
  - Doc 04 §3.10 HARD RULES
  - Doc 11 §12.18 temporal loop integration tests (enforce this rule at test level)
```

---

## CGS-18: SVI Generation Lineage — Traceability Enforcement (NEW v17.2)

```
Rule: When generating or updating temporal_engine.py, svi_wrapper.py, and
      any code that produces SVIGenerationRecord, Claude Code MUST ensure
      that every SVI generation is traceable to a specific TemporalBuffer.

REQUIRED fields in every SVIGenerationRecord:
  - previous_segment_id: str         ← ID of Segment_n that produced the buffer used
  - source_buffer_frame_ids: List[str]  ← MUST have exactly 5 frame identifiers
  - buffer_timestamp_range: Tuple[float, float]  ← (oldest_ts, newest_ts) from buffer
  - generation_mode: str             ← MUST be "autoregressive"

REQUIRED validation in temporal_engine.py before logging SVIGenerationRecord:
  assert svi_record.source_buffer_frame_ids is not None, "SVI lineage REQUIRED"
  assert len(svi_record.source_buffer_frame_ids) == 5, "Must trace to 5 frames"
  assert svi_record.generation_mode == "autoregressive", "Mode MUST be autoregressive"

FORBIDDEN:
  - Logging SVIGenerationRecord without source_buffer_frame_ids
  - Setting generation_mode to any value other than "autoregressive"
  - Generating a segment without setting previous_segment_id

RATIONALE: Without lineage traceability, it is impossible to audit whether the
autoregressive contract was honoured. Missing lineage = INVALID GENERATION.
```

---

## CGS-19: TemporalEngine Authority Guard Integration (NEW v17.2)

```
Rule: When generating or updating temporal_engine.py, svi_wrapper.py, and
      any other component that performs protected temporal operations,
      Claude Code MUST integrate TemporalAuthorityGuard.

REQUIRED pattern in temporal_engine.py generate_scene() method:
  # At top of method:
  from vga.temporal.temporal_authority_guard import TemporalAuthorityGuard
  TemporalAuthorityGuard.guard_segment_iteration(
      self.__class__.__module__ + "." + self.__class__.__qualname__
  )

REQUIRED pattern in svi_wrapper.py generate() method:
  # At top of method:
  from vga.temporal.temporal_authority_guard import TemporalAuthorityGuard
  TemporalAuthorityGuard.guard_svi_invoke(
      caller_qualname  # passed as parameter from temporal_engine
  )

REQUIRED pattern in temporal_buffer_manager.py update() method:
  # At top of method:
  from vga.temporal.temporal_authority_guard import TemporalAuthorityGuard
  TemporalAuthorityGuard.guard_buffer_update(
      caller_qualname  # passed as parameter from temporal_engine
  )

FORBIDDEN:
  - Calling SVI without passing caller_qualname for authority check
  - Calling TemporalBufferManager.update() from any file other than temporal_engine.py
  - Disabling or bypassing TemporalAuthorityGuard checks in production code

RATIONALE: Without authority enforcement, any component could silently call SVI
or modify the buffer, violating the autoregressive contract without detection.
```

---

## CGS-20: System Certification Validator Integration (NEW v17.2)

```
Rule: When updating quality_agent.py (S-16c), Claude Code MUST integrate
      SystemCertificationValidator as the final step before writing PipelineReport.

REQUIRED pattern in quality_agent.py:
  from vga.validation.system_certification_validator import SystemCertificationValidator
  from vga.core.exceptions import SystemCertificationFailureError

  # In QualityAgent.run():
  # ... [existing pipeline report construction] ...

  # MANDATORY final certification step (v17.2):
  certification_validator = SystemCertificationValidator(tracer=self.tracer)
  try:
      cert_result = certification_validator.certify(pipeline_report_dict)
      pipeline_report.certified = True
      pipeline_report.certification_version = "v17.2"
  except SystemCertificationFailureError as e:
      pipeline_report.certified = False
      pipeline_report.certification_failures = e.failures
      # CRITICAL: log and halt — do NOT proceed with uncertified output
      self.tracer.log({"event": "certification_failure", "failures": e.failures})
      raise  # propagate to MasterOrchestrator for HRG escalation

FORBIDDEN:
  - Writing PipelineReport.certified = True without calling SystemCertificationValidator
  - Suppressing SystemCertificationFailureError
  - Marking output as deployable before certification passes

PipelineReport MUST gain these fields (add to schemas.py):
  certified: bool = False
  certification_version: Optional[str] = None
  certification_failures: Optional[List[str]] = None

RATIONALE: Without formal certification, there is no authoritative signal that
a pipeline run meets all 7 v17.2 guarantees. Every deployment decision depends
on a certified PipelineReport.
```

---

## File Generation Checklist (v17.1)

Before submitting any v17.0 generated file, Claude Code MUST verify:

```
□ All imports reference previously-generated files (no forward dependencies)
□ schema_version = "v6.0" in all schema output methods
□ TEMPORAL_BUFFER_SIZE used (not hardcoded 5)
□ SVI_CFG_MIN/MAX used (not hardcoded 5.0/6.0)
□ No static LoRA weight in svi_wrapper.py
□ No single-frame init in svi_wrapper.generate()
□ context.evolve() called after every segment in temporal_engine.py
□ IdentityStateTracker.update() called after every CLIPValidator.score()
□ AudioQualityValidator.validate() called before storage write in audio_mixing_agent.py
□ CompositionPlanValidator.assert_in_context() called before generate() in image/video agents
□ HRGController VALID_CHECKPOINTS has exactly 11 entries
□ execute_stage() used for all stage calls in master_orchestrator.py
□ All new exception classes inherit from VGABaseError
□ All new Pydantic models have schema_version: str = "v6.0"
□ DEVIATION_LOG.md updated if any deviation from spec is necessary
□ Pre-commit hooks pass (ruff, mypy, architecture_linter)
□ temporal_engine.py uses explicit for loop (not comprehension, gather, or single-call)
□ No generate_all_segments() or merge_segments() call exists in temporal_engine.py
□ buffer updated inside the for loop body (not outside; not after the loop)
□ assert buffer.frames.shape[0] == TEMPORAL_BUFFER_SIZE present before AND inside loop
□ Doc 11 §12.18 integration tests referenced in commit message when temporal files change
□ SVIGenerationRecord includes source_buffer_frame_ids (exactly 5), previous_segment_id, generation_mode="autoregressive" (CGS-18)
□ TemporalAuthorityGuard.guard_svi_invoke() called in svi_wrapper.py (CGS-19)
□ TemporalAuthorityGuard.guard_buffer_update() called in temporal_buffer_manager.py update() (CGS-19)
□ TemporalAuthorityGuard.guard_segment_iteration() called in temporal_engine.py generate_scene() (CGS-19)
□ SystemCertificationValidator.certify() called in quality_agent.py before writing PipelineReport (CGS-20)
□ PipelineReport.certified field present and set by SystemCertificationValidator (CGS-20)
□ temporal_authority_guard.py, cross_modal_validation_unified.py, system_certification_validator.py generated (Doc 11 §12.19)
□ CrossModalValidationContract schema present in schemas.py with all required fields (Doc 05 §30)
□ Phase 12.X temporal loop integration tests pass (CERTIFICATION GATE — Doc 11 §12.X)
```

---

## v17.0 Rule Cross-Reference

| Code Rule | Spec Rule | CGS Rule | Test |
|---|---|---|---|
| RULE-86 | Temporal buffer 5-frame; no static LoRA; CFG [5.0, 6.0] | CGS-10 | chaos/test_temporal_buffer_error.py, chaos/test_svi_cfg_violation.py |
| RULE-87 | Autoregressive; multi-frame latents; sequential | CGS-10 | chaos/test_autoregressive_gate.py, integration/test_temporal_engine.py |
| RULE-88 | CompositionPlan required before image/video gen | CGS-11 | chaos/test_composition_plan_missing.py |
| RULE-89 | Identity validated at image + video + lip sync stages | CGS-12, CGS-16 | integration/test_identity_cross_phase.py |
| RULE-90 | Validation before progression at every stage | CGS-16 (execute_stage) | integration/test_hrg_flow.py |
| RULE-91 | Base generation purity (no LoRA in S-05) | CGRL-40 (existing) | chaos/test_lora_violation_halt.py |
| RULE-92 | Identity lock threshold ≥ 0.93 | CGRL-66 (existing) | unit/test_clip_validator.py |
| RULE-93 | Drift budget ≤ 0.02 per image stage | CGRL-66 (existing) | unit/test_image_refinement_agent.py |
| RULE-94 | Multi-pass identity construction (6A/6B/6C) | CGRL-66 (existing) | integration/test_full_image_pipeline.py |
| RULE-95 | Cross-stage identity propagation (frozen ref) | CGS-12 | chaos/test_identity_reference_immutability.py |
| RULE-96 | Timing contract ±0.10s | CGS-13 | integration/test_cross_modal_alignment.py |
| RULE-97 | Lip sync identity delta ≤ 0.03 | CGS-12 | integration/test_identity_cross_phase.py |
| RULE-98 | Audio mixing priority (D > A > M) | CGRL-73 (existing) | unit/test_audio_mixing_agent.py |
| RULE-99 | SNR ≥ 10 dB; peaks ≤ 0 dBFS | CGS-13 | unit/test_audio_quality_validator.py |
| CGRL-85 | execute_stage() mandatory | CGS-16 | chaos/test_execute_stage_contract.py |
| CGRL-86–100 | All v17.0 coding rules | CGS-09 through CGS-16 | v17.0 test suite |
| RULE-87     | Autoregressive loop; sequential | CGS-17 | tests/integration/test_temporal_loop_integration.py |
| RULE-86     | Buffer size = 5; loop enforcement | CGS-17 | tests/regression/test_temporal_regression_guard.py |
| SVI Lineage | SVI traceable to TemporalBuffer | CGS-18 | tests/integration/test_svi_lineage.py |
| TE Authority | TemporalEngine exclusive control | CGS-19 | tests/unit/test_temporal_authority_guard.py |
| Certification | All 7 certification conditions | CGS-20 | tests/integration/test_system_certification.py |
```
