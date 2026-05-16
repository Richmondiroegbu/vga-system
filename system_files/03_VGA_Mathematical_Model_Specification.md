# VGA Mathematical Model Specification
**Project:** Video Generation Automation (VGA)
**Version:** 17.2.0
**Status:** Authoritative Reference
**Audience:** Pipeline Engineers, Agent Implementors, Claude Code Agent

---

## Table of Contents

1. [Overview](#1-overview)
2–37. [All v15.0 Models Retained](#2-37-all-v150-mathematical-models-retained)
38. [CLIP Identity Validation Model (v16.0 — retained)](#38-clip-identity-validation-model)
39. [LoRA Conditional Application Model (v16.0 — retained)](#39-lora-conditional-application-model)
40. [Image Refinement Drift Model (v16.0 — retained)](#40-image-refinement-drift-model)
41. [Continuity Scoring Model (v16.0 — retained)](#41-continuity-scoring-model)
42. [Audio Timing Alignment Model (v16.0 — retained)](#42-audio-timing-alignment-model)
43. [Audio Mixing Priority Model (v16.0 — retained)](#43-audio-mixing-priority-model)
44. [Lip Sync Identity Guard Model (v16.0 — retained)](#44-lip-sync-identity-guard-model)
45. [Scene Composition Model (NEW v17.0)](#45-scene-composition-model)
46. [Temporal Buffer Model (NEW v17.0)](#46-temporal-buffer-model)
47. [Autoregressive Generation Model (NEW v17.0)](#47-autoregressive-generation-model)
48. [SVIScheduler Noise-Aware LoRA Model (NEW v17.0)](#48-svischeduler-noise-aware-lora-model)
49. [Motion State Estimation Model (NEW v17.0)](#49-motion-state-estimation-model)
50. [Cross-Phase Identity State Model (NEW v17.0)](#50-cross-phase-identity-state-model)
51. [Audio Quality Validation Model (NEW v17.0)](#51-audio-quality-validation-model)
52. [Cross-Modal Alignment Model (NEW v17.0)](#52-cross-modal-alignment-model)
53. [Constants & Tuning Parameters (v17.0)](#53-constants--tuning-parameters)

---

## 1. Overview

This document formalises every mathematical model, scoring function, and allocation algorithm used by VGA pipeline agents. **Every formula defined here has a direct implementation in agent code and a direct binding to a pipeline execution decision.**

**Retained from v16.0 (unchanged):** §2–§44 (all scoring models, adaptive engine, consistency validation, identity consistency, CFG control, temporal extension, motion progression, long-sequence identity stability, enforcement scoring, compound failure probability, retry strategy formulas, session health scoring, adaptive calibration loop, smart gating model, SLA enforcement model, immutable context safety bounds, CLIP identity validation, LoRA conditional application, image refinement drift, continuity scoring, audio timing, audio mixing priority, lip sync identity guard).

**New in v17.0:**
- §45 Scene Composition Model — mapping narrative intent to visual directives
- §46 Temporal Buffer Model — 5-frame buffer formal definition and update contract
- §47 Autoregressive Generation Model — mathematical conditioning contract
- §48 SVIScheduler Noise-Aware LoRA Model — timestep-conditional weight function
- §49 Motion State Estimation Model — optical flow derivation and propagation
- §50 Cross-Phase Identity State Model — cumulative drift tracking and threshold
- §51 Audio Quality Validation Model — SNR and peak level formal specification
- §52 Cross-Modal Alignment Model — video ↔ audio duration alignment
- §53 Constants & Tuning Parameters (v17.0 complete)

---

## 2–37: All v15.0 Mathematical Models Retained

Sections §2 through §37 are retained without modification from v15.0.

---

## 38. CLIP Identity Validation Model (retained from v16.0)

All §38.1 through §38.4 retained without modification.

**v17.0 addition to §38.2 — Application Points:**

```
Position 7: After each SVI segment in TemporalEngine (S-09) (NEW v17.0)
  For each generated segment keyframe:
    clip_score = clip_validator.score(segment.keyframe, char_identity_ref)
    assert clip_score >= CLIP_IDENTITY_THRESHOLD  [RULE-89]
  Note: char_identity_ref is frozen from S-07; never recomputed

Position 8: After each lip-synced segment in LipSyncAgent (S-12) (updated from v16.0)
  [unchanged from v16.0 §38.2 Position 6]
```

---

## 39–44: LoRA, Drift, Continuity, Audio Timing, Mixing, Lip Sync Models (retained from v16.0)

Sections §39 through §44 are retained without modification from v16.0.

---

## 45. Scene Composition Model (NEW v17.0)

### 45.1 Composition Function

```
SceneCompositionAgent maps narrative fields to visual directives:

  f_compose(dialogue, emotion, motion_intent, characters, environment)
  → CompositionPlan {
    camera_angle,
    camera_motion,
    character_positions,
    focus_subject,
    lighting_style,
    motion_vector
  }

  Where:
    dialogue        ∈ str   (scene dialogue text from ScriptSchema)
    emotion         ∈ str   (e.g., "tense", "hopeful", "sorrowful")
    motion_intent   ∈ str   (e.g., "walking slowly", "running", "standing still")
    characters      ∈ List  (character identity descriptors from IdentityDesignSchema)
    environment     ∈ str   (environment_description from IdentityDesignSchema)

  Output fields (all REQUIRED):
    camera_angle        ∈ {"close-up", "medium shot", "wide shot", "overhead", "low angle"}
    camera_motion       ∈ {"static", "slow dolly forward", "slow dolly back", "pan left",
                            "pan right", "tilt up", "tilt down", "tracking shot"}
    character_positions ∈ List[{character_id: str, position: str, facing: str}]
    focus_subject       ∈ str  (which element is the primary visual focus)
    lighting_style      ∈ {"soft natural", "low-key dramatic", "high-key bright",
                            "golden hour", "cool tone", "warm tone"}
    motion_vector       ∈ {"stationary", "forward_slow", "forward_fast",
                            "backward_slow", "left_slow", "right_slow",
                            "right_medium", "upward", "downward"}
```

### 45.2 Composition Validation

```
All 6 fields are mandatory. Validation gate:

  CompositionPlanValidator.validate(plan):
    assert plan.camera_angle is not None and len(plan.camera_angle) > 0
    assert plan.camera_motion is not None
    assert len(plan.character_positions) >= 1
    assert plan.focus_subject is not None
    assert plan.lighting_style is not None
    assert plan.motion_vector is not None

  On failure: regenerate SceneCompositionAgent (up to 3 retries)
  On 3rd failure: raise CompositionPlanValidationError → HRG-4 escalation
```

### 45.3 CompositionPlan → Prompt Injection

```
For FLUX wrapper (image generation):
  image_prompt = base_prompt
    + f", camera: {plan.camera_angle}"
    + f", lighting: {plan.lighting_style}"
    + f", focus: {plan.focus_subject}"
    + f", character at: {plan.character_positions[0].position}"

For Wan2.2/SVI wrapper (video generation):
  video_prompt = base_prompt
    + f", camera motion: {plan.camera_motion}"
    + f", motion direction: {plan.motion_vector}"

  motion_params = {
    "camera_motion": plan.camera_motion,
    "motion_vector": plan.motion_vector
  }
```

---

## 46. Temporal Buffer Model (v17.1 — HARDENED)

### 46.1 Buffer Formal Definition

```
TemporalBuffer B is a fixed-size ordered tensor collection:

  B = {frames: Tensor, timestamps: List[float], motion_vector: Optional[Tensor], scene_id: str}

  Where:
    frames.shape     = (N_BUFFER, C, H, W)  ← N_BUFFER = BUFFER_SIZE = 5 (STRICT)
    C                = colour channels (3 for RGB)
    H, W             = spatial dimensions (constant across all frames in buffer)
    timestamps[i]    = wall-clock time when frame i was extracted (seconds)
    motion_vector    = optical flow aggregate from last 2 frames (Optional)
    scene_id         = identifier of the scene this buffer belongs to

  Invariants:
    len(B.frames) == BUFFER_SIZE           ← always exactly 5
    all(f.shape == B.frames[0].shape for f in B.frames)  ← same resolution
    all(f.dtype == B.frames[0].dtype for f in B.frames)  ← same dtype
    all(f.min() >= 0.0 and f.max() <= 1.0 for f in B.frames)  ← normalized [0,1]
```

### 46.2 Buffer Initialization

```
After Segment_1 generation by Wan2.2:

  def init(segment_1: VideoSegment) -> TemporalBuffer:
    frames = extract_last_N_frames(segment_1.video_tensor, N=BUFFER_SIZE)
    # frames.shape = (5, C, H, W)

    if frames.shape[0] < BUFFER_SIZE:
      raise TemporalBufferError(
        scene_id=segment_1.scene_id,
        frame_count=frames.shape[0],
        required=BUFFER_SIZE
      )

    return TemporalBuffer(
      frames     = frames,
      timestamps = [time.time() - (BUFFER_SIZE - i) * dt for i in range(BUFFER_SIZE)],
      motion_vector = None,  # computed lazily by MotionStateTracker
      scene_id   = segment_1.scene_id
    )
```

### 46.3 Buffer Update Contract

```
After each SVI segment generation:

  def update(B: TemporalBuffer, new_segment: VideoSegment) -> TemporalBuffer:
    new_frames = extract_last_N_frames(new_segment.video_tensor, N=BUFFER_SIZE)

    # Rolling update: keep last 5 frames from new_segment
    # (not a concatenation of old + new — purely from new segment)
    updated_buffer = TemporalBuffer(
      frames        = new_frames,          # shape: (5, C, H, W)
      timestamps    = extract_timestamps(new_segment, N=BUFFER_SIZE),
      motion_vector = None,                # reset; recomputed next call
      scene_id      = B.scene_id
    )

    assert len(updated_buffer.frames) == BUFFER_SIZE  ← enforce invariant
    return updated_buffer

  Temporal constraint:
    B_n.frames[-1]  ← last frame of Segment[n]
    B_n.frames[-5]  ← 5th-from-last frame of Segment[n]
    All drawn from Segment[n] only — not concatenated from multiple segments
```

### 46.4 Buffer → SVI Latent Encoding

```
Before SVI conditioning:

  def encode(B: TemporalBuffer) -> Tensor:
    """
    Encodes 5 frames into latent space for SVI conditioning.
    Returns multi-frame latent tensor, NOT a single-image latent.
    """
    latents = []
    for frame in B.frames:  # 5 frames
      latent = vae_encoder.encode(frame)    # shape: (C', H', W') per frame
      latents.append(latent)

    stacked = torch.stack(latents, dim=0)  # shape: (5, C', H', W')

    # HARD CONSTRAINT: stacked.shape[0] MUST equal BUFFER_SIZE
    assert stacked.shape[0] == BUFFER_SIZE, "Multi-frame latent required; single-frame FORBIDDEN"

    return stacked   # shape: (5, C', H', W') → passed to SVI as init_latents
```

### 46.5 Buffer Integrity Gates

```
Gate A — Before TemporalEngine starts:
  assert buffer is not None, "Buffer must be initialized from Segment_1"

Gate B — Before every SVI call:
  assert buffer.frames.shape[0] == BUFFER_SIZE  # = 5
  assert buffer.scene_id == current_scene_id

Gate C — Frame dimension consistency:
  for i in range(1, len(buffer.frames)):
    assert buffer.frames[i].shape == buffer.frames[0].shape

  On any violation: raise TemporalBufferError → pipeline halts (CRITICAL)
```

---

### 46.4 Temporal Buffer Hard Constraint Equations (v17.1 — CRITICAL ADDITION)

```
TEMPORAL CONSTRAINT EQUATIONS (authoritative — must be enforced in code):

Constraint 1 — Buffer Size Invariant:
  |B_t| = 5     (fixed; non-negotiable; violation = SYSTEM FAILURE)

Constraint 2 — Generation Conditioning:
  Segment_{n+1} = Diffusion(B_t, prompt_n)
    where Diffusion = SVI Pro 2
          B_t       = TemporalBuffer at time t (5 frames)
          prompt_n  = composed temporal prompt for segment n+1

Constraint 3 — Buffer Update Rule:
  B_{t+1} = last_5_frames(Segment_{n+1})

Constraint 4 — Autoregressive Chain:
  Segment_1            → B_1 = last_5_frames(Segment_1)
  Segment_2 = f(B_1)   → B_2 = last_5_frames(Segment_2)
  Segment_3 = f(B_2)   → B_3 = last_5_frames(Segment_3)
  ...
  Segment_N = f(B_{N-1})

  This is a strict Markov chain: Segment_n+1 depends ONLY on B_n.
  It does NOT depend on any segment prior to n.

Constraint 5 — Forbidden Operations:
  generate_all_at_once(scene_plan)     FORBIDDEN — violates Constraint 2
  merge(Segment_a, Segment_b)          FORBIDDEN — violates Constraint 3
  SVI.generate(init_image=single_img)  FORBIDDEN — must use 5-frame tensor

Hard Enforcement:
  Code implementing these constraints MUST assert:
    assert buffer.frames.shape[0] == TEMPORAL_BUFFER_SIZE  # before generate
    assert latents.shape[0] == TEMPORAL_BUFFER_SIZE        # before SVI call
    assert buffer.frames.shape[0] == TEMPORAL_BUFFER_SIZE  # after update
```

---

## 47. Autoregressive Generation Model (v17.1 — HARDENED)

### 47.1 Autoregressive Conditioning Contract

```
VGA's temporal generation follows a strict autoregressive model:

  P(Segment[n] | Segment[n-1], Segment[n-2], ..., Segment[n-4])
  ≡ P(Segment[n] | TemporalBuffer[n-1])   ← Markov approximation over 5-frame buffer

  Where:
    Segment[1]   ~ P(Segment[1] | init_image, CompositionPlan)  ← Wan2.2
    Segment[n]   ~ P(Segment[n] | B[n-1], plan[n])              ← SVI for n ≥ 2
    B[n-1]       = TemporalBuffer updated from Segment[n-1]

  Forbidden:
    P(Segment[n] | Segment[1])         ← skipping segments is FORBIDDEN
    P(Segment[n] | single_image)       ← single-frame conditioning FORBIDDEN for n ≥ 2
    batch_merge(Segments[1..N])        ← batch merging is FORBIDDEN
```

### 47.2 Generation Loop Invariants

```
The autoregressive loop maintains these invariants at every iteration:

  Invariant I1: Buffer currency
    B before Segment[n] generation contains frames from Segment[n-1] ONLY
    (not from any earlier segment)

  Invariant I2: Buffer completeness
    B.frames.shape[0] == 5 at every iteration

  Invariant I3: Conditioning completeness
    SVI init_latents.shape[0] == 5 at every SVI call

  Invariant I4: Sequential ordering
    No Segment[n] may be generated before Segment[n-1] is complete
    (no parallel segment generation)

  Invariant I5: Context currency
    context = context.evolve(Segment[n]) after every segment completion
    context.temporal_state == TemporalBuffer after Segment[n] update

  Violation of any invariant: raise AutoregressiveViolationError → CRITICAL halt
```

### 47.3 Error Recycling Loop

```
For segment n with plan segment_plan[n]:

  for attempt in range(TEMPORAL_MAX_RETRIES_PER_SEGMENT):

    segment_n = TemporalEngine.generate_next(buffer=B[n-1], plan=segment_plan[n])

    # Identity check (RULE-89)
    score = CLIPValidator.score(segment_n.keyframe, char_identity_ref)

    # Per-segment continuity check
    cont = ContinuityValidator.score_segment(B[n-1].frames[-1], segment_n.frames[0])

    if score >= CLIP_IDENTITY_THRESHOLD and cont >= SEGMENT_CONTINUITY_MIN:
      # Update buffer; continue loop
      B[n] = TemporalBufferManager.update(B[n-1], segment_n)
      return segment_n

    # Adjust parameters and retry
    adjust_generation_params(attempt)

  # All retries exhausted
  raise TemporalSegmentFailureError(scene_id, segment_id=n, best_score=max_score)
  ← Escalate to HRG-8 with failure annotation

  Where:
    TEMPORAL_MAX_RETRIES_PER_SEGMENT = 3
    SEGMENT_CONTINUITY_MIN           = 0.85  (per-segment; lower than full-scene 0.90)
```

---

### 47.3 Required Implementation Model (v17.1)

```python
# This is the REQUIRED implementation model for TemporalEngine.
# Claude Code MUST produce code that matches this structure.

def temporal_loop(scene_plan: List[SegmentPlan], initial_frame: Tensor) -> List[VideoSegment]:
    """
    AUTOREGRESSIVE TEMPORAL LOOP — the ONLY valid implementation of S-09.
    Single-call or batch implementations are FORBIDDEN (CGS-17).
    """
    # Step 1: Generate Segment_1 via Wan2.2 (S-08, called before this function)
    segment_1 = generate_segment_1(initial_frame)

    # Step 2: Initialize buffer from Segment_1
    buffer = TemporalBufferManager.init(segment_1)
    assert buffer.frames.shape[0] == TEMPORAL_BUFFER_SIZE  # = 5

    segments = [segment_1]

    # Step 3: Autoregressive loop for all remaining segments
    for plan in scene_plan[1:]:
        # Validate buffer before each generation
        assert buffer.frames.shape[0] == TEMPORAL_BUFFER_SIZE

        # Estimate motion from buffer
        motion_state = MotionStateTracker.estimate(buffer.frames)

        # Generate next segment from buffer (NOT from a single image)
        segment_n = generate_next_from_buffer(buffer, plan, motion_state)

        # Validate the generated segment
        validate_segment(segment_n, char_identity_ref)   # CLIPValidator + ContinuityValidator

        # Update buffer rolling window
        buffer = TemporalBufferManager.update(buffer, segment_n)
        assert buffer.frames.shape[0] == TEMPORAL_BUFFER_SIZE  # invariant

        segments.append(segment_n)

    return segments
```

### 47.4 Temporal Constraint System (NEW v17.2 — Formal Mathematical Definition)

```
TEMPORAL CONSTRAINT SYSTEM v17.2 (authoritative):

Constraint 1 — Buffer Size Invariant:
  |B_t| = 5     (fixed; non-negotiable; violation = SYSTEM FAILURE)

Constraint 2 — Autoregressive Generation:
  Segment_{n+1} = Diffusion(B_t, prompt_n)
    where:
      Diffusion = SVI Pro 2
      B_t       = TemporalBuffer at step t (exactly 5 frames)
      prompt_n  = composed temporal prompt for segment n+1

Constraint 3 — Buffer Update Rule:
  B_{t+1} = last_5_frames(Segment_{n+1})

Constraint 4 — Strict Markov Chain:
  ∀ n ≥ 1:
    Segment_{n+1} depends ONLY on B_n
    It does NOT depend on any segment prior to n
    (strict Markov property; memory window = 5 frames)

  Full chain:
    Segment_1            → B_1 = last_5_frames(Segment_1)
    Segment_2 = f(B_1)   → B_2 = last_5_frames(Segment_2)
    Segment_3 = f(B_2)   → B_3 = last_5_frames(Segment_3)
    ...
    Segment_N = f(B_{N-1})

Constraint 5 — Forbidden Operations (mathematical definition):
  ∄ batch_fn : scene_plan → Segments          (batch generation is mathematically undefined)
  ∄ merge_fn : Segment_a × Segment_b → Segment (post-hoc merge violates Constraint 3)
  single_frame_conditioning(init_img) for n ≥ 2 ← violates Constraint 2 (B_t not single img)

All constraints are enforced by runtime assertions; any violation raises:
  TemporalBufferError   (Constraint 1 violation)
  AutoregressiveViolationError (Constraint 2, 5 violation)
  CriticalPipelineError (Constraint 4 violation — wrong segment ordering)
```

---

## 48. SVIScheduler Noise-Aware LoRA Model (NEW v17.0)

### 48.1 Timestep-Conditional Weight Function

```
SVI generation uses T denoising timesteps: t ∈ {T, T-1, ..., 1, 0}
  (t=T: pure noise; t=0: clean image)

  LoRA weight function w(t):

    w(t) = 0.6   if t > THRESHOLD_HIGH_NOISE    ← structure + motion reinforcement
    w(t) = 0.5   if THRESHOLD_MID_NOISE < t ≤ THRESHOLD_HIGH_NOISE
    w(t) = 0.4   if t ≤ THRESHOLD_MID_NOISE     ← detail preservation

  Where:
    THRESHOLD_HIGH_NOISE = floor(T × HIGH_NOISE_FRACTION)   = floor(T × 0.67)
    THRESHOLD_MID_NOISE  = floor(T × MID_NOISE_FRACTION)    = floor(T × 0.33)

  For T = 30 steps (standard):
    THRESHOLD_HIGH_NOISE = 20  → timesteps 30..21: w=0.6
    THRESHOLD_MID_NOISE  = 10  → timesteps 20..11: w=0.5
                               → timesteps 10..0:  w=0.4

  For T = 50 steps (critical):
    THRESHOLD_HIGH_NOISE = 34  → timesteps 50..35: w=0.6
    THRESHOLD_MID_NOISE  = 17  → timesteps 34..18: w=0.5
                               → timesteps 17..0:  w=0.4

  Static weight: FORBIDDEN. Any call with a single constant w is a RULE-86 violation.
```

### 48.2 CFG Constraint

```
SVI CFG (classifier-free guidance scale) contract:

  cfg ∈ [SVI_CFG_MIN, SVI_CFG_MAX] = [5.0, 6.0]

  Rationale:
    cfg < 5.0: insufficient guidance → identity/structure drift
    cfg > 6.0: over-guidance → color banding artifacts (empirically confirmed)
    cfg = 5.5: recommended default for balanced quality

  Clamping enforcement:
    cfg_clamped = max(SVI_CFG_MIN, min(SVI_CFG_MAX, requested_cfg))
    if cfg_clamped != requested_cfg:
      raise SVICFGViolationError(requested=requested_cfg, clamped=cfg_clamped)
      ← CRITICAL: do not silently clamp; raise and halt
```

### 48.3 Steps Strategy

```
Dynamic steps based on segment criticality:

  is_critical = segment_plan.is_scene_boundary OR segment_plan.has_dialogue_cutoff

  steps = STEPS_CRITICAL if is_critical else STEPS_STANDARD

  Where:
    STEPS_CRITICAL = 50   (scene boundaries; dialogue onset frames)
    STEPS_STANDARD = 30   (mid-sequence segments with smooth motion)

  Note: Steps in range [4, 8] permitted for rapid preview mode only;
        production pipeline always uses STEPS_STANDARD or STEPS_CRITICAL.
```

### 48.4 LoRA Weight × Output Blend

```
SVI output at each timestep t:

  output(t) = (1 - w(t)) × base_output(t) + w(t) × lora_conditioned_output(t)

  Interpretation:
    w=0.6 (high noise): 60% LoRA influence → strong motion/structure imprinting
    w=0.5 (mid noise):  50% LoRA influence → balanced identity/structure
    w=0.4 (low noise):  40% LoRA influence → preserves fine detail from base model
```

---

## 49. Motion State Estimation Model (NEW v17.0)

### 49.1 Optical Flow Computation

```
MotionStateTracker estimates motion from TemporalBuffer frames:

  Input:  B.frames = [f_0, f_1, f_2, f_3, f_4]  (5 consecutive frames)

  Pairwise optical flows:
    flow_i = optical_flow(f_i, f_{i+1})  for i ∈ {0, 1, 2, 3}
    flow_i.shape = (H, W, 2)  ← (H, W) pixel displacements in (x, y)

  Aggregate flow:
    mean_flow = mean(flow_0, flow_1, flow_2, flow_3)  ← per-pixel mean
    velocity_vector = spatial_mean(mean_flow)           ← scalar (v_x, v_y)

  Magnitude:
    magnitude = norm(velocity_vector) = sqrt(v_x² + v_y²)

  Direction classification:
    direction = classify_direction(v_x, v_y):
      if magnitude < MOTION_STATIONARY_THRESHOLD:      → "stationary"
      elif |v_x| > |v_y| and v_x > 0:                 → "right"
      elif |v_x| > |v_y| and v_x < 0:                 → "left"
      elif |v_y| > |v_x| and v_y > 0:                 → "forward"  (camera forward)
      elif |v_y| > |v_x| and v_y < 0:                 → "backward"
      else:                                             → "diagonal"

  Where:
    MOTION_STATIONARY_THRESHOLD = 0.02  (normalised pixel displacement)
```

### 49.2 Motion State Propagation

```
MotionState is propagated into the next segment's generation plan:

  motion_state = MotionStateTracker.estimate(buffer.frames)

  plan.motion_vector = map_to_motion_vector(motion_state.direction, motion_state.magnitude)
  # e.g., direction="forward", magnitude=0.12 → motion_vector="forward_slow"

  context = context.evolve({motion_state: motion_state})
  # motion_state persists in context for downstream stages

  Purpose:
    Prevents "motion reset" between segments (discontinuous direction change)
    Informs SVIScheduler of expected motion pattern
    Logged to motion_state_log.json for observability
```

### 49.3 Motion Consistency Constraint

```
Between consecutive segments, motion direction must not change abruptly:

  direction_change_penalty(d_prev, d_curr):
    if d_prev == d_curr:                        return 0.0   (no change)
    if are_adjacent(d_prev, d_curr):            return 0.2   (slight change)
    if are_opposite(d_prev, d_curr):            return 1.0   (full reversal — penalised)

  where are_opposite = {("forward","backward"), ("left","right"), ...}

  If direction_change_penalty > 0.5:
    tracer.log({"event": "motion_direction_inconsistency", ...})
    Adjust generation plan: increase steps, reduce motion magnitude
    (Soft constraint: does not halt pipeline, but triggers advisory retry)
```

---

## 50. Cross-Phase Identity State Model (NEW v17.0)

### 50.1 IdentityState Definition

```
IdentityState IS is a mutable accumulator updated at every validation checkpoint:

  IS = {
    embedding_vector: Tensor,   ← frozen CLIP embedding of best base image; NEVER changes
    drift_score:      float,    ← cumulative drift from baseline (sum of per-stage deltas)
    history:          List[float]  ← per-stage drift values in order
  }

  embedding_vector is computed ONCE at S-07 completion:
    char_identity_ref = CLIP_encoder(best_refined_image)
    # "best" = highest CLIP score among Stage S-07 outputs

  drift_score is monotonically increasing (drift can only accumulate):
    IS.drift_score[n] = IS.drift_score[n-1] + delta[n]
    IS.history.append(delta[n])
```

### 50.2 Per-Stage Drift Measurement

```
At each cross-phase identity validation:

  Let e_ref = IS.embedding_vector           (frozen; never recomputed)
  Let e_new = CLIP_encoder(new_frame)       (frame from current stage)

  delta = 1.0 - cosine_similarity(e_ref, e_new)
  ← Normalised: delta=0.0 means perfect match; delta=1.0 means completely different

  IS = IdentityStateTracker.update(IS, e_new):
    IS.drift_score += delta
    IS.history.append(delta)

    if IS.drift_score > IDENTITY_CUMULATIVE_DRIFT_THRESHOLD:
      raise IdentityCumulativeDriftError(
        scene_id=scene_id,
        cumulative_drift=IS.drift_score,
        threshold=IDENTITY_CUMULATIVE_DRIFT_THRESHOLD
      )
    return IS

  Where:
    IDENTITY_CUMULATIVE_DRIFT_THRESHOLD = 0.15
    (sum of deltas across all validation points; exceeding triggers full phase regen)
```

### 50.3 Identity Validation Schedule

```
Identity is validated at these pipeline positions (comprehensive from v17.0):

  Position 1: After S-05 (BaseImageAgent) — 6 base images
    Per-image: clip_score >= 0.93  (absolute; not delta-based)

  Position 2: After S-06A (MultiAngleAgent) — angle variants
    Per-image: clip_score >= 0.93

  Position 3: After S-06B (ImageMergeAgent) — composed images
    Per-image: clip_score >= 0.93

  Position 4: After S-06C (SceneExpansionAgent) — expanded images
    Per-image: clip_score >= 0.93

  Position 5: After S-07 (ImageRefinementAgent) — refined images
    Per-image: clip_score >= 0.93 AND drift <= 0.02
    [char_identity_ref FROZEN here]

  Position 6: After each SVI segment in S-09 (TemporalEngine) — NEW v17.0
    Per-segment keyframe: clip_score >= 0.93
    delta = abs(clip_score - clip_score_stage7[segment_id])
    IdentityStateTracker.update(IS, e_new)

  Position 7: After S-12 (LipSyncAgent) — synced frames — NEW v17.0 (enhanced)
    Per-synced-segment: delta <= 0.03
    IdentityStateTracker.update(IS, e_synced)
```

### 50.4 Cumulative Drift Failure and Recovery

```
On IdentityCumulativeDriftError:

  1. Log IS.drift_score, IS.history to identity_state_log.json
  2. Identify failing phase (image vs video vs lip sync)
  3. Trigger targeted phase regeneration:
     - If drift accumulates in S-09 (video): regenerate from S-08 with stronger identity prompting
     - If drift accumulates in S-12 (lip sync): reduce sync_strength; retry all lip sync segments
  4. Reset IS.drift_score = 0.0 after successful phase regeneration
  5. Maximum 1 full phase regeneration per scene before SceneHaltError

  Statistical expectation:
    Under normal operation (stable identity, correct models):
      per-stage delta ≈ 0.01–0.03
      cumulative after all positions ≈ 0.05–0.10
      IDENTITY_CUMULATIVE_DRIFT_THRESHOLD = 0.15 provides 50% safety margin
```

---

## 51. Audio Quality Validation Model (NEW v17.0)

### 51.1 Signal-to-Noise Ratio (SNR)

```
After AudioMixingAgent completes mixing:

  SNR measures intelligibility of dialogue above background audio:

  snr_db = 10 × log10(P_dialogue / P_background)

  Where:
    P_dialogue   = mean power of dialogue track during speech-active segments
    P_background = mean power of (ambient + music) tracks during same segments

  Simplified dB computation:
    dialogue_rms  = rms_level_db(dialogue_track)
    background_rms = rms_level_db(ambient_track + music_track)
    snr_db = dialogue_rms - background_rms

  Threshold:
    assert snr_db >= MIN_SNR_DB   ← MIN_SNR_DB = 10.0 dB (RULE-99)

  Interpretation:
    snr = 10 dB:  dialogue clearly intelligible above background (minimum acceptable)
    snr = 15 dB:  excellent intelligibility
    snr = 20 dB:  broadcast quality
    snr < 10 dB:  dialogue buried; violates RULE-99; triggers re-mix
```

### 51.2 Peak Level (Clipping) Gate

```
Peak level check prevents clipping distortion:

  peak_db = max(peak_level_db(mixed_audio_samples))

  Where:
    peak_level_db(samples) = 20 × log10(max(|samples|))
    max taken over all time samples and all channels

  Constraint:
    assert peak_db <= 0.0 dBFS   ← 0 dBFS = digital full scale; RULE-99

  Clipping condition:
    peak_db > 0.0: clipping detected → normalization required

  Normalization:
    if peak_db > 0.0:
      gain_reduction = -peak_db - HEADROOM_DB  ← HEADROOM_DB = 1.0 dB
      mixed_audio = apply_gain(mixed_audio, gain_reduction)
      peak_db_after = max_peak(mixed_audio)
      assert peak_db_after <= -HEADROOM_DB  ← at least -1 dBFS after normalization

  Maximum re-mix attempts for quality failures: 3
  After 3 failures: accept best result; flag in HRG-11 with warning badge
```

### 51.3 Audio Quality Record

```
After every AudioMixingAgent completion:

  record = AudioQualityRecord(
    scene_id          = scene_id,
    snr_db            = snr_db,
    peak_db           = peak_db,
    clipping_detected = peak_db > 0.0,
    snr_passed        = snr_db >= MIN_SNR_DB,
    clipping_passed   = peak_db <= 0.0,
    schema_version    = "v6.0"
  )

  storage.append_audio_quality_record(job_id, record)
  tracer.log({"event": "audio_quality_validation", **record.model_dump()})
```

---

## 52. Cross-Modal Alignment Model (NEW v17.0)

### 52.1 Duration Alignment Contract

```
For each scene segment i:

  T_video[i] = duration(video_segment[i])    ← from S-09 TemporalEngine
  T_audio[i] = duration(dialogue_audio[i])   ← from S-11 DialogueAgent

  alignment_error[i] = |T_audio[i] - T_video[i]|

  Constraint:
    assert alignment_error[i] <= TIMING_TOLERANCE_S   ← 0.10 seconds (RULE-96)

  This is the same timing contract as the audio-segment gate in v16.0 (§42),
  now explicitly framed as cross-modal alignment for video ↔ audio consistency.
```

### 52.2 Segment Boundary Alignment

```
Segment boundaries must be identical between video and audio tracks:

  For scene with N segments:
    video_boundaries = [0, T_v1, T_v1+T_v2, ..., sum(T_vi for i in 1..N)]
    audio_boundaries = [0, T_a1, T_a1+T_a2, ..., sum(T_ai for i in 1..N)]

    For each boundary b_j:
      boundary_error[j] = |video_boundaries[j] - audio_boundaries[j]|
      assert boundary_error[j] <= TIMING_TOLERANCE_S

  Cumulative alignment check:
    total_video_duration = sum(T_video[i] for all i)
    total_audio_duration = sum(T_audio[i] for all i)
    total_error = |total_video_duration - total_audio_duration|
    assert total_error <= N_segments * TIMING_TOLERANCE_S
```

### 52.3 Identity Reference Consistency

```
Cross-modal identity consistency contract:

  The same char_identity_ref embedding MUST be used at:
    - All image validation points (S-05, S-06A/B/C, S-07)
    - All video segment validation points (S-09)
    - Lip sync validation (S-12)

  Verification:
    reference_hash = hash(char_identity_ref.numpy().tobytes())

  At every CLIPValidator call:
    current_ref_hash = hash(identity_ref.numpy().tobytes())
    assert current_ref_hash == reference_hash, "Identity reference has been mutated — CRITICAL"

  On mismatch: raise IdentityReferenceCorruptionError → CRITICAL halt
```

---

## 53. Constants & Tuning Parameters (v17.0)

All v16.0 constants retained (§45 from v16.0). v17.0 additions:

```python
# ── Scene Composition ─────────────────────────────────────────────────────
SLA_COMPOSITION_AGENT_MAX_S: float = 15.0   # SceneCompositionAgent SLA
COMPOSITION_MAX_RETRIES: int = 3            # max retries on schema validation failure

# ── Temporal Buffer ───────────────────────────────────────────────────────
TEMPORAL_BUFFER_SIZE: int = 5               # STRICT: always exactly 5 frames (RULE-86)
TEMPORAL_MAX_RETRIES_PER_SEGMENT: int = 3   # max retries per SVI segment
SEGMENT_CONTINUITY_MIN: float = 0.85       # per-segment continuity threshold (lower than full-scene)

# ── SVIScheduler ─────────────────────────────────────────────────────────
SVI_CFG_MIN: float = 5.0                   # minimum SVI CFG (RULE-86)
SVI_CFG_MAX: float = 6.0                   # maximum SVI CFG (RULE-86)
SVI_CFG_DEFAULT: float = 5.5              # recommended default SVI CFG
STEPS_CRITICAL: int = 50                   # SVI steps for critical segments
STEPS_STANDARD: int = 30                   # SVI steps for standard segments
STEPS_PREVIEW: int = 8                     # SVI steps for preview mode only
LORA_WEIGHT_HIGH_NOISE: float = 0.6       # SVI LoRA weight at high-noise phase
LORA_WEIGHT_MID_NOISE: float = 0.5        # SVI LoRA weight at mid-noise phase
LORA_WEIGHT_LOW_NOISE: float = 0.4        # SVI LoRA weight at low-noise phase
HIGH_NOISE_FRACTION: float = 0.67         # timestep fraction for high-noise boundary
MID_NOISE_FRACTION: float = 0.33          # timestep fraction for mid-noise boundary

# ── Motion State ─────────────────────────────────────────────────────────
MOTION_STATIONARY_THRESHOLD: float = 0.02  # normalised pixel displacement for "stationary"
MOTION_DIRECTION_PENALTY_THRESHOLD: float = 0.5  # above this → advisory retry

# ── Identity State (cross-phase) ─────────────────────────────────────────
IDENTITY_CUMULATIVE_DRIFT_THRESHOLD: float = 0.15  # cumulative drift across all phases
IDENTITY_MAX_PHASE_REGENERATIONS: int = 1          # max full phase regen per scene
IDENTITY_REFERENCE_HEADROOM_DB: float = 1.0        # audio normalization headroom
HEADROOM_DB: float = 1.0                           # audio peak headroom after normalization

# ── Audio Quality (NEW v17.0) ─────────────────────────────────────────────
MIN_SNR_DB: float = 10.0                   # minimum SNR for dialogue intelligibility (RULE-99)
MAX_PEAK_DBFS: float = 0.0                # maximum peak level; above = clipping (RULE-99)
AUDIO_QUALITY_MAX_RETRIES: int = 3        # max re-mix attempts on quality failure

# ── SLA Additions (v17.0) ────────────────────────────────────────────────
SLA_COMPOSITION_MAX_S: float = 15.0
SLA_TEMPORAL_BUFFER_UPDATE_MAX_S: float = 0.5
SLA_MOTION_STATE_ESTIMATION_MAX_S: float = 1.0
SLA_SVI_SEGMENT_STANDARD_MAX_S: float = 120.0
SLA_SVI_SEGMENT_CRITICAL_MAX_S: float = 300.0
SLA_IDENTITY_STATE_UPDATE_MAX_S: float = 0.2
SLA_AUDIO_QUALITY_VALIDATION_MAX_S: float = 5.0

# ── HRG (updated for 11 checkpoints) ────────────────────────────────────
HRG_CHECKPOINT_COUNT: int = 11            # v17.0 expands from 9 to 11
HRG_TIMEOUT_S: int = 300                  # unchanged from v16.0
HRG_UI_RENDER_BUDGET_S: float = 5.0      # unchanged from v16.0

# ── Schema Version ────────────────────────────────────────────────────────
SCHEMA_VERSION: str = "v6.0"              # ALL artifacts written by v17.0 agents
```
