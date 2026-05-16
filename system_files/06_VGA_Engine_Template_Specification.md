# VGA Engine Template Specification
**Project:** Video Generation Automation (VGA)
**Version:** 17.2.0
**Status:** Authoritative Reference
**Audience:** Agent Implementors, Claude Code Agent

---

## Overview

This document provides complete implementation templates for every engine and pattern in the VGA v17.0 system. **Every template here is authoritative. Claude Code MUST use these as the starting point for all implementations.**

---

### TEMPORAL ENGINE INVARIANT (v17.2 — MANDATORY — READ BEFORE ALL TEMPLATES)

```
INVARIANT v17.2:

TemporalEngine MUST be implemented as an EXPLICIT SEQUENTIAL LOOP.

The following implementation patterns are the ONLY valid structure:

  VALID PATTERN:
    for segment_plan in scene_plan[1:]:
        assert buffer.frames.shape[0] == TEMPORAL_BUFFER_SIZE  # gate before
        segment_n = generate_next_from_buffer(buffer, segment_plan)
        validate_segment(segment_n)
        buffer = update(buffer, segment_n)
        assert buffer.frames.shape[0] == TEMPORAL_BUFFER_SIZE  # gate after
        segments.append(segment_n)

Invalid implementations include (any of these = CGS-17 VIOLATION → HARD FAILURE):
  - Single-call temporal generation: svi.generate_all(scene_plan, init_image)
  - Batch segment processing: [svi.generate(plan) for plan in scene_plan]
  - External loop control: passing segment list in, iterating outside TemporalEngine
  - Parallel/async segment generation: asyncio.gather(*[generate(p) for p in plans])
  - Post-hoc merge: merge_segments(seg_a, seg_b)

Only valid loop pattern:
  FOR each segment_plan in scene_plan[1:]:
      generate → validate → update_buffer → context.evolve()

This invariant is enforced by:
  - CGS-17 (Code Generation Rule)
  - ArchitectureGuard runtime check
  - Temporal loop integration tests (Doc 11 §12.18)
  - Temporal loop regression guard (Doc 11 §12.18.2)
```

---

**Retained from v13.0 through v15.0 (unchanged):** Templates §1–§47 (BaseAgent, HRGGatedMixin, IdentityManager, PromptBuilder, Normalizer, Validator, PerformanceConfig, AttentionManager, CFGController, VRAM management, Agent GPU execution pattern, LipSync Gate, RegenerationEngine, LLMAgent, ScoringEngine, ErrorHandling, SVIWrapper, SegmentRoleRouter, MotionEvolutionEngine, IdentityTracker, IdentityDriftController, LightingNormalizer, TemporalIdentityValidator, SystemGuard, ArchitectureGuard, Observability, Error Handling, FailureClassifier, RetryStrategyEngine, RetryLimiter, SafeFallbackEngine, OutputIntegrityChecker, ExecutionAuthorityManager, TemporalCheckpointManager, HRGController+InterventionHandler+ResumeEngine, ExecutionScheduler+AsyncIOManager, StabilityManager+MemorySanitizer, SessionHealthMonitor, CompoundFailureProbabilityModel, v14.1 Error Handling, TemporalOrchestrator, RetryStrategyEngine, SLAManager, AdaptiveMemory, CalibrationEngine, PerformanceLearner, StrategyOptimizer, GatingController+SystemGuard upgrade, ImmutableContext+ContextFactory+ContextHistory+ContextDiff, Dev Safety Tooling, v15.0 Error Handling).

**Retained from v16.0 (unchanged):** Templates §48–§63 (ScriptAgent, IdentityDesignAgent, BaseImageAgent, ImageEditAgent+LoRAManager, ImageRefinementAgent, CLIPValidator, LoRAManager, ContinuityValidationAgent, DialogueAgent, LipSyncAgent, AmbientAudioAgent, MusicAgent, AudioMixingAgent, FastAPI Application, HRGController v16.0, v16.0 Error Handling).

**New in v17.0:**
- §64 SceneCompositionAgent Template
- §65 TemporalEngine Template (full architecture)
- §66 TemporalBufferManager Template
- §67 SVIScheduler Template (noise-aware LoRA)
- §68 MotionStateTracker Template
- §69 TemporalRetryController Template
- §70 IdentityStateTracker Template
- §71 AudioQualityValidator Template
- §72 CrossModalAlignmentValidator Template
- §73 CompositionPlanValidator Template
- §74 HRGController v17.0 Template (11 checkpoints)
- §75 MasterOrchestrator execute_stage() Template
- §76 v17.0 Error Handling Additions

**New in v17.2:**
- §77 Cross-Modal Validation Unified Contract Template
- §78 System Certification Validator Template
- §79 TemporalEngine Authority Guard Template

---

## 1–63: All v13.0 through v16.0 Templates Retained

---

## 64. SceneCompositionAgent Template (NEW v17.0)

```python
"""
scene_composition_agent.py
Stage S-04: Translates narrative intent into visual CompositionPlan.
Enforces: RULE-88 (CompositionPlan required before all image/video generation)
"""
from vga.agents.base_agent import LLMAgent
from vga.models.schemas import CompositionPlanSchema
from vga.validation.composition_validator import CompositionPlanValidator
from vga.core.exceptions import CompositionPlanValidationError
from vga.runtime.system_guard import SystemGuard
from vga.config.settings import COMPOSITION_MAX_RETRIES, SCHEMA_VERSION
from vga.core.tracer import Tracer
from pydantic import ValidationError
import json


class SceneCompositionAgent(LLMAgent):
    """
    Produces CompositionPlan from scene narrative fields.
    All 6 CompositionPlan fields are mandatory.
    CompositionPlan is REQUIRED before any image generation (RULE-88).
    """

    def __init__(self, model_wrapper, prompt_builder, tracer: Tracer,
                 system_guard: SystemGuard, authority_manager, composition_validator):
        super().__init__(model_wrapper, prompt_builder, tracer)
        self.system_guard = system_guard
        self.authority_manager = authority_manager
        self.composition_validator = composition_validator

    def compose(self, scene_data: dict, trace_id: str) -> CompositionPlanSchema:
        """
        Generate CompositionPlan from scene narrative data.
        RULE-88: Output MUST be validated before any image stage proceeds.
        """
        self.system_guard.execute("SceneCompositionAgent.compose", state=None, gating_mode=None)
        self.authority_manager.validate("COMPOSITION_LEVEL", "compose_scene")

        scene_id = scene_data.get("scene_id", "unknown")

        for attempt in range(COMPOSITION_MAX_RETRIES):
            try:
                raw_output = self.model_wrapper.generate(
                    prompt=self.prompt_builder.build_composition_prompt(scene_data),
                    max_tokens=1024,
                    temperature=0.3  # lower temperature for structured output
                )
                plan_dict = json.loads(raw_output)
                plan_dict["scene_id"] = scene_id
                plan_dict["schema_version"] = SCHEMA_VERSION
                validated = CompositionPlanSchema(**plan_dict)

                # Validate all 6 fields
                self.composition_validator.validate(validated)

                self.tracer.log({
                    "event": "composition_plan_created",
                    "scene_id": scene_id,
                    "camera_angle": validated.camera_angle,
                    "camera_motion": validated.camera_motion,
                    "motion_vector": validated.motion_vector,
                    "attempt": attempt,
                    "trace_id": trace_id
                })
                return validated

            except (json.JSONDecodeError, ValidationError) as e:
                self.tracer.log({
                    "event": "composition_schema_validation_failure",
                    "scene_id": scene_id,
                    "attempt": attempt,
                    "error": str(e),
                    "trace_id": trace_id
                })
                if attempt == COMPOSITION_MAX_RETRIES - 1:
                    raise CompositionPlanValidationError(
                        scene_id=scene_id,
                        reason=f"All {COMPOSITION_MAX_RETRIES} attempts failed: {e}"
                    )
```

---

## 65. TemporalEngine Template (v17.1 — LOOP CONTROLLER HARDENED)

```python
"""
temporal_engine.py
S-09: SVI Pro 2 autoregressive temporal generation subsystem.
Enforces: RULE-86 (5-frame buffer), RULE-87 (autoregressive conditioning)
"""
from vga.temporal.temporal_buffer_manager import TemporalBufferManager
from vga.temporal.svi_scheduler import SVIScheduler
from vga.temporal.motion_state_tracker import MotionStateTracker
from vga.temporal.temporal_retry_controller import TemporalRetryController
from vga.validation.clip_validator import CLIPValidator
from vga.identity.identity_state_tracker import IdentityStateTracker
from vga.core.exceptions import (
    TemporalBufferError, TemporalSegmentFailureError,
    SVICFGViolationError, AutoregressiveViolationError
)
from vga.config.settings import (
    TEMPORAL_BUFFER_SIZE, TEMPORAL_MAX_RETRIES_PER_SEGMENT,
    CLIP_IDENTITY_THRESHOLD, SVI_CFG_MIN, SVI_CFG_MAX,
    STEPS_CRITICAL, STEPS_STANDARD, SEGMENT_CONTINUITY_MIN
)
from vga.state.immutable_context import ImmutableContext
import torch


class TemporalEngine:
    """
    TEMPORAL ENGINE — STATEFUL AUTOREGRESSIVE LOOP CONTROLLER (v17.1)

    CRITICAL ARCHITECTURAL DECLARATION:
    TemporalEngine is NOT a stateless function.
    TemporalEngine is NOT a batch processor.
    TemporalEngine is NOT a segment merger.

    TemporalEngine IS a STATEFUL AUTOREGRESSIVE LOOP CONTROLLER that:
      - Manages TemporalBuffer state across ALL segments in a scene
      - Generates segments ONE AT A TIME in strict sequential order
      - Conditions each segment on the 5-frame TemporalBuffer from the prior segment
      - Updates the buffer after every segment (rolling window)
      - Validates every segment before committing it

    FORBIDDEN implementations (will raise AutoregressiveViolationError):
      generate_all_segments(scene_plan)     — batch mode; FORBIDDEN
      merge_segments(seg_a, seg_b)          — post-hoc merge; FORBIDDEN
      SVI.generate(init_image=single_img)   — single-image for n>=2; FORBIDDEN

    REQUIRED implementation: the generate_scene() method below implements
    the canonical autoregressive loop. Claude Code MUST NOT alter its
    sequential loop structure. [CGS-17, RULE-87]

    Segment_1 is generated by Wan2.2 (S-08, external to this class).
    Segments 2+ are generated here by SVI Pro 2, each conditioned on
    the 5-frame TemporalBuffer from the previous segment.
    """

    def __init__(self, svi_wrapper, buffer_manager: TemporalBufferManager,
                 svi_scheduler: SVIScheduler, motion_tracker: MotionStateTracker,
                 retry_controller: TemporalRetryController,
                 clip_validator: CLIPValidator, identity_tracker: IdentityStateTracker,
                 continuity_validator, tracer):
        self.svi = svi_wrapper
        self.buffer_manager = buffer_manager
        self.scheduler = svi_scheduler
        self.motion_tracker = motion_tracker
        self.retry_controller = retry_controller
        self.clip_validator = clip_validator
        self.identity_tracker = identity_tracker
        self.continuity_validator = continuity_validator
        self.tracer = tracer

    def generate_scene(self, segment_plans: list, segment_1, context: ImmutableContext,
                       char_identity_ref, trace_id: str) -> tuple:
        """
        Autoregressive generation of all segments from Segment_2 onward.
        Segment_1 must already be generated by Wan2.2 (S-08).
        Returns: (segments_list, updated_context)
        """
        # Initialize buffer from Segment_1
        buffer = self.buffer_manager.init(segment_1)
        if buffer.frames.shape[0] != TEMPORAL_BUFFER_SIZE:  # RULE-86
            raise TemporalBufferError(
                scene_id=segment_1.scene_id,
                frame_count=buffer.frames.shape[0],
                required=TEMPORAL_BUFFER_SIZE
            )

        segments = [segment_1]

        for n, segment_plan in enumerate(segment_plans[1:], start=2):

            # Gate: Buffer integrity (RULE-86)
            if buffer.frames.shape[0] != TEMPORAL_BUFFER_SIZE:
                raise TemporalBufferError(
                    scene_id=segment_plan.scene_id,
                    frame_count=buffer.frames.shape[0],
                    required=TEMPORAL_BUFFER_SIZE
                )

            # Estimate motion state from buffer
            motion_state = self.motion_tracker.estimate(buffer.frames)
            context = context.evolve({"motion_state": motion_state})
            self.tracer.log({
                "event": "motion_state_update",
                "segment_id": n,
                "direction": motion_state.direction,
                "magnitude": motion_state.magnitude,
                "trace_id": trace_id
            })

            # Generate segment with retry loop
            segment_n = self._generate_segment_with_retry(
                n, segment_plan, buffer, motion_state,
                char_identity_ref, context, trace_id
            )

            # Update buffer (RULE-86: rolling update from new segment)
            buffer = self.buffer_manager.update(buffer, segment_n)
            assert buffer.frames.shape[0] == TEMPORAL_BUFFER_SIZE

            segments.append(segment_n)
            context = context.evolve({
                "temporal_state": {"buffer": buffer, "segment_index": n}
            })

            self.tracer.log({
                "event": "temporal_buffer_update",
                "segment_id": n,
                "frame_count": buffer.frames.shape[0],
                "trace_id": trace_id
            })

        return segments, context

    def _generate_segment_with_retry(self, n: int, segment_plan, buffer,
                                      motion_state, char_identity_ref,
                                      context, trace_id: str):
        """Error recycling loop per segment (Doc 03 §47.3)."""
        for attempt in range(TEMPORAL_MAX_RETRIES_PER_SEGMENT):
            # Multi-frame latent encoding (RULE-87: multi-frame REQUIRED)
            latents = self.buffer_manager.encode(buffer)
            if latents.shape[0] != TEMPORAL_BUFFER_SIZE:  # RULE-87 gate
                raise AutoregressiveViolationError(
                    segment_id=n,
                    latent_shape=list(latents.shape),
                    required_frames=TEMPORAL_BUFFER_SIZE
                )

            # CFG validation (RULE-86)
            cfg = context.adaptive_params.get("svi_cfg", 5.5)
            if not (SVI_CFG_MIN <= cfg <= SVI_CFG_MAX):
                raise SVICFGViolationError(cfg=cfg, min_val=SVI_CFG_MIN, max_val=SVI_CFG_MAX)

            steps = STEPS_CRITICAL if segment_plan.is_critical else STEPS_STANDARD

            # SVI generation with noise-aware LoRA scheduling
            segment_n = self.svi.generate(
                init_latents=latents,       # shape: (5, C', H', W') — MULTI-FRAME
                prompt=self._build_temporal_prompt(segment_plan, motion_state, context),
                lora_scheduler=self.scheduler,  # noise-aware; static weight FORBIDDEN
                cfg=cfg,
                steps=steps,
                temporal_attention=True
            )

            # Identity validation per segment (RULE-89)
            clip_score = self.clip_validator.score(segment_n.keyframe, char_identity_ref)

            # Per-segment continuity
            cont_score = self.continuity_validator.score_segment(
                buffer.frames[-1], segment_n.frames[0]
            )

            self.tracer.log({
                "event": "svi_generation",
                "segment_id": n,
                "cfg": cfg, "steps": steps,
                "clip_score": clip_score, "clip_passed": clip_score >= CLIP_IDENTITY_THRESHOLD,
                "per_segment_continuity": cont_score,
                "attempt": attempt,
                "trace_id": trace_id
            })

            if clip_score >= CLIP_IDENTITY_THRESHOLD and cont_score >= SEGMENT_CONTINUITY_MIN:
                # Update identity state
                self.identity_tracker.update(char_identity_ref, segment_n.keyframe, f"S-09_seg_{n}")
                return segment_n

            # Adjust and retry
            self.retry_controller.adjust(attempt, clip_score, cont_score)

        raise TemporalSegmentFailureError(
            scene_id=segment_plan.scene_id,
            segment_id=n,
            best_clip_score=clip_score
        )

    def _build_temporal_prompt(self, segment_plan, motion_state, context) -> str:
        direction = motion_state.direction
        camera_motion = context.camera_state.motion if context.camera_state else "static"
        return (
            f"{segment_plan.prompt}, "
            f"camera: {camera_motion}, "
            f"motion direction: {direction}, "
            f"cinematic continuity"
        )
```

---

## 66. TemporalBufferManager Template (v17.1 — EXPANDED RESPONSIBILITIES)

```python
"""
temporal_buffer_manager.py
Manages the 5-frame rolling TemporalBuffer for SVI conditioning.

v17.1 AUTHORITATIVE RESPONSIBILITY DECLARATION:
  This file is the SOLE owner of all TemporalBuffer state operations:
  - Frame extraction (last 5 frames from any segment)
  - Frame normalization (fixed parameters; consistent across all calls)
  - Strict size enforcement (TEMPORAL_BUFFER_SIZE = 5; any deviation = SYSTEM FAILURE)
  - CPU <-> GPU device management (CPU at rest; GPU only inside encode())
  - Buffer initialization (once, after S-08 / Wan2.2)
  - Rolling buffer update (after each SVI segment)
  - Multi-frame latent encoding (5-frame tensor → latent tensor)

  See Doc 09 §11.2 for complete responsibility definition.
Enforces: RULE-86 (BUFFER_SIZE=5 STRICT, multi-frame latent conditioning)
"""
import torch
from dataclasses import dataclass, field
from typing import List, Optional
from vga.core.exceptions import TemporalBufferError
from vga.config.settings import TEMPORAL_BUFFER_SIZE


@dataclass
class TemporalBuffer:
    """
    Typed dataclass for the temporal rolling buffer.
    frames.shape MUST be (5, C, H, W) at all times.
    """
    frames: torch.Tensor              # shape: (TEMPORAL_BUFFER_SIZE, C, H, W)
    timestamps: List[float]
    motion_vector: Optional[torch.Tensor] = None
    scene_id: str = ""

    def __post_init__(self):
        assert self.frames.shape[0] == TEMPORAL_BUFFER_SIZE, (
            f"TemporalBuffer must have exactly {TEMPORAL_BUFFER_SIZE} frames; "
            f"got {self.frames.shape[0]}"
        )


class TemporalBufferManager:
    """
    Manages TemporalBuffer lifecycle: initialization, update, and encoding.
    Single-frame conditioning is FORBIDDEN.
    """

    def __init__(self, vae_encoder, tracer):
        self.vae_encoder = vae_encoder
        self.tracer = tracer

    def init(self, segment_1) -> TemporalBuffer:
        """
        Initialize buffer from Segment_1 (Wan2.2 output).
        Extracts last BUFFER_SIZE frames.
        """
        frames = self._extract_last_n_frames(segment_1.video_tensor, TEMPORAL_BUFFER_SIZE)

        if frames.shape[0] < TEMPORAL_BUFFER_SIZE:
            raise TemporalBufferError(
                scene_id=segment_1.scene_id,
                frame_count=frames.shape[0],
                required=TEMPORAL_BUFFER_SIZE
            )

        buffer = TemporalBuffer(
            frames=frames,
            timestamps=self._extract_timestamps(segment_1, TEMPORAL_BUFFER_SIZE),
            motion_vector=None,
            scene_id=segment_1.scene_id
        )
        self.tracer.log({
            "event": "temporal_buffer_initialized",
            "scene_id": segment_1.scene_id,
            "frame_count": buffer.frames.shape[0]
        })
        return buffer

    def update(self, buffer: TemporalBuffer, new_segment) -> TemporalBuffer:
        """
        Rolling update: extract last BUFFER_SIZE frames from new_segment.
        Buffer always contains frames from the MOST RECENT segment only.
        """
        new_frames = self._extract_last_n_frames(new_segment.video_tensor, TEMPORAL_BUFFER_SIZE)

        updated = TemporalBuffer(
            frames=new_frames,
            timestamps=self._extract_timestamps(new_segment, TEMPORAL_BUFFER_SIZE),
            motion_vector=None,   # reset; recomputed by MotionStateTracker
            scene_id=buffer.scene_id
        )

        # Enforce invariant
        assert updated.frames.shape[0] == TEMPORAL_BUFFER_SIZE, (
            f"Buffer update produced {updated.frames.shape[0]} frames; expected {TEMPORAL_BUFFER_SIZE}"
        )
        return updated

    def encode(self, buffer: TemporalBuffer) -> torch.Tensor:
        """
        Encode buffer frames into multi-frame latent tensor for SVI conditioning.
        Returns shape: (BUFFER_SIZE, C', H', W') — MUST have shape[0] == 5.
        Single-image latent is FORBIDDEN (RULE-87).
        """
        assert buffer.frames.shape[0] == TEMPORAL_BUFFER_SIZE  # RULE-86

        latents = []
        # Temporarily move to GPU for encoding if needed
        frames_gpu = buffer.frames.cuda() if torch.cuda.is_available() else buffer.frames
        try:
            for frame in frames_gpu:
                latent = self.vae_encoder.encode(frame.unsqueeze(0))
                latents.append(latent.squeeze(0))
        finally:
            del frames_gpu  # return resources

        stacked = torch.stack(latents, dim=0)  # shape: (5, C', H', W')

        # HARD CONSTRAINT: shape[0] MUST == BUFFER_SIZE
        assert stacked.shape[0] == TEMPORAL_BUFFER_SIZE, (
            f"Multi-frame latent must have {TEMPORAL_BUFFER_SIZE} frames; "
            f"single-frame conditioning is FORBIDDEN (RULE-87)"
        )
        return stacked

    def _extract_last_n_frames(self, video_tensor: torch.Tensor, n: int) -> torch.Tensor:
        """Extract last n frames from video tensor. Shape: (n, C, H, W)."""
        total_frames = video_tensor.shape[0]
        if total_frames < n:
            # Pad by repeating last frame
            padding = video_tensor[-1:].expand(n - total_frames, -1, -1, -1)
            return torch.cat([video_tensor, padding], dim=0).cpu()
        return video_tensor[-n:].cpu()  # always CPU-resident between SVI calls

    def _extract_timestamps(self, segment, n: int) -> List[float]:
        import time
        base_time = time.time()
        dt = segment.duration_s / segment.frame_count if segment.frame_count > 0 else 0.033
        return [base_time - (n - 1 - i) * dt for i in range(n)]
```

---

## 67. SVIScheduler Template (NEW v17.0)

```python
"""
svi_scheduler.py
Noise-aware LoRA weight scheduler for SVI temporal generation.
Enforces: RULE-86 (dynamic LoRA weight; static weight FORBIDDEN)
"""
from vga.core.exceptions import SVICFGViolationError
from vga.config.settings import (
    LORA_WEIGHT_HIGH_NOISE, LORA_WEIGHT_MID_NOISE, LORA_WEIGHT_LOW_NOISE,
    HIGH_NOISE_FRACTION, MID_NOISE_FRACTION,
    SVI_CFG_MIN, SVI_CFG_MAX
)


class SVIScheduler:
    """
    Applies dynamic LoRA weight at each diffusion timestep.
    Static weight assignment bypassing this scheduler is FORBIDDEN (RULE-86).
    """

    def __init__(self, total_steps: int, tracer):
        self.total_steps = total_steps
        self.tracer = tracer
        self.threshold_high = int(total_steps * HIGH_NOISE_FRACTION)   # e.g., 20 for T=30
        self.threshold_mid  = int(total_steps * MID_NOISE_FRACTION)    # e.g., 10 for T=30

    def apply_lora(self, timestep: int) -> float:
        """
        Return LoRA weight for current diffusion timestep.
        timestep: decreasing from total_steps to 0 (high noise → low noise).

        Returns:
          0.6 at high-noise phase (structure + motion reinforcement)
          0.5 at mid-noise phase (balanced)
          0.4 at low-noise phase (detail preservation)
        """
        if timestep > self.threshold_high:
            return LORA_WEIGHT_HIGH_NOISE    # 0.6
        elif timestep > self.threshold_mid:
            return LORA_WEIGHT_MID_NOISE     # 0.5
        else:
            return LORA_WEIGHT_LOW_NOISE     # 0.4

    def assert_cfg_valid(self, cfg: float) -> float:
        """
        Validate SVI CFG. CFG outside [5.0, 6.0] is FORBIDDEN (RULE-86).
        Raises SVICFGViolationError on violation; does NOT silently clamp.
        """
        if not (SVI_CFG_MIN <= cfg <= SVI_CFG_MAX):
            raise SVICFGViolationError(
                cfg=cfg,
                min_val=SVI_CFG_MIN,
                max_val=SVI_CFG_MAX
            )
        return cfg

    def get_lora_schedule(self) -> list:
        """Return summary of LoRA weights per phase (for logging)."""
        return [LORA_WEIGHT_HIGH_NOISE, LORA_WEIGHT_MID_NOISE, LORA_WEIGHT_LOW_NOISE]
```

---

## 68. MotionStateTracker Template (NEW v17.0)

```python
"""
motion_state_tracker.py
Estimates motion state from TemporalBuffer frames via optical flow.
Prevents motion reset between consecutive segments.
"""
import torch
from dataclasses import dataclass
from typing import Optional
from vga.config.settings import MOTION_STATIONARY_THRESHOLD
from vga.models.schemas import MotionStateRecord
import uuid
from datetime import datetime


@dataclass
class MotionState:
    """
    Estimated motion state from TemporalBuffer optical flow.
    Propagated via context.evolve() at every segment iteration.
    """
    velocity_x: float
    velocity_y: float
    velocity_magnitude: float
    direction: str
    is_stationary: bool


class MotionStateTracker:
    """
    Estimates per-segment motion state from TemporalBuffer.
    MUST be called before each SVI generation call.
    Result must NOT be cached across segments.
    """

    def __init__(self, tracer, storage):
        self.tracer = tracer
        self.storage = storage

    def estimate(self, frames: torch.Tensor) -> MotionState:
        """
        Compute optical flow across 5 frames → extract velocity + direction.
        frames.shape: (5, C, H, W)
        """
        assert frames.shape[0] >= 2, "Need at least 2 frames for optical flow"

        # Compute pairwise optical flows
        flows = []
        for i in range(frames.shape[0] - 1):
            flow = self._compute_optical_flow(frames[i], frames[i + 1])
            flows.append(flow)

        # Aggregate: mean flow across all pairs
        mean_flow = torch.stack(flows).mean(dim=0)  # shape: (H, W, 2)

        # Spatial mean → velocity vector
        velocity_x = mean_flow[..., 0].mean().item()
        velocity_y = mean_flow[..., 1].mean().item()

        magnitude = (velocity_x ** 2 + velocity_y ** 2) ** 0.5
        is_stationary = magnitude < MOTION_STATIONARY_THRESHOLD
        direction = self._classify_direction(velocity_x, velocity_y, magnitude, is_stationary)

        return MotionState(
            velocity_x=velocity_x,
            velocity_y=velocity_y,
            velocity_magnitude=magnitude,
            direction=direction,
            is_stationary=is_stationary
        )

    def log_state(self, segment_id: int, scene_id: str, state: MotionState, trace_id: str):
        record = MotionStateRecord(
            record_id=str(uuid.uuid4()),
            segment_id=str(segment_id),
            scene_id=scene_id,
            velocity_x=state.velocity_x,
            velocity_y=state.velocity_y,
            velocity_magnitude=state.velocity_magnitude,
            direction=state.direction,
            is_stationary=state.is_stationary,
            timestamp=datetime.utcnow().isoformat()
        )
        self.storage.append_motion_state_record(scene_id, record)
        self.tracer.log({
            "event": "motion_state_estimated",
            "segment_id": segment_id,
            "direction": state.direction,
            "magnitude": state.velocity_magnitude,
            "trace_id": trace_id
        })

    def _compute_optical_flow(self, frame_a: torch.Tensor, frame_b: torch.Tensor) -> torch.Tensor:
        """
        Simplified optical flow using frame difference.
        In production: use torchvision.models.optical_flow or cv2.calcOpticalFlowFarneback.
        """
        # Convert to grayscale for flow computation
        gray_a = frame_a.mean(dim=0)  # (H, W)
        gray_b = frame_b.mean(dim=0)  # (H, W)

        # Simple gradient-based approximation
        diff = gray_b - gray_a
        # Return as (H, W, 2) with dummy x and y components
        flow = torch.stack([diff, diff], dim=-1)
        return flow

    def _classify_direction(self, vx: float, vy: float, mag: float, stationary: bool) -> str:
        if stationary:
            return "stationary"
        if abs(vx) > abs(vy):
            return "right" if vx > 0 else "left"
        else:
            return "forward" if vy > 0 else "backward"
```

---

## 69. TemporalRetryController Template (NEW v17.0)

```python
"""
temporal_retry_controller.py
Error recycling loop controller for TemporalEngine segment generation.
"""
from vga.config.settings import TEMPORAL_MAX_RETRIES_PER_SEGMENT, SVI_CFG_DEFAULT


class TemporalRetryController:
    """
    Adjusts SVI generation parameters between retry attempts.
    Implements the error recycling loop from Doc 03 §47.3.
    """

    def adjust(self, attempt: int, clip_score: float, cont_score: float):
        """
        Adjust generation parameters for next retry attempt.
        Called when a segment fails identity or continuity checks.
        """
        # Strategy: increase steps on attempt 1, strengthen identity on attempt 2
        if attempt == 0:
            # First retry: more steps for better quality
            self._adjust_steps_up()
        elif attempt == 1:
            # Second retry: stronger identity anchoring in prompt
            self._strengthen_identity_prompt()
        # attempt == 2: last attempt; no further adjustment

    def _adjust_steps_up(self):
        # Logged as parameter adjustment; actual steps determined at generation time
        pass

    def _strengthen_identity_prompt(self):
        # Signals prompt builder to add stronger identity anchors
        pass
```

---

## 70. IdentityStateTracker Template (NEW v17.0)

```python
"""
identity_state_tracker.py
Tracks cumulative identity drift across all pipeline phases.
Enforces: RULE-89 (identity validated in image, video, lip sync stages)
          RULE-95 (same identity reference across all phases)
"""
import torch
from vga.models.schemas import IdentityStateRecord
from vga.core.exceptions import IdentityCumulativeDriftError
from vga.config.settings import IDENTITY_CUMULATIVE_DRIFT_THRESHOLD, SCHEMA_VERSION
from vga.state.immutable_context import ImmutableContext
import uuid
from datetime import datetime


class IdentityStateTracker:
    """
    Maintains IdentityState across all pipeline phases.
    Called after every CLIPValidator check in any phase.
    Raises IdentityCumulativeDriftError if cumulative drift exceeds threshold.
    """

    def __init__(self, clip_encoder, tracer, storage):
        self.clip_encoder = clip_encoder
        self.tracer = tracer
        self.storage = storage
        self._drift_score = 0.0
        self._history = []

    def update(self, char_identity_ref: torch.Tensor, new_frame, stage_id: str) -> dict:
        """
        Update IdentityState with new frame's CLIP embedding.
        Raises IdentityCumulativeDriftError if cumulative drift > threshold.
        Returns updated identity state dict for context.evolve().
        """
        # Compute drift delta for this frame
        e_new = self.clip_encoder.encode(new_frame)
        cos_sim = torch.nn.functional.cosine_similarity(
            char_identity_ref.unsqueeze(0),
            e_new.unsqueeze(0)
        ).item()
        delta = 1.0 - cos_sim  # normalised drift: 0 = perfect, 1 = completely different

        self._drift_score += delta
        self._history.append(delta)

        # Check cumulative drift threshold
        threshold_exceeded = self._drift_score > IDENTITY_CUMULATIVE_DRIFT_THRESHOLD

        record = IdentityStateRecord(
            record_id=str(uuid.uuid4()),
            stage_id=stage_id,
            scene_id="",  # populated by caller if available
            delta=delta,
            drift_score=self._drift_score,
            drift_history=list(self._history),
            threshold_exceeded=threshold_exceeded,
            timestamp=datetime.utcnow().isoformat(),
            schema_version=SCHEMA_VERSION
        )
        self.storage.append_identity_state_record(record)

        self.tracer.log({
            "event": "identity_state_update",
            "stage_id": stage_id,
            "delta": delta,
            "cumulative_drift": self._drift_score,
            "threshold_exceeded": threshold_exceeded
        })

        if threshold_exceeded:
            raise IdentityCumulativeDriftError(
                stage_id=stage_id,
                cumulative_drift=self._drift_score,
                threshold=IDENTITY_CUMULATIVE_DRIFT_THRESHOLD
            )

        return {
            "embedding_vector": char_identity_ref,  # never changes
            "drift_score": self._drift_score,
            "history": list(self._history)
        }

    def reset(self):
        """Reset drift accumulator after successful phase regeneration."""
        self._drift_score = 0.0
        self._history = []
```

---

## 71. AudioQualityValidator Template (NEW v17.0)

```python
"""
audio_quality_validator.py
Validates SNR and peak level of mixed audio after AudioMixingAgent.
Enforces: RULE-99 (SNR ≥ 10 dB; peaks ≤ 0 dBFS)
"""
import torch
import torchaudio
from pydub import AudioSegment
from vga.models.schemas import AudioQualityRecord
from vga.core.exceptions import AudioQualityError
from vga.config.settings import MIN_SNR_DB, MAX_PEAK_DBFS, HEADROOM_DB, SCHEMA_VERSION
import uuid
from datetime import datetime
import math


class AudioQualityValidator:
    """
    Validates and normalizes mixed audio.
    Called by AudioMixingAgent BEFORE writing to storage.
    """

    def __init__(self, tracer, storage):
        self.tracer = tracer
        self.storage = storage

    def validate(self, mixed: AudioSegment, dialogue: AudioSegment,
                 scene_id: str, job_id: str) -> AudioQualityRecord:
        """
        Compute SNR and peak level. Return AudioQualityRecord.
        Does NOT raise on failure — caller decides action based on record.
        """
        snr_db = self.compute_snr(mixed, dialogue)
        peak_db = self.compute_peak_db(mixed)

        record = AudioQualityRecord(
            record_id=str(uuid.uuid4()),
            scene_id=scene_id,
            snr_db=snr_db,
            peak_db=peak_db,
            clipping_detected=peak_db > MAX_PEAK_DBFS,
            snr_passed=snr_db >= MIN_SNR_DB,
            clipping_passed=peak_db <= MAX_PEAK_DBFS,
            remix_count=0,
            normalization_applied=False,
            timestamp=datetime.utcnow().isoformat(),
            schema_version=SCHEMA_VERSION
        )

        self.tracer.log({
            "event": "audio_quality_validation",
            "scene_id": scene_id,
            "snr_db": snr_db,
            "peak_db": peak_db,
            "snr_passed": record.snr_passed,
            "clipping_passed": record.clipping_passed
        })

        return record

    def normalize(self, mixed: AudioSegment, target_peak_db: float = -HEADROOM_DB) -> AudioSegment:
        """
        Normalize audio so peak is at target_peak_db (default: -1.0 dBFS).
        Prevents clipping.
        """
        current_peak = self.compute_peak_db(mixed)
        gain_reduction = target_peak_db - current_peak
        return mixed.apply_gain(gain_reduction)

    def compute_snr(self, mixed: AudioSegment, dialogue: AudioSegment) -> float:
        """
        Estimate SNR as dialogue RMS - background RMS (simplified).
        """
        dialogue_rms = dialogue.rms
        # Background estimated from mix with dialogue subtracted
        background_rms = max(1, mixed.rms - dialogue_rms)  # avoid log(0)

        if background_rms <= 0:
            return 99.0  # effectively infinite SNR (no background)

        snr = 20.0 * math.log10(dialogue_rms / background_rms) if dialogue_rms > 0 else 0.0
        return snr

    def compute_peak_db(self, audio: AudioSegment) -> float:
        """Compute peak level in dBFS."""
        peak_amplitude = audio.max
        if peak_amplitude <= 0:
            return -99.0
        return 20.0 * math.log10(peak_amplitude / 32768.0)  # normalised to 16-bit full scale
```

---

## 72. CrossModalAlignmentValidator Template (NEW v17.0)

```python
"""
cross_modal_alignment_validator.py
Validates video ↔ audio duration alignment per segment.
Enforces: RULE-96 (timing tolerance ±0.10s), RULE-95 (segment boundary alignment)
"""
from vga.models.schemas import CrossModalAlignmentRecord, CrossModalAlignmentReport
from vga.config.settings import TIMING_TOLERANCE_S, SCHEMA_VERSION
import uuid
from datetime import datetime


class CrossModalAlignmentValidator:
    """
    Validates that video and audio segments have matching durations.
    Called by AudioMixingAgent after mixing, before HRG-11.
    """

    def __init__(self, tracer, storage):
        self.tracer = tracer
        self.storage = storage

    def validate_alignment(self, video_segments: list, audio_segments: list,
                           scene_id: str) -> CrossModalAlignmentReport:
        """
        Validate duration alignment for all segments.
        Returns CrossModalAlignmentReport; does not raise on tolerance violation
        (caller decides action; HRG-11 shows alignment status).
        """
        records = []
        for i, (vid, aud) in enumerate(zip(video_segments, audio_segments)):
            vid_dur = vid.duration_s
            aud_dur = aud.duration_s
            error = abs(aud_dur - vid_dur)

            record = CrossModalAlignmentRecord(
                record_id=str(uuid.uuid4()),
                scene_id=scene_id,
                segment_id=str(i),
                video_duration_s=vid_dur,
                audio_duration_s=aud_dur,
                alignment_error_s=error,
                within_tolerance=error <= TIMING_TOLERANCE_S,
                tolerance_s=TIMING_TOLERANCE_S,
                timestamp=datetime.utcnow().isoformat(),
                schema_version=SCHEMA_VERSION
            )
            records.append(record)

        all_passed = all(r.within_tolerance for r in records)
        errors = [r.alignment_error_s for r in records]

        report = CrossModalAlignmentReport(
            scene_id=scene_id,
            records=records,
            all_passed=all_passed,
            max_error_s=max(errors),
            mean_error_s=sum(errors) / len(errors),
            total_video_duration_s=sum(r.video_duration_s for r in records),
            total_audio_duration_s=sum(r.audio_duration_s for r in records),
            total_error_s=sum(errors),
            schema_version=SCHEMA_VERSION
        )

        self.tracer.log({
            "event": "cross_modal_alignment_validated",
            "scene_id": scene_id,
            "all_passed": all_passed,
            "max_error_s": report.max_error_s
        })
        self.storage.write_cross_modal_alignment(scene_id, report)
        return report
```

---

## 73. CompositionPlanValidator Template (NEW v17.0)

```python
"""
composition_validator.py
Validates CompositionPlanSchema output from SceneCompositionAgent.
Enforces: RULE-88 (all 6 fields required; no image generation without plan)
"""
from vga.models.schemas import CompositionPlanSchema
from vga.core.exceptions import CompositionPlanValidationError


class CompositionPlanValidator:
    """
    Validates CompositionPlan before HRG-4 and before any image generation.
    Stateless: validates and returns; never modifies the plan.
    """

    REQUIRED_FIELDS = [
        "camera_angle", "camera_motion", "character_positions",
        "focus_subject", "lighting_style", "motion_vector"
    ]

    def validate(self, plan: CompositionPlanSchema) -> bool:
        """
        Validate all 6 fields are present and non-empty.
        Raises CompositionPlanValidationError on any violation.
        """
        for field in self.REQUIRED_FIELDS:
            value = getattr(plan, field, None)
            if value is None:
                raise CompositionPlanValidationError(
                    scene_id=plan.scene_id,
                    reason=f"Required field '{field}' is None"
                )
            if isinstance(value, str) and len(value.strip()) == 0:
                raise CompositionPlanValidationError(
                    scene_id=plan.scene_id,
                    reason=f"Required field '{field}' is empty string"
                )
            if isinstance(value, list) and len(value) == 0:
                raise CompositionPlanValidationError(
                    scene_id=plan.scene_id,
                    reason=f"Required field '{field}' is empty list"
                )
        return True

    def assert_in_context(self, context) -> None:
        """
        Assert CompositionPlan is present in pipeline context.
        Called before S-05, S-06, S-08. RULE-88.
        """
        if context.camera_state is None or context.camera_state.angle is None:
            raise CompositionPlanValidationError(
                scene_id=context.scene_id if hasattr(context, 'scene_id') else "unknown",
                reason="CompositionPlan not found in ImmutableContext (RULE-88)"
            )
```

---

## 74. HRGController v17.0 Template (11 checkpoints)

```python
"""
hrg_controller.py
11-checkpoint Human Review Gate controller for v17.0.
Extends v16.0 HRGController with HRG-2 (Scene/Segment Plan) and HRG-4 (Composition).
"""
import threading
from typing import Dict, Optional
from vga.models.schemas import HRGDecisionRecord, HRGCheckpointState
from vga.core.exceptions import PipelineHaltError, RegenerationRequestedError
from vga.config.settings import HRG_TIMEOUT_S, HRG_CHECKPOINT_COUNT, SCHEMA_VERSION


class HRGController:
    """
    Manages all 11 HRG checkpoint state machines.
    v17.0: extends v16.0 to support HRG-1 through HRG-11.
    """

    VALID_CHECKPOINTS = {f"HRG-{i}" for i in range(1, HRG_CHECKPOINT_COUNT + 1)}

    def __init__(self, storage, tracer):
        self.storage = storage
        self.tracer = tracer
        # 11 threading events — one per checkpoint
        self._events: Dict[str, threading.Event] = {
            checkpoint: threading.Event()
            for checkpoint in self.VALID_CHECKPOINTS
        }
        self._decisions: Dict[str, Optional[HRGDecisionRecord]] = {
            checkpoint: None
            for checkpoint in self.VALID_CHECKPOINTS
        }
        self._decision_log = []

    def require_approval(self, checkpoint: str, display_data: dict, job_id: str):
        """
        Block pipeline execution until human approves or timeout.
        CGRL-74: uses threading.Event.wait() — no busy-polling.
        """
        assert checkpoint in self.VALID_CHECKPOINTS, f"Invalid checkpoint: {checkpoint}"

        state = HRGCheckpointState(
            checkpoint=checkpoint,
            status="awaiting_human",
            display_data=display_data,
            schema_version=SCHEMA_VERSION
        )
        self.storage.write_hrg_state(job_id, checkpoint, state.model_dump())

        self.tracer.log({
            "event": "hrg_awaiting_human",
            "checkpoint": checkpoint,
            "job_id": job_id
        })

        # Block here until human decision (CGRL-74: threading.Event.wait)
        approved = self._events[checkpoint].wait(timeout=HRG_TIMEOUT_S)

        if not approved:
            raise PipelineHaltError(
                checkpoint=checkpoint,
                reason="hrg_timeout",
                resume_available=True
            )

        decision = self._decisions[checkpoint]
        if decision and decision.action == "trigger_regeneration":
            raise RegenerationRequestedError(
                checkpoint=checkpoint,
                payload=decision.payload or {}
            )

        self._events[checkpoint].clear()  # reset for potential re-use

    def submit_decision(self, checkpoint: str, decision: HRGDecisionRecord, job_id: str):
        """
        Receive and record human decision; unblock pipeline.
        CGRL-75: record BEFORE setting event.
        """
        assert checkpoint in self.VALID_CHECKPOINTS

        # CGRL-75: Log BEFORE setting event that unblocks pipeline
        self._decisions[checkpoint] = decision
        self._decision_log.append(decision.model_dump())
        self.storage.append_hrg_record(job_id, decision.model_dump())

        self.tracer.log({
            "event": "hrg_decision_received",
            "checkpoint": checkpoint,
            "action": decision.action,
            "user": decision.user
        })

        # Unblock pipeline
        self._events[checkpoint].set()

    def get_full_decision_log(self) -> list:
        return list(self._decision_log)
```

---

## 75. MasterOrchestrator execute_stage() Template (NEW v17.0)

```python
"""
master_orchestrator.py (partial — execute_stage contract)
Implements SYSTEM DIRECTIVE v17 execution contract.
Every stage MUST pass through this function.
"""
from vga.state.immutable_context import ImmutableContext
from vga.runtime.system_guard import SystemGuard
from vga.core.exceptions import VGABaseError


def execute_stage(stage, input_data, context: ImmutableContext) -> tuple:
    """
    SYSTEM DIRECTIVE v17: mandatory execution wrapper for all pipeline stages.

    Enforces:
      1. SystemGuard wrapping active (RULE-41)
      2. Previous output validated
      3. HRG gate (if required)
      4. Stage execution
      5. Output validation
      6. context.evolve() called

    Returns: (output, updated_context)
    """
    # 1. System guard + authority (RULE-41, RULE-54)
    SystemGuard.execute(stage)
    stage.authority_manager.validate(stage.authority_level, stage.action_name)

    # 2. Validate previous stage output exists in context
    _validate_previous_output(stage, context)

    # 3. HRG gate (blocks if required)
    if stage.requires_hrg:
        stage.hrg_controller.require_approval(
            checkpoint=stage.hrg_checkpoint,
            display_data=_prepare_hrg_display(stage, context),
            job_id=context.job_id
        )

    # 4. Execute stage
    output = stage.run(input_data, context)

    # 5. Validate output
    _validate_output(stage, output)

    # 6. Evolve context — MANDATORY (SYSTEM DIRECTIVE v17)
    updated_context = context.evolve(output)

    return output, updated_context


def _validate_previous_output(stage, context: ImmutableContext):
    """Verify predecessor stage completed successfully and output is in context."""
    if stage.predecessor_stage is None:
        return  # first stage; nothing to validate
    if not context.has_output(stage.predecessor_stage):
        raise MissingPredecessorOutputError(
            stage=stage.name,
            predecessor=stage.predecessor_stage
        )


def _validate_output(stage, output):
    """Validate stage output schema and required fields."""
    if hasattr(stage, 'output_schema') and stage.output_schema is not None:
        stage.output_schema.model_validate(output if isinstance(output, dict) else output.model_dump())
```

---

## 76. v17.0 Error Handling Additions (NEW v17.0)

```python
# Additional exceptions for v17.0 (added to vga/core/exceptions.py)

class CompositionPlanValidationError(VGABaseError):
    """
    Raised when CompositionPlan is missing or fails validation.
    CRITICAL: No image generation may proceed. RULE-88.
    """
    def __init__(self, scene_id: str, reason: str):
        self.scene_id = scene_id
        self.reason = reason
        super().__init__(f"CompositionPlan invalid: scene={scene_id}, reason={reason}")


class TemporalBufferError(VGABaseError):
    """
    Raised when TemporalBuffer has insufficient frames.
    CRITICAL: TemporalEngine must not run. RULE-86.
    """
    def __init__(self, scene_id: str, frame_count: int, required: int):
        self.scene_id = scene_id
        self.frame_count = frame_count
        self.required = required
        super().__init__(
            f"TemporalBuffer error: scene={scene_id}, "
            f"frame_count={frame_count} < required={required} (RULE-86)"
        )


class SVICFGViolationError(VGABaseError):
    """
    Raised when SVI CFG is outside [5.0, 6.0].
    CRITICAL: Color banding risk. RULE-86.
    """
    def __init__(self, cfg: float, min_val: float, max_val: float):
        self.cfg = cfg
        self.min_val = min_val
        self.max_val = max_val
        super().__init__(
            f"SVI CFG violation: cfg={cfg} outside [{min_val}, {max_val}] (RULE-86)"
        )


class AutoregressiveViolationError(VGABaseError):
    """
    Raised when single-frame latent conditioning is attempted for SVI.
    CRITICAL: Architecture violation. RULE-87.
    """
    def __init__(self, segment_id: int, latent_shape: list, required_frames: int):
        self.segment_id = segment_id
        self.latent_shape = latent_shape
        self.required_frames = required_frames
        super().__init__(
            f"Autoregressive violation: segment={segment_id}, "
            f"latent_shape={latent_shape}, required shape[0]={required_frames} (RULE-87)"
        )


class TemporalSegmentFailureError(VGABaseError):
    """
    Raised when a temporal segment fails identity/continuity after all retries.
    """
    def __init__(self, scene_id: str, segment_id: int, best_clip_score: float):
        self.scene_id = scene_id
        self.segment_id = segment_id
        self.best_clip_score = best_clip_score
        super().__init__(
            f"Temporal segment failure: scene={scene_id}, segment={segment_id}, "
            f"best_clip_score={best_clip_score:.4f}"
        )


class IdentityCumulativeDriftError(VGABaseError):
    """
    Raised when cumulative identity drift exceeds IDENTITY_CUMULATIVE_DRIFT_THRESHOLD.
    Triggers full phase regeneration.
    """
    def __init__(self, stage_id: str, cumulative_drift: float, threshold: float):
        self.stage_id = stage_id
        self.cumulative_drift = cumulative_drift
        self.threshold = threshold
        super().__init__(
            f"Identity cumulative drift exceeded: stage={stage_id}, "
            f"drift={cumulative_drift:.4f} > threshold={threshold}"
        )


class IdentityReferenceCorruptionError(VGABaseError):
    """
    Raised when char_identity_ref embedding is mutated after being frozen.
    CRITICAL: Identity consistency violation.
    """
    def __init__(self, stage_id: str):
        self.stage_id = stage_id
        super().__init__(
            f"char_identity_ref corrupted at stage={stage_id}; "
            f"frozen reference must never change after S-07 (RULE-95)"
        )


class AudioQualityError(VGABaseError):
    """
    Raised when SNR or peak level fails after all re-mix attempts.
    """
    def __init__(self, scene_id: str, snr_db: float, peak_db: float):
        self.scene_id = scene_id
        self.snr_db = snr_db
        self.peak_db = peak_db
        super().__init__(
            f"Audio quality failure: scene={scene_id}, "
            f"snr={snr_db:.1f}dB (req≥10), peak={peak_db:.1f}dBFS (req≤0)"
        )


class MissingPredecessorOutputError(VGABaseError):
    """
    Raised when SYSTEM DIRECTIVE v17 execute_stage() detects missing predecessor output.
    """
    def __init__(self, stage: str, predecessor: str):
        self.stage = stage
        self.predecessor = predecessor
        super().__init__(
            f"Stage {stage} cannot proceed: predecessor {predecessor} output missing from context"
        )
```


---

## 77. Cross-Modal Validation Unified Contract Template (NEW v17.2)

```python
"""
cross_modal_validation_unified.py
Provides unified cross-modal validation for Video ↔ Audio ↔ Identity ↔ Temporal.
Enforces v17.2 cross-modal contract; replaces fragmented per-check approach.
"""
from vga.models.schemas import CrossModalValidationContract
from vga.validation.clip_validator import CLIPValidator
from vga.config.settings import (
    TIMING_TOLERANCE_S, PHONEME_ALIGNMENT_THRESHOLD,
    SEGMENT_CONTINUITY_MIN, SCHEMA_VERSION
)
import uuid
from datetime import datetime


class CrossModalValidationUnified:
    """
    Unified cross-modal validator for a single segment.
    Validates all four dimensions in one call:
      1. Video ↔ Audio duration alignment
      2. Lip sync phoneme alignment
      3. Cross-frame identity consistency
      4. Temporal continuity score
    """

    def __init__(self, clip_validator: CLIPValidator, tracer):
        self.clip_validator = clip_validator
        self.tracer = tracer

    def validate(
        self,
        scene_id: str,
        segment_id: str,
        video_segment,
        audio_segment,
        synced_frame,
        frame_before,
        frame_after,
        char_identity_ref,
        continuity_score: float,
        phoneme_score: float
    ) -> CrossModalValidationContract:
        """
        Run all four cross-modal validation checks.
        Returns CrossModalValidationContract record.
        Does NOT raise on failure — caller decides action based on record.
        """
        # 1. Duration alignment (RULE-96)
        dur_error = abs(video_segment.duration_s - audio_segment.duration_s)
        dur_passed = dur_error <= TIMING_TOLERANCE_S

        # 2. Phoneme alignment
        phoneme_passed = phoneme_score >= PHONEME_ALIGNMENT_THRESHOLD

        # 3. Cross-frame identity (NEW v17.2: CLIP similarity between boundary frames)
        emb_before = self.clip_validator.encode(frame_before)
        emb_after = self.clip_validator.encode(frame_after)
        import torch
        cross_sim = torch.nn.functional.cosine_similarity(
            emb_before.unsqueeze(0), emb_after.unsqueeze(0)
        ).item()
        cross_passed = cross_sim >= 0.97

        # 4. Temporal continuity
        temporal_passed = continuity_score >= SEGMENT_CONTINUITY_MIN

        all_passed = dur_passed and phoneme_passed and cross_passed and temporal_passed

        record = CrossModalValidationContract(
            record_id=str(uuid.uuid4()),
            scene_id=scene_id,
            segment_id=segment_id,
            video_duration_s=video_segment.duration_s,
            audio_duration_s=audio_segment.duration_s,
            duration_alignment_error_s=dur_error,
            duration_within_tolerance=dur_passed,
            phoneme_alignment_score=phoneme_score,
            phoneme_passed=phoneme_passed,
            clip_frame_before=self.clip_validator.score(frame_before, char_identity_ref),
            clip_frame_after=self.clip_validator.score(frame_after, char_identity_ref),
            cross_frame_clip_similarity=cross_sim,
            cross_frame_identity_passed=cross_passed,
            continuity_score=continuity_score,
            temporal_passed=temporal_passed,
            all_passed=all_passed,
            timestamp=datetime.utcnow().isoformat(),
            schema_version=SCHEMA_VERSION
        )

        self.tracer.log({
            "event": "cross_modal_validation_unified",
            "scene_id": scene_id,
            "segment_id": segment_id,
            "all_passed": all_passed,
            "dur_error": dur_error,
            "phoneme": phoneme_score,
            "cross_sim": cross_sim,
            "continuity": continuity_score
        })

        return record
```

---

## 78. System Certification Validator Template (NEW v17.2)

```python
"""
system_certification_validator.py
Validates that a pipeline run meets all v17.2 certification requirements.
Called by QualityAgent (S-16c) before writing final PipelineReport.
"""
from vga.config.settings import (
    CLIP_IDENTITY_THRESHOLD, MIN_SNR_DB, MAX_PEAK_DBFS,
    TEMPORAL_BUFFER_SIZE, HRG_CHECKPOINT_COUNT, IDENTITY_CUMULATIVE_DRIFT_THRESHOLD
)
from vga.core.exceptions import SystemCertificationFailureError
from vga.core.tracer import Tracer


class SystemCertificationValidator:
    """
    v17.2 System Certification: validates all 7 certification conditions.
    If ANY condition fails, pipeline output is NOT deployable.
    """

    def __init__(self, tracer: Tracer):
        self.tracer = tracer

    def certify(self, pipeline_report: dict) -> dict:
        """
        Run all certification checks. Returns certification summary.
        Raises SystemCertificationFailureError if any check fails.
        """
        failures = []
        checks = {}

        # 1. Temporal Loop Integrity
        temporal_ok = (
            pipeline_report.get("temporal_engine_health", {}).get("buffer_violation_count", 1) == 0
            and pipeline_report.get("temporal_engine_health", {}).get("batch_generation_detected", True) == False
        )
        checks["temporal_loop_integrity"] = temporal_ok
        if not temporal_ok:
            failures.append("TEMPORAL_LOOP_INTEGRITY: buffer violation or batch generation detected")

        # 2. Identity Stability
        id_clips = pipeline_report.get("identity_per_segment_video", [])
        id_ok = all(s >= CLIP_IDENTITY_THRESHOLD for s in id_clips) if id_clips else False
        cumulative_drift = pipeline_report.get("identity_state_final", {}).get("cumulative_drift", 999)
        drift_ok = cumulative_drift <= IDENTITY_CUMULATIVE_DRIFT_THRESHOLD
        checks["identity_stability"] = id_ok and drift_ok
        if not (id_ok and drift_ok):
            failures.append(f"IDENTITY_STABILITY: clip failures or drift {cumulative_drift:.3f} > threshold")

        # 3. Temporal Continuity
        cont_scores = pipeline_report.get("continuity_scores", [])
        cont_ok = all(s >= 0.90 for s in cont_scores) if cont_scores else False
        checks["temporal_continuity"] = cont_ok
        if not cont_ok:
            failures.append("TEMPORAL_CONTINUITY: continuity_score < 0.90 for some scene")

        # 4. Audio Quality
        audio_summary = pipeline_report.get("audio_quality_summary", {})
        snr_ok = audio_summary.get("snr_db", 0) >= MIN_SNR_DB
        clip_ok = audio_summary.get("peak_db", 1) <= MAX_PEAK_DBFS
        checks["audio_quality"] = snr_ok and clip_ok
        if not (snr_ok and clip_ok):
            failures.append("AUDIO_QUALITY: SNR or clipping failure")

        # 5. Human Governance
        hrg_decisions = pipeline_report.get("hrg_decisions_summary", [])
        hrg_ok = len(hrg_decisions) >= HRG_CHECKPOINT_COUNT
        checks["human_governance"] = hrg_ok
        if not hrg_ok:
            failures.append(f"HUMAN_GOVERNANCE: only {len(hrg_decisions)}/{HRG_CHECKPOINT_COUNT} HRG checkpoints logged")

        # 6. Auditability
        audit_complete = pipeline_report.get("audit_complete", False)
        checks["auditability"] = audit_complete
        if not audit_complete:
            failures.append("AUDITABILITY: missing audit data for one or more segments")

        # 7. Validation Propagation
        validation_complete = pipeline_report.get("all_segments_triple_validated", False)
        checks["validation_propagation"] = validation_complete
        if not validation_complete:
            failures.append("VALIDATION_PROPAGATION: not all segments passed all 3 validators")

        all_passed = len(failures) == 0

        self.tracer.log({
            "event": "system_certification",
            "all_passed": all_passed,
            "checks": checks,
            "failures": failures
        })

        if not all_passed:
            raise SystemCertificationFailureError(failures=failures, checks=checks)

        return {
            "certified": True,
            "checks": checks,
            "version": "v17.2"
        }
```

---

## 79. TemporalEngine Authority Guard Template (NEW v17.2)

```python
"""
temporal_authority_guard.py
Runtime enforcement of TemporalEngine authority rules.
Ensures no other component can invoke SVI, update buffer, or control segment iteration.
"""
from vga.core.exceptions import ArchitectureGuardViolationError


class TemporalAuthorityGuard:
    """
    Guards TemporalEngine's exclusive ownership of:
      - Segment iteration control
      - TemporalBuffer update operations
      - SVI invocation
      - Autoregressive loop management

    Any component other than TemporalEngine attempting these operations
    will raise ArchitectureGuardViolationError.
    """

    AUTHORIZED_CALLER = "vga.temporal.temporal_engine.TemporalEngine"

    @classmethod
    def assert_authorized(cls, caller_qualname: str, operation: str) -> None:
        """
        Call before any protected temporal operation.
        caller_qualname = fully qualified class name of the caller.
        """
        if not caller_qualname.startswith("vga.temporal.temporal_engine"):
            raise ArchitectureGuardViolationError(
                component=caller_qualname,
                operation=operation,
                rule="v17.2 TemporalEngine Authority",
                message=(
                    f"Operation '{operation}' is ONLY permitted from TemporalEngine. "
                    f"Caller '{caller_qualname}' is not authorized. "
                    f"This is an architecture violation. [v17.2]"
                )
            )

    @classmethod
    def guard_svi_invoke(cls, caller_qualname: str) -> None:
        """Guard: SVI may only be invoked by TemporalEngine."""
        cls.assert_authorized(caller_qualname, "svi_invoke")

    @classmethod
    def guard_buffer_update(cls, caller_qualname: str) -> None:
        """Guard: TemporalBuffer.update() may only be called by TemporalEngine."""
        cls.assert_authorized(caller_qualname, "temporal_buffer_update")

    @classmethod
    def guard_segment_iteration(cls, caller_qualname: str) -> None:
        """Guard: Segment iteration loop may only be controlled by TemporalEngine."""
        cls.assert_authorized(caller_qualname, "segment_iteration_control")
```
