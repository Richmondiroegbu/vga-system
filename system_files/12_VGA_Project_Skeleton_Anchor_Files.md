# VGA Project Skeleton & Anchor Files
**Project:** Video Generation Automation (VGA) — Motivation System
**Version:** 17.2.0
**Status:** Authoritative Reference
**Audience:** Claude Code Agent

---

## Overview

This document provides the authoritative skeleton for every anchor file in the VGA v17.0 project. **Anchor files define module boundaries, public APIs, and structural invariants that all other files must respect.**

**Retained from v15.0:** All §1–§8 anchor files unchanged.
**Retained from v16.0:** All §9–§18 anchor files unchanged.
**New in v17.0:** §19–§30 — anchor files for new v17.0 components.

---

## 1–18: All v15.0 and v16.0 Anchor Files Retained

---

## 19. `vga/config/settings.py` — v17.0 Constants Block (anchor addition)

The v17.0 constants block is appended after all v16.0 constants. This is the authoritative reference for all v17.0 tuning values.

```python
# ════════════════════════════════════════════════════════════════════════
# VGA SETTINGS v17.0 — appended after v16.0 constants (DO NOT REORDER)
# ════════════════════════════════════════════════════════════════════════

# ── Scene Composition ─────────────────────────────────────────────────
COMPOSITION_MAX_RETRIES: int = 3
SLA_COMPOSITION_MAX_S: float = 15.0

# ── Temporal Engine (RULE-86, RULE-87) ───────────────────────────────
TEMPORAL_BUFFER_SIZE: int = 5                    # STRICT: always exactly 5
TEMPORAL_MAX_RETRIES_PER_SEGMENT: int = 3
SEGMENT_CONTINUITY_MIN: float = 0.85             # per-segment (lower than scene 0.90)
SVI_CFG_MIN: float = 5.0                         # SVI CFG lower bound (RULE-86)
SVI_CFG_MAX: float = 6.0                         # SVI CFG upper bound (RULE-86)
SVI_CFG_DEFAULT: float = 5.5                     # recommended production default
STEPS_CRITICAL: int = 50                         # critical segments (scene boundaries)
STEPS_STANDARD: int = 30                         # standard segments
STEPS_PREVIEW: int = 8                           # preview mode only; NOT production
LORA_WEIGHT_HIGH_NOISE: float = 0.6              # timestep t > 0.67*T
LORA_WEIGHT_MID_NOISE: float = 0.5              # timestep 0.33*T < t <= 0.67*T
LORA_WEIGHT_LOW_NOISE: float = 0.4              # timestep t <= 0.33*T
HIGH_NOISE_FRACTION: float = 0.67               # boundary fraction for high noise phase
MID_NOISE_FRACTION: float = 0.33               # boundary fraction for mid noise phase
MOTION_STATIONARY_THRESHOLD: float = 0.02       # normalised pixel displacement

# ── Identity State (RULE-89, RULE-95) ────────────────────────────────
IDENTITY_CUMULATIVE_DRIFT_THRESHOLD: float = 0.15
IDENTITY_MAX_PHASE_REGENERATIONS: int = 1

# ── Audio Quality (RULE-99) ───────────────────────────────────────────
MIN_SNR_DB: float = 10.0                        # minimum dialogue SNR (dB)
MAX_PEAK_DBFS: float = 0.0                      # maximum peak level (dBFS)
HEADROOM_DB: float = 1.0                        # normalization headroom (dB)
AUDIO_QUALITY_MAX_RETRIES: int = 3

# ── HRG (11 checkpoints) ─────────────────────────────────────────────
HRG_CHECKPOINT_COUNT: int = 11

# ── Schema Version ────────────────────────────────────────────────────
SCHEMA_VERSION: str = "v6.0"
```

---

## 20. `vga/core/exceptions.py` — v17.0 Exception Block (anchor addition)

```python
# ════════════════════════════════════════════════════════════════════════
# VGA EXCEPTIONS v17.0 — appended after v16.0 exceptions
# ════════════════════════════════════════════════════════════════════════

class CompositionPlanValidationError(VGABaseError):
    """Raised when CompositionPlan is missing or fails field validation. RULE-88."""
    def __init__(self, scene_id: str, reason: str):
        self.scene_id = scene_id
        self.reason = reason
        super().__init__(f"CompositionPlan invalid: scene={scene_id}, reason={reason}")


class TemporalBufferError(VGABaseError):
    """Raised when TemporalBuffer has wrong frame count. RULE-86. CRITICAL."""
    def __init__(self, scene_id: str, frame_count: int, required: int):
        self.scene_id = scene_id
        self.frame_count = frame_count
        self.required = required
        super().__init__(
            f"TemporalBuffer error: scene={scene_id}, "
            f"frame_count={frame_count}, required={required}"
        )


class SVICFGViolationError(VGABaseError):
    """Raised when SVI CFG is outside [5.0, 6.0]. RULE-86. CRITICAL."""
    def __init__(self, cfg: float, min_val: float, max_val: float):
        self.cfg = cfg
        self.min_val = min_val
        self.max_val = max_val
        super().__init__(f"SVI CFG violation: cfg={cfg} outside [{min_val}, {max_val}]")


class AutoregressiveViolationError(VGABaseError):
    """Raised when single-frame latent conditioning attempted for SVI. RULE-87. CRITICAL."""
    def __init__(self, segment_id: int, latent_shape: list, required_frames: int):
        self.segment_id = segment_id
        self.latent_shape = latent_shape
        self.required_frames = required_frames
        super().__init__(
            f"Autoregressive violation: segment={segment_id}, "
            f"latent_shape={latent_shape}, required shape[0]={required_frames}"
        )


class TemporalSegmentFailureError(VGABaseError):
    """Raised when temporal segment fails after all retries."""
    def __init__(self, scene_id: str, segment_id: int, best_clip_score: float):
        self.scene_id = scene_id
        self.segment_id = segment_id
        self.best_clip_score = best_clip_score
        super().__init__(
            f"Temporal segment failure: scene={scene_id}, "
            f"segment={segment_id}, best_clip={best_clip_score:.4f}"
        )


class IdentityCumulativeDriftError(VGABaseError):
    """Raised when cumulative identity drift exceeds threshold. Triggers phase regen."""
    def __init__(self, stage_id: str, cumulative_drift: float, threshold: float):
        self.stage_id = stage_id
        self.cumulative_drift = cumulative_drift
        self.threshold = threshold
        super().__init__(
            f"Identity cumulative drift: stage={stage_id}, "
            f"drift={cumulative_drift:.4f} > threshold={threshold}"
        )


class IdentityReferenceCorruptionError(VGABaseError):
    """Raised when char_identity_ref is mutated after being frozen at S-07. CRITICAL."""
    def __init__(self, stage_id: str):
        self.stage_id = stage_id
        super().__init__(
            f"Identity reference corrupted at stage={stage_id}; "
            f"frozen after S-07 (RULE-95)"
        )


class AudioQualityError(VGABaseError):
    """Raised when SNR/peak fails after all re-mix attempts. RULE-99."""
    def __init__(self, scene_id: str, snr_db: float, peak_db: float):
        self.scene_id = scene_id
        self.snr_db = snr_db
        self.peak_db = peak_db
        super().__init__(
            f"Audio quality failure: scene={scene_id}, "
            f"snr={snr_db:.1f}dB, peak={peak_db:.1f}dBFS"
        )


class MissingPredecessorOutputError(VGABaseError):
    """Raised by execute_stage() when predecessor stage output is missing. CGRL-85."""
    def __init__(self, stage: str, predecessor: str):
        self.stage = stage
        self.predecessor = predecessor
        super().__init__(
            f"Stage {stage} requires predecessor {predecessor} output (not found in context)"
        )
```

---

## 21. `vga/temporal/__init__.py` — Package Anchor (NEW v17.0)

```python
"""
vga.temporal — TemporalEngine subsystem package.
Version: 17.0.0

Public API:
  TemporalEngine          — autoregressive SVI segment generation (Stage S-09)
  TemporalBufferManager   — 5-frame rolling buffer management
  SVIScheduler            — noise-aware LoRA scheduling per timestep
  MotionStateTracker      — optical flow motion estimation per segment
  TemporalRetryController — segment retry parameter adjustment

Architectural contracts:
  TEMPORAL_BUFFER_SIZE = 5 (STRICT; RULE-86)
  Multi-frame latent conditioning REQUIRED (RULE-87)
  Sequential segment generation REQUIRED (RULE-87, Invariant I4)
  SVIScheduler REQUIRED; static LoRA weight FORBIDDEN (RULE-86)
  SVI CFG ∈ [5.0, 6.0] (RULE-86)
"""

from vga.temporal.temporal_engine import TemporalEngine
from vga.temporal.temporal_buffer_manager import TemporalBufferManager, TemporalBuffer
from vga.temporal.svi_scheduler import SVIScheduler
from vga.temporal.motion_state_tracker import MotionStateTracker, MotionState
from vga.temporal.temporal_retry_controller import TemporalRetryController

__all__ = [
    "TemporalEngine",
    "TemporalBufferManager",
    "TemporalBuffer",
    "SVIScheduler",
    "MotionStateTracker",
    "MotionState",
    "TemporalRetryController",
]
```

---

## 22. `vga/temporal/temporal_buffer_manager.py` — Skeleton

```python
"""
temporal_buffer_manager.py
Manages the 5-frame rolling TemporalBuffer for SVI conditioning.

Invariants:
  buffer.frames.shape[0] == TEMPORAL_BUFFER_SIZE (5) at ALL times
  Frames are CPU-resident between SVI calls
  encode() returns Tensor of shape (5, C', H', W') — NEVER (C', H', W')
"""
from __future__ import annotations
import torch
from dataclasses import dataclass
from typing import List, Optional
from vga.core.exceptions import TemporalBufferError
from vga.config.settings import TEMPORAL_BUFFER_SIZE


@dataclass
class TemporalBuffer:
    frames: torch.Tensor          # MUST be shape (5, C, H, W)
    timestamps: List[float]
    motion_vector: Optional[torch.Tensor] = None
    scene_id: str = ""

    def __post_init__(self):
        if self.frames.shape[0] != TEMPORAL_BUFFER_SIZE:
            raise TemporalBufferError(
                scene_id=self.scene_id,
                frame_count=self.frames.shape[0],
                required=TEMPORAL_BUFFER_SIZE
            )


class TemporalBufferManager:
    def __init__(self, vae_encoder, tracer): ...
    def init(self, segment_1) -> TemporalBuffer: ...
    def update(self, buffer: TemporalBuffer, new_segment) -> TemporalBuffer: ...
    def encode(self, buffer: TemporalBuffer) -> torch.Tensor: ...
    def _extract_last_n_frames(self, video_tensor: torch.Tensor, n: int) -> torch.Tensor: ...
    def _extract_timestamps(self, segment, n: int) -> List[float]: ...
```

---

## 23. `vga/temporal/svi_scheduler.py` — Skeleton

```python
"""
svi_scheduler.py
Noise-aware LoRA weight scheduler for SVI temporal generation.

Contract:
  apply_lora(timestep) → 0.6 | 0.5 | 0.4
  assert_cfg_valid(cfg) → cfg or raises SVICFGViolationError
  Static LoRA weight (passing float to SVI) is FORBIDDEN (RULE-86)
  CFG outside [SVI_CFG_MIN, SVI_CFG_MAX] MUST raise (not clamp) (RULE-86)
"""
from vga.core.exceptions import SVICFGViolationError
from vga.config.settings import (
    LORA_WEIGHT_HIGH_NOISE, LORA_WEIGHT_MID_NOISE, LORA_WEIGHT_LOW_NOISE,
    HIGH_NOISE_FRACTION, MID_NOISE_FRACTION,
    SVI_CFG_MIN, SVI_CFG_MAX
)


class SVIScheduler:
    def __init__(self, total_steps: int, tracer): ...
    def apply_lora(self, timestep: int) -> float: ...
    def assert_cfg_valid(self, cfg: float) -> float: ...
    def get_lora_schedule(self) -> list: ...
```

---

## 24. `vga/temporal/motion_state_tracker.py` — Skeleton

```python
"""
motion_state_tracker.py
Estimates motion state from TemporalBuffer frames.

Contract:
  estimate(frames) called per-segment; result NOT cached across segments
  Returns MotionState with velocity/direction/magnitude
  Logs MotionStateRecord to storage
"""
from __future__ import annotations
import torch
from dataclasses import dataclass
from vga.config.settings import MOTION_STATIONARY_THRESHOLD


@dataclass
class MotionState:
    velocity_x: float
    velocity_y: float
    velocity_magnitude: float
    direction: str            # "stationary" | "forward" | "backward" | "left" | "right" | "diagonal"
    is_stationary: bool


class MotionStateTracker:
    def __init__(self, tracer, storage): ...
    def estimate(self, frames: torch.Tensor) -> MotionState: ...
    def log_state(self, segment_id: int, scene_id: str, state: MotionState, trace_id: str): ...
    def _compute_optical_flow(self, frame_a: torch.Tensor, frame_b: torch.Tensor) -> torch.Tensor: ...
    def _classify_direction(self, vx: float, vy: float, magnitude: float, stationary: bool) -> str: ...
```

---

## 25. `vga/temporal/temporal_engine.py` — Skeleton

```python
"""
temporal_engine.py
Stage S-09: Autoregressive cinematic segment generation.

Contracts:
  Segment_1 provided by VideoSegmentGenerator (S-08); NOT generated here
  Autoregressive loop: Segment[n] ALWAYS conditioned on TemporalBuffer from Segment[n-1]
  Sequential only: parallel generation is FORBIDDEN (RULE-87, Invariant I4)
  context.evolve() called after EVERY segment (RULE-87, Invariant I5)
  CLIPValidator.score() called per segment (RULE-89)
  IdentityStateTracker.update() called per segment (CGRL-96)
"""
from vga.temporal.temporal_buffer_manager import TemporalBufferManager, TemporalBuffer
from vga.temporal.svi_scheduler import SVIScheduler
from vga.temporal.motion_state_tracker import MotionStateTracker
from vga.temporal.temporal_retry_controller import TemporalRetryController
from vga.validation.clip_validator import CLIPValidator
from vga.identity.identity_state_tracker import IdentityStateTracker
from vga.state.immutable_context import ImmutableContext
from vga.config.settings import (
    TEMPORAL_BUFFER_SIZE, TEMPORAL_MAX_RETRIES_PER_SEGMENT,
    CLIP_IDENTITY_THRESHOLD, SVI_CFG_MIN, SVI_CFG_MAX,
    STEPS_CRITICAL, STEPS_STANDARD, SEGMENT_CONTINUITY_MIN
)
from vga.core.exceptions import (
    TemporalBufferError, TemporalSegmentFailureError,
    SVICFGViolationError, AutoregressiveViolationError
)
import torch


class TemporalEngine:
    def __init__(
        self, svi_wrapper, buffer_manager: TemporalBufferManager,
        svi_scheduler: SVIScheduler, motion_tracker: MotionStateTracker,
        retry_controller: TemporalRetryController, clip_validator: CLIPValidator,
        identity_tracker: IdentityStateTracker, continuity_validator, tracer
    ): ...

    def generate_scene(
        self, segment_plans: list, segment_1,
        context: ImmutableContext, char_identity_ref: torch.Tensor,
        trace_id: str
    ) -> tuple: ...     # returns (List[VideoSegment], ImmutableContext)

    def _generate_segment_with_retry(
        self, n: int, segment_plan, buffer: TemporalBuffer,
        motion_state, char_identity_ref: torch.Tensor,
        context: ImmutableContext, trace_id: str
    ): ...

    def _build_temporal_prompt(self, segment_plan, motion_state, context: ImmutableContext) -> str: ...
```

---

## 26. `vga/identity/identity_state_tracker.py` — Skeleton

```python
"""
identity_state_tracker.py
Cross-phase cumulative identity drift tracker.

Contracts:
  update() called after EVERY CLIPValidator.score() in any phase (CGRL-96)
  Raises IdentityCumulativeDriftError when drift_score > IDENTITY_CUMULATIVE_DRIFT_THRESHOLD
  reset() called by caller after successful phase regeneration
  char_identity_ref (embedding_vector) NEVER changes here — only delta is computed
"""
from __future__ import annotations
import torch
from vga.models.schemas import IdentityStateRecord
from vga.core.exceptions import IdentityCumulativeDriftError
from vga.config.settings import IDENTITY_CUMULATIVE_DRIFT_THRESHOLD, SCHEMA_VERSION


class IdentityStateTracker:
    def __init__(self, clip_encoder, tracer, storage): ...
    def update(self, char_identity_ref: torch.Tensor, new_frame, stage_id: str) -> dict: ...
    def reset(self) -> None: ...

    @property
    def current_state(self) -> dict:
        return {
            "embedding_vector": None,    # never stored here; only in frozen context
            "drift_score": self._drift_score,
            "history": list(self._history)
        }
```

---

## 27. `vga/validation/composition_validator.py` — Skeleton

```python
"""
composition_validator.py
Validates CompositionPlanSchema completeness.

Contracts:
  All 6 fields mandatory: camera_angle, camera_motion, character_positions,
    focus_subject, lighting_style, motion_vector
  assert_in_context() used as gate before S-05, S-08, S-09 (RULE-88)
  STATELESS: validates and returns; never creates or modifies plans
"""
from vga.models.schemas import CompositionPlanSchema
from vga.core.exceptions import CompositionPlanValidationError


class CompositionPlanValidator:
    REQUIRED_FIELDS = [
        "camera_angle", "camera_motion", "character_positions",
        "focus_subject", "lighting_style", "motion_vector"
    ]

    def validate(self, plan: CompositionPlanSchema) -> bool: ...
    def assert_in_context(self, context) -> None: ...
```

---

## 28. `vga/validation/audio_quality_validator.py` — Skeleton

```python
"""
audio_quality_validator.py
Validates and normalizes mixed audio. RULE-99.

Contracts:
  validate() returns AudioQualityRecord; does NOT raise on failure
  normalize() returns normalized AudioSegment; MUST be called before storage write
  compute_snr() computes dialogue SNR in dB (min: MIN_SNR_DB = 10.0)
  compute_peak_db() computes peak level in dBFS (max: MAX_PEAK_DBFS = 0.0)
  STATELESS: no mutable internal state
  CALLED BY: audio_mixing_agent.py ONLY
"""
from pydub import AudioSegment
import math
from vga.models.schemas import AudioQualityRecord
from vga.config.settings import MIN_SNR_DB, MAX_PEAK_DBFS, HEADROOM_DB, SCHEMA_VERSION


class AudioQualityValidator:
    def __init__(self, tracer, storage): ...
    def validate(self, mixed: AudioSegment, dialogue: AudioSegment,
                 scene_id: str, job_id: str) -> AudioQualityRecord: ...
    def normalize(self, mixed: AudioSegment, target_peak_db: float = -1.0) -> AudioSegment: ...
    def compute_snr(self, mixed: AudioSegment, dialogue: AudioSegment) -> float: ...
    def compute_peak_db(self, audio: AudioSegment) -> float: ...
```

---

## 29. `vga/validation/cross_modal_alignment_validator.py` — Skeleton

```python
"""
cross_modal_alignment_validator.py
Validates video ↔ audio duration alignment per segment. RULE-96.

Contracts:
  validate_alignment() returns CrossModalAlignmentReport; does NOT raise on tolerance violation
  Caller (audio_mixing_agent.py) decides action based on report
  STATELESS: no mutable internal state
  CALLED BY: audio_mixing_agent.py ONLY
"""
from vga.models.schemas import CrossModalAlignmentRecord, CrossModalAlignmentReport
from vga.config.settings import TIMING_TOLERANCE_S, SCHEMA_VERSION


class CrossModalAlignmentValidator:
    def __init__(self, tracer, storage): ...
    def validate_alignment(
        self, video_segments: list, audio_segments: list, scene_id: str
    ) -> CrossModalAlignmentReport: ...
```

---

## 30. `vga/agents/scene_composition_agent.py` — Skeleton

```python
"""
scene_composition_agent.py
Stage S-04: Translates scene narrative data into CompositionPlan.

Contracts:
  All 6 CompositionPlan fields are mandatory (RULE-88)
  Retries up to COMPOSITION_MAX_RETRIES on schema failure (CGRL-99)
  Writes composition_plan_{scene_id}.json to storage
  Raises CompositionPlanValidationError if all retries fail
  Called via execute_stage() contract (CGRL-85)
"""
from vga.agents.base_agent import LLMAgent
from vga.models.schemas import CompositionPlanSchema
from vga.validation.composition_validator import CompositionPlanValidator
from vga.core.exceptions import CompositionPlanValidationError
from vga.config.settings import COMPOSITION_MAX_RETRIES, SCHEMA_VERSION
import json


class SceneCompositionAgent(LLMAgent):
    def __init__(
        self, model_wrapper, prompt_builder, tracer,
        system_guard, authority_manager, composition_validator: CompositionPlanValidator
    ): ...

    def compose(self, scene_data: dict, trace_id: str) -> CompositionPlanSchema: ...
```

---

## 31. `vga/agents/video_segment_generator.py` — Skeleton (NEW v17.0)

```python
"""
video_segment_generator.py
Stage S-08: Wan2.2 Segment_1 generation + TemporalBuffer initialization.

Contracts:
  Generates ONLY Segment_1; Segments 2..N owned by TemporalEngine
  CompositionPlan MUST be in context before generation (RULE-88)
  CLIPValidator called on Segment_1 keyframe (RULE-89)
  TemporalBufferManager.init() called immediately after Segment_1
  Buffer frame count asserted == TEMPORAL_BUFFER_SIZE (RULE-86)
  Raises TemporalBufferError if buffer init fails
"""
from vga.temporal.temporal_buffer_manager import TemporalBufferManager
from vga.validation.clip_validator import CLIPValidator
from vga.validation.composition_validator import CompositionPlanValidator
from vga.state.immutable_context import ImmutableContext
from vga.core.exceptions import TemporalBufferError
from vga.config.settings import TEMPORAL_BUFFER_SIZE, CLIP_IDENTITY_THRESHOLD
import torch


class VideoSegmentGenerator:
    def __init__(
        self, wan_wrapper, buffer_manager: TemporalBufferManager,
        clip_validator: CLIPValidator, composition_validator: CompositionPlanValidator,
        storage, tracer
    ): ...

    def generate(
        self, refined_image_path: str, composition_plan,
        context: ImmutableContext, char_identity_ref: torch.Tensor,
        trace_id: str
    ) -> tuple: ...     # returns (VideoSegment, TemporalBuffer)
```

---

## 32. `vga/core/master_orchestrator.py` — execute_stage Anchor (v17.0)

```python
"""
master_orchestrator.py
SYSTEM DIRECTIVE v17: execute_stage() is the MANDATORY wrapper for all pipeline stages.
No stage may call agent.run() directly outside this function.
"""
from vga.state.immutable_context import ImmutableContext
from vga.runtime.system_guard import SystemGuard
from vga.core.exceptions import MissingPredecessorOutputError
from typing import Any, Tuple


def execute_stage(stage: Any, input_data: Any, context: ImmutableContext) -> Tuple[Any, ImmutableContext]:
    """
    SYSTEM DIRECTIVE v17 — Mandatory execution contract for all pipeline stages.

    Steps:
      1. SystemGuard.execute(stage)
      2. authority_manager.validate(stage.authority_level, stage.action_name)
      3. _validate_previous_output(stage, context)
      4. hrg_controller.require_approval() if stage.requires_hrg
      5. output = stage.run(input_data, context)
      6. _validate_output(stage, output)
      7. context = context.evolve(output) — MANDATORY
      8. return output, context

    Returns: (output, updated_context)
    Raises: MissingPredecessorOutputError, HRG-related exceptions
    """
    ...


def _validate_previous_output(stage: Any, context: ImmutableContext) -> None: ...
def _validate_output(stage: Any, output: Any) -> None: ...
def _prepare_hrg_display(stage: Any, context: ImmutableContext) -> dict: ...


class MasterOrchestrator:
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

    def __init__(self, stage_registry: dict, hrg_controller, tracer): ...
    def run_scene(self, scene_id: str, job: Any) -> Any: ...
    def _execute_stage(self, stage_id: str, job: Any) -> Any: ...
```

---

## 33. `vga/state/immutable_context.py` — v17.0 Anchor (5-dimensional)

```python
"""
immutable_context.py
5-dimensional frozen pipeline context.
v17.0: adds temporal_state, camera_state, lighting_state.

Contracts:
  frozen=True: NO attribute mutation; evolve() always returns new instance
  has_output(stage_id) used by execute_stage() to validate predecessor
  All 5 dimensions MUST be initialized by context_factory before pipeline starts
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Optional, Dict, List


@dataclass
class TemporalState:
    buffer: Optional[Any] = None    # TemporalBuffer once initialized at S-08
    segment_index: int = 0


@dataclass
class CameraState:
    angle: Optional[str] = None
    motion: Optional[str] = None


@dataclass
class LightingState:
    style: Optional[str] = None


@dataclass
class MotionStateContext:
    velocity_x: float = 0.0
    velocity_y: float = 0.0
    velocity_magnitude: float = 0.0
    direction: str = "stationary"
    is_stationary: bool = True


@dataclass
class IdentityStateContext:
    embedding_vector: Optional[Any] = None   # torch.Tensor; frozen at S-07
    drift_score: float = 0.0
    history: List[float] = field(default_factory=list)


@dataclass(frozen=True)
class ImmutableContext:
    job_id: str
    scene_id: str
    identity_state: IdentityStateContext
    motion_state: MotionStateContext
    camera_state: CameraState
    lighting_state: LightingState
    temporal_state: TemporalState          # NEW v17.0
    adaptive_params: Dict[str, Any]
    last_output: Optional[Any] = None

    def evolve(self, updates: dict) -> "ImmutableContext":
        """Return new ImmutableContext with applied updates. Thread-safe."""
        current = {
            field.name: getattr(self, field.name)
            for field in self.__dataclass_fields__.values()
        }
        current.update(updates)
        return ImmutableContext(**current)

    def has_output(self, stage_id: str) -> bool:
        """CGRL-85: check predecessor output present in context."""
        return (
            self.last_output is not None and
            getattr(self.last_output, 'stage_id', None) == stage_id
        )
```

---

## 34. `vga/ui/components/hrg_panels/hrg_4_composition.py` — Skeleton (NEW v17.0)

```python
"""
hrg_4_composition.py
HRG-4 Checkpoint: Scene Composition Review.
Renders all 6 CompositionPlan fields as labeled, editable widgets.
Gates S-05 (BaseImageAgent) on approval — RULE-88.

Actions: approve | edit any field | trigger recompose
"""
import streamlit as st
import httpx
from vga.models.schemas import CompositionPlanSchema, CompositionPlanUpdateRequest


def render_hrg_4_composition_panel(job_id: str, scene_id: str, api_base: str = "http://localhost:8000"):
    """
    Renders HRG-4 review panel.
    Blocks until user approves or edits and recomposes.
    """
    st.header("HRG-4: Scene Composition Review")
    st.caption("Review the visual composition plan before image generation begins.")

    # Fetch current plan
    resp = httpx.get(f"{api_base}/jobs/{job_id}/composition/{scene_id}")
    plan = CompositionPlanSchema(**resp.json())

    # Display all 6 fields
    col1, col2 = st.columns(2)
    with col1:
        camera_angle = st.text_input("Camera Angle", value=plan.camera_angle)
        camera_motion = st.text_input("Camera Motion", value=plan.camera_motion)
        lighting_style = st.text_input("Lighting Style", value=plan.lighting_style)
    with col2:
        focus_subject = st.text_input("Focus Subject", value=plan.focus_subject)
        motion_vector = st.text_input("Motion Vector", value=plan.motion_vector)

    st.write("**Character Positions:**")
    for pos in plan.character_positions:
        st.write(f"  • {pos.get('character_id', 'Unknown')}: {pos.get('position', '?')} facing {pos.get('facing', '?')}")

    # Actions
    col_approve, col_edit, col_recompose = st.columns(3)
    with col_approve:
        if st.button("✅ Approve Composition", type="primary"):
            httpx.post(f"{api_base}/jobs/{job_id}/hrg/HRG-4", json={"action": "approved", "user": "human"})
            st.success("Composition approved. S-05 image generation will begin.")

    with col_edit:
        if st.button("✏️ Save Edits"):
            update = CompositionPlanUpdateRequest(
                scene_id=scene_id,
                camera_angle=camera_angle,
                camera_motion=camera_motion,
                lighting_style=lighting_style,
                focus_subject=focus_subject,
                motion_vector=motion_vector
            )
            httpx.patch(f"{api_base}/jobs/{job_id}/composition/{scene_id}", json=update.model_dump())
            st.info("Composition plan updated.")

    with col_recompose:
        if st.button("🔄 Trigger Recompose"):
            httpx.post(
                f"{api_base}/jobs/{job_id}/hrg/HRG-4",
                json={"action": "trigger_regeneration", "user": "human", "payload": {"regen_stage": "S-04"}}
            )
            st.warning("Recomposition triggered.")
```
