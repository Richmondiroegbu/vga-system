"""
ImmutableContext — 5-dimensional frozen pipeline state.
The ONLY permitted context type. Dict-based context raises ImmutableContextViolationError.
Spec: VGA Data Contracts v17.2 §3; RULE-108, FR-950–FR-954
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from vga.core.exceptions import (
    CompositionPlanValidationError,
    ContextEvolutionError,
    ImmutableContextViolationError,
    IdentityReferenceCorruptionError,
)
from vga.models.schemas import CompositionPlanSchema


# ─── Sub-State Dataclasses ────────────────────────────────────────────────────

@dataclass
class IdentityState:
    """Character identity state. Frozen at S-07 — never recomputed downstream. RULE-95."""

    embedding_vector: Optional[List[float]] = None   # char_identity_ref (frozen after S-07)
    drift_score: float = 0.0                          # single-stage drift
    cumulative_drift: float = 0.0                     # cross-phase accumulation
    history: Tuple[float, ...] = ()                   # append-only drift log
    is_frozen: bool = False                           # True after ImageRefinementAgent (S-07)

    def update_drift(self, new_drift: float) -> "IdentityState":
        """Return updated IdentityState with drift appended. Does NOT mutate in place."""
        return dataclasses.replace(
            self,
            drift_score=new_drift,
            cumulative_drift=self.cumulative_drift + new_drift,
            history=self.history + (new_drift,),
        )

    def freeze(self, embedding_vector: List[float]) -> "IdentityState":
        """Freeze the identity reference embedding. Raises if already frozen (RULE-95)."""
        if self.is_frozen:
            raise IdentityReferenceCorruptionError(
                "char_identity_ref is already frozen — recomputing it mid-pipeline violates RULE-95"
            )
        return dataclasses.replace(self, embedding_vector=embedding_vector, is_frozen=True)


@dataclass
class MotionState:
    """Per-segment motion estimation from TemporalEngine."""

    velocity_vector: Tuple[float, float] = (0.0, 0.0)   # (dx, dy) mean optical flow
    direction: str = "stationary"                          # forward/backward/left/right/stationary
    magnitude: float = 0.0                                 # scalar speed


@dataclass
class CameraState:
    """Camera framing state from SceneCompositionAgent."""

    angle: str = "eye level"      # camera_angle from CompositionPlan
    motion: str = "static"        # camera_motion from CompositionPlan


@dataclass
class LightingState:
    """Lighting style from SceneCompositionAgent."""

    style: str = "natural"        # lighting_style from CompositionPlan


@dataclass
class TemporalState:
    """Temporal generation progress tracking."""

    segment_index: int = 0          # current segment being generated
    total_segments: int = 0         # total segments in this scene
    buffer_initialized: bool = False


# ─── ImmutableContext ─────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ImmutableContext:
    """5-dimensional frozen pipeline context.

    Created once per scene by ContextFactory.create_initial().
    Updated ONLY via evolve() — each call returns a new instance.
    Dict-based context is FORBIDDEN (RULE-108).

    Dimensions:
        1. identity_state   — character identity embedding + drift tracking
        2. motion_state     — optical flow velocity from TemporalEngine
        3. camera_state     — camera angle/motion from CompositionPlan
        4. lighting_state   — lighting style from CompositionPlan
        5. temporal_state   — segment index and buffer status
    """

    # === 5 Mandatory Dimensions ===
    identity_state: IdentityState
    motion_state: MotionState
    camera_state: CameraState
    lighting_state: LightingState
    temporal_state: TemporalState

    # === Pipeline Metadata ===
    job_id: str
    scene_id: str
    current_stage: Optional[str] = None
    composition_plan: Optional[CompositionPlanSchema] = None
    completed_stages: Tuple[str, ...] = ()

    # === Stage Outputs (for predecessor checking, RULE-90) ===
    stage_outputs: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Enforce RULE-108: reject if someone passes a dict as context."""
        self._reject_dict()

    def _reject_dict(self) -> None:
        """Guard against dict-based context usage."""
        for dim_name in ("identity_state", "motion_state", "camera_state", "lighting_state", "temporal_state"):
            val = getattr(self, dim_name)
            if isinstance(val, dict):
                raise ImmutableContextViolationError(
                    f"ImmutableContext.{dim_name} must be a typed dataclass, not a dict. "
                    f"RULE-108 violation."
                )

    def evolve(self, **kwargs: Any) -> "ImmutableContext":
        """Return a new ImmutableContext with specified fields updated.

        This is the ONLY permitted way to 'mutate' the context.
        All unspecified fields are carried forward unchanged.
        """
        if not kwargs:
            raise ContextEvolutionError("evolve() called with no changes — this is likely a bug")
        return dataclasses.replace(self, **kwargs)

    def with_stage_completed(self, stage_id: str, output: Any = None) -> "ImmutableContext":
        """Return new context marking stage_id as completed, optionally storing its output."""
        new_stages = self.completed_stages + (stage_id,)
        new_outputs = {**self.stage_outputs}
        if output is not None:
            new_outputs[stage_id] = output
        return dataclasses.replace(
            self,
            completed_stages=new_stages,
            current_stage=stage_id,
            stage_outputs=new_outputs,
        )

    def has_output(self, stage_id: str) -> bool:
        """Check whether a predecessor stage has produced its output (RULE-90)."""
        return stage_id in self.stage_outputs

    def get_output(self, stage_id: str) -> Any:
        """Retrieve a predecessor stage output. Returns None if not present."""
        return self.stage_outputs.get(stage_id)

    def assert_composition_plan(self) -> None:
        """Raise CompositionPlanValidationError if CompositionPlan is not present. RULE-88."""
        if self.composition_plan is None:
            raise CompositionPlanValidationError(
                f"CompositionPlan is required before image/video generation but is missing "
                f"in context for scene {self.scene_id}. RULE-88 violation."
            )

    def assert_identity_frozen(self) -> None:
        """Raise IdentityReferenceCorruptionError if identity has not been frozen yet."""
        if not self.identity_state.is_frozen:
            raise IdentityReferenceCorruptionError(
                f"char_identity_ref must be frozen before video generation "
                f"(scene {self.scene_id}) — freeze at S-07. RULE-95 violation."
            )
        if self.identity_state.embedding_vector is None:
            raise IdentityReferenceCorruptionError(
                f"char_identity_ref is marked frozen but embedding_vector is None "
                f"(scene {self.scene_id}). RULE-95 violation."
            )
