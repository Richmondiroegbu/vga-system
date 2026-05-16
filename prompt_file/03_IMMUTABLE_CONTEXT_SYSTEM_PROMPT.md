# Prompt 03: Immutable Context System
**Category:** Core Foundation  
**Files to implement:**
- `vga/state/immutable_context.py`
- `vga/state/context_factory.py`
- `vga/state/context_history.py`
- `vga/state/context_diff.py`
**Spec References:**
- `01_VGA_SRD_v17.2.md` §3.70 (Context Propagation Contract, FR-950–FR-954)
- `01_VGA_SRD_v17.2.md` §5.26 (Identity State Constraints)
- `04_VGA_Pipeline_Execution_Flow_v17.2.md` (context.evolve() contract)  
**Dependencies:** Prompt 02 complete (settings.py and schemas.py exist)  
**Build Order:** Step 12.1.1 → 12.1.2

---

## Context

The `ImmutableContext` is the **central nervous system** of the VGA pipeline. Every stage consumes it and produces an evolved version. It is:
- **Frozen** — all fields are immutable after creation
- **5-dimensional** — tracks identity, motion, camera, lighting, temporal state
- **Forward-propagating** — `evolve()` creates a new instance; the old one is untouched
- **Type-enforced** — `isinstance(context, ImmutableContext)` is checked at `execute_stage()` entry

**RULE-108**: Dict-based context is FORBIDDEN. Any `context["key"]` pattern must be rejected at runtime.

---

## vga/state/immutable_context.py

```python
"""
ImmutableContext — The 5-dimensional pipeline state container.
RULE-108: Dict-based context FORBIDDEN. Only ImmutableContext is valid.
FR-950: Every stage MUST call context.evolve() after completion.
FR-951: Context tracks ALL five dimensions.
FR-952: context.evolve() is the ONLY permitted update mechanism.
Spec: VGA SRD v17.2 §3.70, §5.23
"""
from __future__ import annotations
from dataclasses import dataclass, field, replace
from typing import Optional, List, Any
import numpy as np
from vga.models.schemas import (
    CompositionPlanSchema, TemporalState, CameraState, LightingState
)
from vga.models.enums import PipelineStageID

@dataclass(frozen=True)
class IdentityState:
    """
    Cross-phase identity persistence tracker.
    FR-953: tracks embedding_vector, drift_score, history.
    FR-954: cumulative drift > threshold triggers full regeneration.
    """
    embedding_vector: Optional[np.ndarray] = None   # char_identity_ref (frozen at S-05)
    drift_score: float = 0.0                         # latest per-stage drift
    cumulative_drift: float = 0.0                    # sum of all stage drifts
    history: tuple[float, ...] = field(default_factory=tuple)  # immutable history
    is_frozen: bool = False   # True once char_identity_ref set at S-05
    
    def update_drift(self, new_drift: float) -> "IdentityState":
        """Return new IdentityState with updated drift (append-only). RULE-95."""
        new_history = self.history + (new_drift,)
        new_cumulative = self.cumulative_drift + new_drift
        return IdentityState(
            embedding_vector=self.embedding_vector,
            drift_score=new_drift,
            cumulative_drift=new_cumulative,
            history=new_history,
            is_frozen=self.is_frozen
        )
    
    def freeze(self, embedding: np.ndarray) -> "IdentityState":
        """
        Freeze the identity reference embedding (called once at S-05).
        RULE-95: embedding frozen here — never recomputed downstream.
        """
        if self.is_frozen:
            from vga.core.exceptions import IdentityReferenceCorruptionError
            raise IdentityReferenceCorruptionError(
                "char_identity_ref already frozen — cannot refreeze (RULE-95)"
            )
        return IdentityState(
            embedding_vector=embedding.copy(),  # defensive copy
            drift_score=0.0,
            cumulative_drift=0.0,
            history=(),
            is_frozen=True
        )

@dataclass(frozen=True)
class ImmutableContext:
    """
    5-dimensional immutable pipeline context.
    
    CRITICAL: This is the ONLY valid context type in VGA v17.2.
    Dict-based context raises ImmutableContextViolationError at execute_stage() entry.
    
    Dimensions:
      1. identity_state   — character identity embedding + drift tracking
      2. motion_state     — velocity, direction, magnitude from TemporalBuffer  
      3. camera_state     — angle and motion from CompositionPlan
      4. lighting_state   — style from CompositionPlan
      5. temporal_state   — TemporalBuffer reference + segment index
    
    Schema: All fields are typed; none are bare dict.
    """
    # Dimension 1: Identity
    identity_state: IdentityState = field(default_factory=IdentityState)
    
    # Dimension 2: Motion
    motion_state: Optional[Any] = None  # MotionState dataclass (temporal/)
    
    # Dimension 3: Camera
    camera_state: CameraState = field(default_factory=CameraState)
    
    # Dimension 4: Lighting
    lighting_state: LightingState = field(default_factory=LightingState)
    
    # Dimension 5: Temporal
    temporal_state: TemporalState = field(default_factory=TemporalState)
    
    # Composition Plan (cross-cutting — required at all image/video stages)
    composition_plan: Optional[CompositionPlanSchema] = None
    
    # Stage tracking
    completed_stages: tuple[str, ...] = field(default_factory=tuple)
    current_stage: Optional[str] = None
    job_id: Optional[str] = None
    scene_id: Optional[str] = None
    
    def evolve(self, **updates) -> "ImmutableContext":
        """
        Create a new ImmutableContext with specified fields updated.
        This is the ONLY permitted mutation mechanism (FR-952).
        
        All 5 state dimensions should be updated at every stage transition.
        """
        return replace(self, **updates)
    
    def with_stage_completed(self, stage_id: str) -> "ImmutableContext":
        """Mark a stage as completed and return new context."""
        return replace(
            self,
            completed_stages=self.completed_stages + (stage_id,),
            current_stage=stage_id,
        )
    
    def has_output(self, stage_id: str) -> bool:
        """Check if a predecessor stage has completed. For RULE-90 enforcement."""
        return stage_id in self.completed_stages
    
    def assert_composition_plan(self) -> None:
        """Assert CompositionPlan exists. Raises CompositionPlanValidationError. RULE-88."""
        if self.composition_plan is None:
            from vga.core.exceptions import CompositionPlanValidationError
            raise CompositionPlanValidationError(
                "CompositionPlan REQUIRED before image/video generation (RULE-88). "
                "SceneCompositionAgent (S-04) must run first."
            )
    
    def assert_identity_frozen(self) -> None:
        """Assert char_identity_ref is frozen. Required for S-06 onwards."""
        if not self.identity_state.is_frozen:
            from vga.core.exceptions import IdentityReferenceCorruptionError
            raise IdentityReferenceCorruptionError(
                "char_identity_ref not frozen. BaseImageAgent (S-05) must run first."
            )
    
    @classmethod
    def _reject_dict(cls) -> None:
        """Called when dict context is detected. RULE-108."""
        from vga.core.exceptions import ImmutableContextViolationError
        raise ImmutableContextViolationError(
            "Dict-based context FORBIDDEN (RULE-108). Use ImmutableContext."
        )
```

---

## vga/state/context_factory.py

```python
"""
ContextFactory — Creates the initial ImmutableContext at S-02.
Called once per pipeline run at scene plan generation.
"""
from vga.state.immutable_context import ImmutableContext, IdentityState
from vga.models.schemas import TemporalState, CameraState, LightingState

class ContextFactory:
    """Creates well-formed initial ImmutableContext instances."""
    
    @staticmethod
    def create_initial(job_id: str, scene_id: str) -> ImmutableContext:
        """
        Create the initial 5-dimensional context at S-02 (ScenePlanner).
        All dimensions initialized to empty/default state.
        """
        return ImmutableContext(
            identity_state=IdentityState(),
            motion_state=None,
            camera_state=CameraState(),
            lighting_state=LightingState(),
            temporal_state=TemporalState(buffer=None, segment_index=0),
            composition_plan=None,
            completed_stages=(),
            current_stage=None,
            job_id=job_id,
            scene_id=scene_id,
        )
    
    @staticmethod
    def validate(context: object) -> ImmutableContext:
        """
        Validate that the context is a proper ImmutableContext.
        Rejects dict contexts with ImmutableContextViolationError (RULE-108).
        """
        if isinstance(context, dict):
            ImmutableContext._reject_dict()
        if not isinstance(context, ImmutableContext):
            raise TypeError(f"Expected ImmutableContext, got {type(context).__name__}")
        return context
```

---

## vga/state/context_history.py

```python
"""
ContextHistory — Append-only log of context snapshots per stage.
Used for debugging, audit trail, and rollback analysis.
"""
# Implement: 
# - append(stage_id, context) — store context snapshot
# - get_at_stage(stage_id) → ImmutableContext | None
# - get_identity_drift_series() → List[float]  
# - to_json() → str (for /workspace/state/ persistence)
```

---

## Acceptance Criteria

- [ ] `ImmutableContext` is a frozen dataclass — `ctx.identity_state = x` raises `FrozenInstanceError`
- [ ] `ctx.evolve(job_id="new")` returns a new instance without modifying `ctx`
- [ ] `ctx.assert_composition_plan()` raises `CompositionPlanValidationError` when plan is None
- [ ] `ContextFactory.validate({"key": "val"})` raises `ImmutableContextViolationError`
- [ ] `identity_state.freeze(embedding).freeze(other)` raises `IdentityReferenceCorruptionError` (double-freeze protection)
- [ ] `identity_state.update_drift(0.01).history` contains the drift value
- [ ] 5 context dimensions are present and typed correctly
