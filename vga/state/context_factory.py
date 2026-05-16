"""
ContextFactory — creates and validates ImmutableContext instances.
Entry point for all new pipeline contexts.
Spec: VGA Data Contracts v17.2 §3.2; FR-950–FR-951
"""
from __future__ import annotations

from vga.core.exceptions import ImmutableContextViolationError
from vga.state.immutable_context import (
    CameraState,
    ImmutableContext,
    IdentityState,
    LightingState,
    MotionState,
    TemporalState,
)


class ContextFactory:
    """Creates ImmutableContext with zeroed 5-dimensional initial state."""

    @staticmethod
    def create_initial(job_id: str, scene_id: str) -> ImmutableContext:
        """Create a blank 5-dimensional context for a new scene.

        All state dimensions start at their zero/default values.
        Called by ScenePlanner (S-02) when initializing context for a scene.
        """
        if not job_id or not job_id.strip():
            raise ValueError("job_id must not be empty")
        if not scene_id or not scene_id.strip():
            raise ValueError("scene_id must not be empty")

        return ImmutableContext(
            identity_state=IdentityState(),
            motion_state=MotionState(),
            camera_state=CameraState(),
            lighting_state=LightingState(),
            temporal_state=TemporalState(),
            job_id=job_id,
            scene_id=scene_id,
            current_stage=None,
            composition_plan=None,
            completed_stages=(),
            stage_outputs={},
        )

    @staticmethod
    def validate(context: object) -> ImmutableContext:
        """Assert that an object is a proper ImmutableContext.

        Raises ImmutableContextViolationError for dicts or other types (RULE-108).
        """
        if isinstance(context, dict):
            raise ImmutableContextViolationError(
                "A dict was used where ImmutableContext is required. "
                "Use ContextFactory.create_initial() or context.evolve(). RULE-108."
            )
        if not isinstance(context, ImmutableContext):
            raise ImmutableContextViolationError(
                f"Expected ImmutableContext, got {type(context).__name__}. RULE-108."
            )
        return context
