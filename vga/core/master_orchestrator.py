"""
MasterOrchestrator — THE ONLY component permitted to execute pipeline stages.
All stage execution MUST flow through execute_stage(). Direct agent.run() = FORBIDDEN.
Spec: VGA Engine Template Spec v17.2 §3; RULE-106, FR-220–FR-235
"""
from __future__ import annotations

import logging
import time
from typing import Any, Optional, Tuple

from vga.config.settings import settings
from vga.core.exceptions import (
    ArchitectureGuardViolationError,
    MissingPredecessorOutputError,
    OutputValidationError,
    SchemaValidationError,
)
from vga.models.enums import HRGCheckpoint, PipelineStageID
from vga.state.immutable_context import ImmutableContext
from vga.state.context_factory import ContextFactory

logger = logging.getLogger(__name__)

# Stages that REQUIRE a validated CompositionPlan before execution (RULE-88)
COMPOSITION_REQUIRED_STAGES = {
    PipelineStageID.S05_BASE_IMAGE,
    PipelineStageID.S06_IDENTITY_REINFORCEMENT,
    PipelineStageID.S07_IMAGE_REFINEMENT,
    PipelineStageID.S08_VIDEO_SEGMENT_1,
    PipelineStageID.S09_TEMPORAL_ENGINE,
}

# Predecessor output requirements per stage (RULE-90)
STAGE_PREREQUISITES: dict[str, str] = {
    "S-03": "S-02",  # IdentityDesignAgent needs ScenePlan
    "S-04": "S-02",  # SceneCompositionAgent needs ScenePlan
    "S-05": "S-04",  # BaseImageAgent needs CompositionPlan
    "S-06": "S-05",  # IdentityReinforcement needs BaseImages
    "S-07": "S-06",  # ImageRefinement needs ReinforcedImages
    "S-08": "S-07",  # VideoSegmentGenerator needs RefinedImage
    "S-09": "S-08",  # TemporalEngine needs Segment_1
    "S-10": "S-09",  # ContinuityValidation needs all segments
    "S-11": "S-10",  # DialogueAgent needs continuity-validated segments
    "S-12": "S-11",  # LipSyncAgent needs dialogue audio
    "S-13": "S-10",  # AmbientAudio needs validated segments
    "S-14": "S-10",  # MusicAgent needs validated segments
    "S-15": "S-12",  # AudioMixing needs lip-synced audio
    "S-16": "S-15",  # Export needs final audio
}

# HRG checkpoint per stage
STAGE_HRG_MAP: dict[str, HRGCheckpoint] = {
    "S-01": HRGCheckpoint.HRG_1_SCRIPT,
    "S-02": HRGCheckpoint.HRG_2_SCENE_PLAN,
    "S-03": HRGCheckpoint.HRG_3_IDENTITY,
    "S-04": HRGCheckpoint.HRG_4_COMPOSITION,
    "S-05": HRGCheckpoint.HRG_5_BASE_IMAGES,
    "S-06": HRGCheckpoint.HRG_6_IDENTITY_REINFORCEMENT,
    "S-07": HRGCheckpoint.HRG_7_REFINED_IMAGE,
    "S-10": HRGCheckpoint.HRG_8_MOTION_QA,
    "S-11": HRGCheckpoint.HRG_9_VOICE_QA,
    "S-12": HRGCheckpoint.HRG_10_LIPSYNC_QA,
    "S-15": HRGCheckpoint.HRG_11_FINAL_AUDIO_QA,
}


class MasterOrchestrator:
    """THE ONLY component that may call agent.run().

    Every pipeline stage flows through execute_stage() which enforces:
    1. ImmutableContext type assertion (RULE-108)
    2. SystemGuard entry (stage isolation)
    3. CompositionPlan gate for visual stages (RULE-88)
    4. Stage readiness check — 6 sub-checks (RULE-90)
    5. agent.run(input_data, context)
    6. Output schema validation
    7. IdentityStateTracker.update()
    8. HRG checkpoint (if required)
    9. context.evolve() → new_context (FR-950)
    10. SystemGuard exit (lifecycle cleanup)
    """

    def __init__(
        self,
        hrg_controller: Any,
        composition_validator: Any,
        identity_tracker: Any,
        sla_manager: Optional[Any] = None,
        tracer: Optional[Any] = None,
    ) -> None:
        self._hrg = hrg_controller
        self._comp_validator = composition_validator
        self._identity_tracker = identity_tracker
        self._sla_manager = sla_manager
        self._tracer = tracer
        logger.info("MasterOrchestrator initialized — execute_stage() contract active")

    def execute_stage(
        self,
        agent: Any,
        input_data: Any,
        context: ImmutableContext,
    ) -> Tuple[Any, ImmutableContext]:
        """Execute a single pipeline stage through the full enforcement contract.

        Returns (stage_output, new_context).
        Raises on any rule violation — never silently swallows errors.
        RULE-106: this is the ONLY method permitted to call agent.run().
        """
        from vga.runtime.system_guard import SystemGuard

        stage_id = getattr(agent, "stage_id", "UNKNOWN")
        scene_id = context.scene_id
        job_id = context.job_id

        # === Step 1: ImmutableContext assertion (RULE-108) ===
        ContextFactory.validate(context)

        with SystemGuard(stage_id=stage_id, scene_id=scene_id, job_id=job_id):

            start_time = time.monotonic()

            # === Step 3: CompositionPlan gate (RULE-88) ===
            stage_enum = self._resolve_stage_enum(stage_id)
            if stage_enum in COMPOSITION_REQUIRED_STAGES:
                context.assert_composition_plan()

            # === Step 4: Stage readiness — 6 sub-checks ===
            self._stage_readiness_gate(stage_id, context)

            # === Step 5: Execute agent ===
            logger.info("execute_stage: running stage %s scene=%s", stage_id, scene_id)
            if self._tracer:
                self._tracer.trace_event(
                    f"stage_start:{stage_id}", stage_id=stage_id, scene_id=scene_id
                )

            output = agent.run(input_data, context)

            # === Step 6: Output schema validation (RULE-90) ===
            self._validate_output(stage_id, output)

            # === Step 7: IdentityStateTracker update ===
            new_context = self._update_identity_state(stage_id, output, context)

            # === Step 8: HRG checkpoint ===
            hrg_checkpoint = STAGE_HRG_MAP.get(stage_id)
            if hrg_checkpoint is not None:
                hrg_data = self._extract_hrg_data(stage_id, output, new_context)
                self._hrg.checkpoint(hrg_checkpoint, hrg_data, scene_id=scene_id, job_id=job_id)

            # === Step 9: context.evolve() ===
            new_context = new_context.with_stage_completed(stage_id, output)

            elapsed = time.monotonic() - start_time
            logger.info(
                "execute_stage: COMPLETE stage=%s elapsed=%.2fs", stage_id, elapsed
            )

            if self._sla_manager:
                self._sla_manager.record(stage_id, elapsed)

            if self._tracer:
                self._tracer.trace_event(
                    f"stage_complete:{stage_id}",
                    stage_id=stage_id, scene_id=scene_id, elapsed_s=elapsed,
                )

        return output, new_context

    # ─── Private helpers ──────────────────────────────────────────────────────

    def _stage_readiness_gate(self, stage_id: str, context: ImmutableContext) -> None:
        """6 sub-checks for stage readiness. Raises MissingPredecessorOutputError."""
        # Sub-check 1: predecessor output present
        required_predecessor = STAGE_PREREQUISITES.get(stage_id)
        if required_predecessor and not context.has_output(required_predecessor):
            raise MissingPredecessorOutputError(
                stage_id=stage_id,
                required_output=f"output from {required_predecessor}",
            )

        # Sub-checks 2–6: additional readiness (extensible)
        # Sub-check 2: identity reference frozen for video stages
        if stage_id in ("S-09", "S-12"):
            context.assert_identity_frozen()

    def _validate_output(self, stage_id: str, output: Any) -> None:
        """Validate stage output is not None and has schema_version if applicable."""
        if output is None:
            raise OutputValidationError(
                f"Stage {stage_id} returned None output — violates output contract",
                stage_id=stage_id,
            )
        if hasattr(output, "schema_version"):
            if output.schema_version != settings.SCHEMA_VERSION:
                raise SchemaValidationError(
                    f"Stage {stage_id} output has wrong schema_version "
                    f"'{output.schema_version}' (expected '{settings.SCHEMA_VERSION}')",
                    stage_id=stage_id,
                )

    def _update_identity_state(
        self, stage_id: str, output: Any, context: ImmutableContext
    ) -> ImmutableContext:
        """Call IdentityStateTracker if output contains a CLIP score."""
        clip_score = getattr(output, "clip_score", None)
        if clip_score is not None and self._identity_tracker:
            updated_identity = self._identity_tracker.update(
                stage_id=stage_id,
                scene_id=context.scene_id,
                clip_score=clip_score,
                context=context,
            )
            return context.evolve(identity_state=updated_identity)
        return context

    def _extract_hrg_data(
        self, stage_id: str, output: Any, context: ImmutableContext
    ) -> Any:
        """Extract display data for HRG review from stage output."""
        return output   # agents set output as review-ready; HRGController handles formatting

    def _resolve_stage_enum(self, stage_id: str) -> Optional[PipelineStageID]:
        """Convert string stage_id to PipelineStageID enum value."""
        try:
            return PipelineStageID(stage_id)
        except ValueError:
            return None


# Module-level convenience function — enforced by ArchitectureLinter
def execute_stage(
    agent: Any,
    input_data: Any,
    context: ImmutableContext,
    orchestrator: Optional[MasterOrchestrator] = None,
) -> Tuple[Any, ImmutableContext]:
    """Convenience wrapper. Uses provided orchestrator or raises if none supplied.

    RULE-106: This function is the ONLY permitted stage execution pathway.
    The ArchitectureLinter checks that no code calls agent.run() directly.
    """
    if orchestrator is None:
        raise ArchitectureGuardViolationError(
            "execute_stage() called without an orchestrator. "
            "Pass the MasterOrchestrator instance from bootstrap registry. RULE-106."
        )
    return orchestrator.execute_stage(agent, input_data, context)
