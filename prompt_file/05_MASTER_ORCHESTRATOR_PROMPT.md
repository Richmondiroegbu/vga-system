# Prompt 05: Master Orchestrator — execute_stage() + SystemGuard + HRGController
**Category:** Orchestration Layer  
**Files to implement:**
- `vga/core/master_orchestrator.py`
- `vga/runtime/system_guard.py`
- `vga/core/hrg_controller.py`
- `vga/core/hrg_state_manager.py`
**Spec References:**
- `RunPod_Model_Download_Specification_v6.5.md` §1.2 (execute_stage enforcement)
- `01_VGA_SRD_v17.2.md` §1.4 (Master Orchestrator execution contract)
- `09_VGA_File_Responsibility_v17.2.md` §core/master_orchestrator.py
**Dependencies:** Prompts 02, 03, 04 complete  
**Build Order:** Step 12.2 — must come before any agent implementation

---

## Context

`execute_stage()` is the **single and only** permitted way to execute any pipeline stage. **RULE-106**: Direct `agent.run()` calls are forbidden and will be detected by ArchitectureLinter.

Every call to `execute_stage()` enforces this exact sequence:
```
1. assert isinstance(context, ImmutableContext)        [RULE-108]
2. SystemGuard.__enter__()                             [isolation]
3. CompositionValidator.assert_in_context(context)     [RULE-88, for image/video stages]
4. stage_readiness_gate()                              [6 sub-checks]
5. agent.run(input_data, context)                      [actual work]
6. output schema validation                            [RULE-90]
7. IdentityStateTracker.update()                       [identity drift]
8. hrg_controller.checkpoint()                         [RULE-109]
9. context.evolve() → new_context                      [FR-950]
10. SystemGuard.__exit__()                             [lifecycle cleanup]
11. return (output, new_context)
```

---

## vga/core/master_orchestrator.py

```python
"""
Master Orchestrator — execute_stage() is the ONLY permitted stage executor.
RULE-106: Direct agent.run() calls are FORBIDDEN.
RULE-108: ImmutableContext mandatory — dict context raises TypeError immediately.
FR-950: context.evolve() called after every stage.
Spec: VGA SRD v17.2 §1.4, VGA Deployment Spec v6.5
"""
from __future__ import annotations
from typing import TypeVar, Type, Callable, Optional
from vga.state.immutable_context import ImmutableContext
from vga.state.context_factory import ContextFactory
from vga.models.enums import PipelineStageID, HRGCheckpoint
from vga.runtime.system_guard import SystemGuard
from vga.core.hrg_controller import HRGController
from vga.core.exceptions import (
    ImmutableContextViolationError, MissingPredecessorOutputError,
    CompositionPlanValidationError, ArchitectureGuardViolationError
)
from vga.config.settings import settings
from vga.core.logger import get_logger

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")

log = get_logger(__name__)

# Stages that require CompositionPlan (RULE-88)
COMPOSITION_REQUIRED_STAGES = {
    PipelineStageID.S05_BASE_IMAGE,
    PipelineStageID.S06_IDENTITY_REINFORCEMENT,
    PipelineStageID.S07_IMAGE_REFINEMENT,
    PipelineStageID.S08_VIDEO_SEGMENT_1,
    PipelineStageID.S09_TEMPORAL_ENGINE,
}

# Stages that require predecessor outputs (FR-950, RULE-90)
STAGE_PREREQUISITES: dict[PipelineStageID, list[str]] = {
    PipelineStageID.S05_BASE_IMAGE: [PipelineStageID.S04_SCENE_COMPOSITION],
    PipelineStageID.S06_IDENTITY_REINFORCEMENT: [PipelineStageID.S05_BASE_IMAGE],
    PipelineStageID.S07_IMAGE_REFINEMENT: [PipelineStageID.S06_IDENTITY_REINFORCEMENT],
    PipelineStageID.S08_VIDEO_SEGMENT_1: [PipelineStageID.S07_IMAGE_REFINEMENT],
    PipelineStageID.S09_TEMPORAL_ENGINE: [PipelineStageID.S08_VIDEO_SEGMENT_1],
    PipelineStageID.S10_CONTINUITY_VALIDATION: [PipelineStageID.S09_TEMPORAL_ENGINE],
    PipelineStageID.S11_DIALOGUE: [PipelineStageID.S10_CONTINUITY_VALIDATION],
    PipelineStageID.S12_LIP_SYNC: [PipelineStageID.S11_DIALOGUE],
    # ...etc
}

def execute_stage(
    stage_id: PipelineStageID,
    agent: object,
    input_data: InputT,
    context: ImmutableContext,
    hrg_checkpoint: Optional[HRGCheckpoint] = None,
    requires_composition: bool = None,  # auto-detected from stage_id if None
) -> tuple[OutputT, ImmutableContext]:
    """
    THE ONLY PERMITTED WAY TO EXECUTE A PIPELINE STAGE.
    
    RULE-106: Direct agent.run() calls are FORBIDDEN.
    RULE-108: Validates ImmutableContext; dict context raises TypeError.
    FR-950: Calls context.evolve() after stage completes.
    
    Enforcement sequence (mandatory):
      1. ImmutableContext type assertion
      2. SystemGuard context manager entry  
      3. CompositionPlan gate (if required)
      4. Stage readiness gate (6 sub-checks)
      5. agent.run(input_data, context)
      6. Output schema validation
      7. IdentityStateTracker update (if output has embedding)
      8. HRG checkpoint (if configured)
      9. context.evolve() with updated state
      10. SystemGuard exit
    
    Returns:
        (output, new_context) — both typed and validated
    """
    # Step 1: ImmutableContext assertion
    context = ContextFactory.validate(context)  # raises ImmutableContextViolationError if dict
    
    # Step 2: Prerequisite check (RULE-90)
    _check_prerequisites(stage_id, context)
    
    # Determine if composition is required
    if requires_composition is None:
        requires_composition = stage_id in COMPOSITION_REQUIRED_STAGES
    
    with SystemGuard(stage_id=stage_id, context=context) as guard:
        # Step 3: CompositionPlan gate
        if requires_composition:
            context.assert_composition_plan()
        
        # Step 4: Stage readiness gate
        stage_readiness_gate(stage_id, context)
        
        # Step 5: Execute agent
        log.info("execute_stage.start", stage_id=stage_id, scene_id=context.scene_id)
        output, new_context = agent.run(input_data, context)
        
        # Step 6: Output validation
        _validate_output(stage_id, output)
        
        # Step 7: Identity tracking (if output has embedding)
        new_context = _update_identity_state(stage_id, output, new_context)
        
        # Step 8: HRG checkpoint
        if hrg_checkpoint is not None:
            _hrg_checkpoint(hrg_checkpoint, new_context, output)
        
        # Step 9: Mark stage complete
        new_context = new_context.with_stage_completed(str(stage_id))
        
        log.info("execute_stage.complete", stage_id=stage_id, scene_id=context.scene_id)
        return output, new_context


def stage_readiness_gate(stage_id: PipelineStageID, context: ImmutableContext) -> None:
    """
    6-sub-check stage readiness enforcer. Hard-stop on any failure.
    Replaces the former bare asset_gate().
    
    Sub-checks:
      1. VRAM headroom (if GPU stage)
      2. Required predecessor outputs exist in context
      3. Required assets loaded in model manager
      4. Diffusion subcomponents present (if diffusion stage)
      5. Identity reference frozen (if post-S05 stage)
      6. Composition plan validated (if image/video stage — already done above)
    """
    # Implement all 6 sub-checks with clear error messages
    ...


def _check_prerequisites(stage_id: PipelineStageID, context: ImmutableContext) -> None:
    """Verify predecessor stages have run (RULE-90)."""
    prerequisites = STAGE_PREREQUISITES.get(stage_id, [])
    for prereq in prerequisites:
        if not context.has_output(str(prereq)):
            raise MissingPredecessorOutputError(
                stage_id=str(stage_id),
                required_output=str(prereq)
            )


def _validate_output(stage_id: PipelineStageID, output: object) -> None:
    """Validate output schema after stage execution."""
    # Use Pydantic model_validate() to confirm output matches expected schema
    ...


def _update_identity_state(
    stage_id: PipelineStageID, 
    output: object, 
    context: ImmutableContext
) -> ImmutableContext:
    """
    Update IdentityState if output contains embedding/drift info.
    Called for stages that produce identity-relevant outputs.
    """
    # Check if output has drift_score or embedding
    # Update context.identity_state via evolve()
    ...


def _hrg_checkpoint(
    checkpoint: HRGCheckpoint,
    context: ImmutableContext,
    output: object,
) -> None:
    """Invoke HRGController for human review. RULE-109."""
    hrg = HRGController.get_instance()
    hrg.checkpoint(checkpoint, context, output)
```

---

## vga/runtime/system_guard.py

```python
"""
SystemGuard — Context manager that wraps every stage execution.
Provides: structured lifecycle logging, failure classification, uncontrolled execution prevention.
Used as: `with SystemGuard(stage_id, context): execute_stage(...)`
Spec: VGA Deployment Spec v6.5 §SystemGuard
"""
from __future__ import annotations
import time
from contextlib import contextmanager
from typing import Optional
from vga.models.enums import PipelineStageID, FailureSeverity
from vga.state.immutable_context import ImmutableContext
from vga.core.logger import get_logger

log = get_logger(__name__)

class SystemGuard:
    """
    Context manager that wraps all stage executions.
    
    Responsibilities:
    - Log stage entry with timestamp and context snapshot
    - Log stage exit with elapsed time and success/failure
    - Classify failures via classify_failure()
    - Prevent uncontrolled execution (double-entry guard)
    - Clean up GPU state on failure
    """
    
    _active_stages: set[str] = set()   # class-level set to detect re-entry
    
    def __init__(self, stage_id: PipelineStageID, context: ImmutableContext):
        self.stage_id = str(stage_id)
        self.context = context
        self._start_time: float = 0.0
        self._entered: bool = False
    
    def __enter__(self) -> "SystemGuard":
        if self.stage_id in self._active_stages:
            raise RuntimeError(f"Stage {self.stage_id} is already executing (re-entry detected)")
        
        self._active_stages.add(self.stage_id)
        self._start_time = time.monotonic()
        self._entered = True
        
        log.info(
            "system_guard.enter",
            stage_id=self.stage_id,
            scene_id=self.context.scene_id,
            job_id=self.context.job_id,
            has_composition=self.context.composition_plan is not None,
            identity_frozen=self.context.identity_state.is_frozen,
        )
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        elapsed = time.monotonic() - self._start_time
        self._active_stages.discard(self.stage_id)
        
        if exc_type is None:
            log.info("system_guard.exit.success", stage_id=self.stage_id, elapsed_s=elapsed)
            return False  # don't suppress
        
        # Classify and log the failure
        severity = self.classify_failure(exc_val)
        log.error(
            "system_guard.exit.failure",
            stage_id=self.stage_id,
            elapsed_s=elapsed,
            severity=severity,
            error=str(exc_val),
            error_type=exc_type.__name__,
        )
        
        # GPU cleanup on failure
        self._cleanup_gpu()
        
        return False  # let exception propagate
    
    @staticmethod
    def classify_failure(exc: Exception | None) -> FailureSeverity:
        """
        Map exception type to severity level.
        CRITICAL = pipeline halts; DEGRADED = retry; WARNING = log and continue.
        """
        from vga.core.exceptions import (
            TemporalBufferError, SVICFGViolationError, AutoregressiveViolationError,
            CompositionPlanValidationError, IdentityReferenceCorruptionError,
            ImmutableContextViolationError, ArchitectureGuardViolationError,
            MissingPredecessorOutputError,
            CLIPValidationError, AudioQualityError, RetryExhaustedError,
        )
        
        CRITICAL_TYPES = (
            TemporalBufferError, SVICFGViolationError, AutoregressiveViolationError,
            IdentityReferenceCorruptionError, ImmutableContextViolationError,
            ArchitectureGuardViolationError, MissingPredecessorOutputError,
        )
        DEGRADED_TYPES = (CLIPValidationError, AudioQualityError, RetryExhaustedError)
        
        if exc is None:
            return FailureSeverity.WARNING
        if isinstance(exc, CRITICAL_TYPES):
            return FailureSeverity.CRITICAL
        if isinstance(exc, DEGRADED_TYPES):
            return FailureSeverity.DEGRADED
        return FailureSeverity.WARNING
    
    def _cleanup_gpu(self) -> None:
        """Emergency GPU cleanup on stage failure."""
        try:
            import gc, torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
```

---

## vga/core/hrg_controller.py

```python
"""
HRGController — Human Review Gate controller for all 11 checkpoints.
RULE-109: HRG checkpoint mandatory after every stage output.
Spec: VGA SRD v17.2 §7.1 (HRG Checkpoint Panels)
"""
from vga.models.enums import HRGCheckpoint
from vga.state.immutable_context import ImmutableContext
from vga.config.settings import settings
from vga.core.logger import get_logger

log = get_logger(__name__)

class HRGController:
    """
    Manages human review at all 11 pipeline checkpoints.
    
    Behavior:
    - If HRG_REVIEW_ENABLED=True: blocks pipeline and waits for human approval
    - If HRG_REVIEW_ENABLED=False: auto-approves (useful for test runs)
    - Timeout: HRG_APPROVAL_TIMEOUT_SECONDS (default 300)
    - All decisions logged to /workspace/hrg/approvals/hrg_log_{job_id}.json
    """
    _instance: "HRGController | None" = None
    
    @classmethod
    def get_instance(cls) -> "HRGController":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def checkpoint(
        self,
        checkpoint: HRGCheckpoint,
        context: ImmutableContext,
        output: object,
    ) -> None:
        """
        Execute a human review checkpoint.
        
        Workflow:
        1. Prepare display data (checkpoint-specific)
        2. POST to FastAPI: /jobs/{job_id}/hrg/{checkpoint}
        3. If enabled: wait_for_approval() blocks until human responds
        4. Log decision with user/action/timestamp (OR-038)
        5. If rejected: raise HRGRejectionError
        """
        ...
    
    def wait_for_approval(
        self,
        checkpoint: HRGCheckpoint,
        job_id: str,
        timeout_s: int = None,
    ) -> dict:
        """
        Block until human approves or rejects at checkpoint.
        Polls /jobs/{job_id}/hrg/{checkpoint}/status every 5 seconds.
        Raises HRGTimeoutError after timeout_s.
        """
        ...
    
    def _prepare_hrg_display_data(
        self,
        checkpoint: HRGCheckpoint,
        context: ImmutableContext,
        output: object,
    ) -> dict:
        """
        Build checkpoint-specific display data for Streamlit UI.
        
        HRG-4 (CompositionPlan) display:
          - camera_angle, camera_motion, character_positions, focus_subject
          - lighting_style, motion_vector
          - editable fields: all 6 CompositionPlan fields
        
        HRG-8 (Motion QA) display:
          - video segments with thumbnails
          - continuity_score with breakdown (motion/lighting/identity weights)
          - identity_per_segment: List[float] (NEW v17.0)
        
        HRG-10 (Lip Sync QA) display:
          - lip-synced video preview
          - phoneme_alignment score
          - identity_delta per segment (NEW v17.0)
        
        HRG-11 (Final Audio QA) display:
          - audio waveform
          - SNR badge (✅ ≥ 10dB or ❌ < 10dB) (NEW v17.0)
          - clipping status (✅ no clipping or ❌ clipping detected) (NEW v17.0)
          - mix levels for dialogue/ambient/music
        """
        ...
```

---

## Acceptance Criteria

- [ ] `execute_stage(stage_id, agent, input, {"key": "val"})` raises `ImmutableContextViolationError`
- [ ] `execute_stage(S-05, agent, input, context_without_composition_plan)` raises `CompositionPlanValidationError`
- [ ] `execute_stage(S-06, agent, input, context_without_S05_completion)` raises `MissingPredecessorOutputError`
- [ ] `SystemGuard.classify_failure(TemporalBufferError(...))` returns `FailureSeverity.CRITICAL`
- [ ] `SystemGuard.classify_failure(CLIPValidationError(...))` returns `FailureSeverity.DEGRADED`
- [ ] `HRGController.get_instance()` returns the same instance (singleton)
- [ ] HRG-4 display data contains all 6 CompositionPlan fields
- [ ] HRG-8 display data contains `identity_per_segment` field
- [ ] HRG-11 display data contains `snr_db`, `peak_db`, `clipping_detected` fields
