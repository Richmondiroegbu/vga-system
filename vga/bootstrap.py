"""
VGA Bootstrap — Singleton initialization sequence.
Called once at FastAPI startup (RULE-85).
Order matters — dependencies must be initialized before dependents.
Spec: VGA System Architecture Document v17.2 §4 (Bootstrap Sequence)
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def initialize_all_singletons() -> dict:
    """Initialize all VGA singletons in dependency order.

    Returns a registry dict mapping component name → singleton instance.
    Raises on any initialization failure (pipeline cannot run with missing singletons).

    v17.0 singletons (6Z-p through 6Z-z) initialized in order:
      6Z-p: SceneCompositionAgent
      6Z-q: TemporalBufferManager
      6Z-r: SVIScheduler factory
      6Z-s: MotionStateTracker
      6Z-t: TemporalRetryController
      6Z-u: IdentityStateTracker
      6Z-v: AudioQualityValidator
      6Z-w: CrossModalAlignmentValidator
      6Z-x: CompositionPlanValidator
      6Z-y: HRGController (11 checkpoints)
      6Z-z: TemporalEngine
    """
    registry: dict = {}

    logger.info("VGA bootstrap: starting singleton initialization sequence")

    # --- Core foundation (no dependencies) ---
    try:
        from vga.core.logger import get_logger  # noqa: F401
        registry["logger"] = get_logger
        logger.info("Bootstrap 1/12: logger ready")
    except Exception as exc:
        raise RuntimeError(f"Bootstrap failed at logger: {exc}") from exc

    try:
        from vga.core.model_manager import ModelManager
        registry["model_manager"] = ModelManager()
        logger.info("Bootstrap 2/12: ModelManager ready")
    except Exception as exc:
        raise RuntimeError(f"Bootstrap failed at ModelManager: {exc}") from exc

    # --- Validation layer ---
    try:
        from vga.validation.clip_validator import CLIPValidator
        from vga.validation.composition_validator import CompositionValidator
        from vga.validation.audio_quality_validator import AudioQualityValidator
        from vga.validation.cross_modal_alignment_validator import CrossModalAlignmentValidator

        registry["clip_validator"] = CLIPValidator()
        registry["composition_validator"] = CompositionValidator()
        registry["audio_quality_validator"] = AudioQualityValidator()         # 6Z-v
        registry["cross_modal_validator"] = CrossModalAlignmentValidator()    # 6Z-w
        registry["composition_plan_validator"] = registry["composition_validator"]  # 6Z-x alias
        logger.info("Bootstrap 3/12: validators ready")
    except Exception as exc:
        raise RuntimeError(f"Bootstrap failed at validators: {exc}") from exc

    # --- Identity system (6Z-u) ---
    try:
        from vga.identity.identity_state_tracker import IdentityStateTracker
        registry["identity_state_tracker"] = IdentityStateTracker()
        logger.info("Bootstrap 4/12: IdentityStateTracker ready (6Z-u)")
    except Exception as exc:
        raise RuntimeError(f"Bootstrap failed at IdentityStateTracker: {exc}") from exc

    # --- Temporal subsystem (6Z-q through 6Z-t, 6Z-z) ---
    try:
        from vga.temporal.temporal_buffer_manager import TemporalBufferManager
        from vga.temporal.motion_state_tracker import MotionStateTracker
        from vga.temporal.temporal_retry_controller import TemporalRetryController
        from vga.temporal.temporal_engine import TemporalEngine

        registry["temporal_buffer_manager"] = TemporalBufferManager()        # 6Z-q
        registry["motion_state_tracker"] = MotionStateTracker()              # 6Z-s
        registry["temporal_retry_controller"] = TemporalRetryController()    # 6Z-t
        registry["temporal_engine"] = TemporalEngine(                        # 6Z-z
            buffer_manager=registry["temporal_buffer_manager"],
            motion_tracker=registry["motion_state_tracker"],
            retry_controller=registry["temporal_retry_controller"],
            clip_validator=registry["clip_validator"],
        )
        logger.info("Bootstrap 5/12: Temporal subsystem ready (6Z-q/s/t/z)")
    except Exception as exc:
        raise RuntimeError(f"Bootstrap failed at Temporal subsystem: {exc}") from exc

    # --- HRG Controller (6Z-y) ---
    try:
        from vga.core.hrg_controller import HRGController
        registry["hrg_controller"] = HRGController()
        logger.info("Bootstrap 6/12: HRGController ready — 11 checkpoints (6Z-y)")
    except Exception as exc:
        raise RuntimeError(f"Bootstrap failed at HRGController: {exc}") from exc

    # --- SLA / Gating ---
    try:
        from vga.runtime.sla_manager import SLAManager
        from vga.runtime.gating_controller import GatingController
        registry["sla_manager"] = SLAManager()
        registry["gating_controller"] = GatingController()
        logger.info("Bootstrap 7/12: SLA and gating ready")
    except Exception as exc:
        raise RuntimeError(f"Bootstrap failed at SLA/Gating: {exc}") from exc

    # --- Observability ---
    try:
        from vga.observability.tracer import Tracer
        registry["tracer"] = Tracer()
        logger.info("Bootstrap 8/12: Tracer ready")
    except Exception as exc:
        raise RuntimeError(f"Bootstrap failed at Tracer: {exc}") from exc

    # --- Pipeline agents (6Z-p: SceneCompositionAgent first) ---
    try:
        from vga.agents.scene_composition_agent import SceneCompositionAgent  # 6Z-p
        registry["scene_composition_agent"] = SceneCompositionAgent()
        logger.info("Bootstrap 9/12: SceneCompositionAgent ready (6Z-p)")
    except Exception as exc:
        raise RuntimeError(f"Bootstrap failed at SceneCompositionAgent: {exc}") from exc

    # --- Master Orchestrator (depends on everything above) ---
    try:
        from vga.core.master_orchestrator import MasterOrchestrator
        registry["orchestrator"] = MasterOrchestrator(
            hrg_controller=registry["hrg_controller"],
            composition_validator=registry["composition_validator"],
            identity_tracker=registry["identity_state_tracker"],
            sla_manager=registry["sla_manager"],
            tracer=registry["tracer"],
        )
        logger.info("Bootstrap 10/12: MasterOrchestrator ready")
    except Exception as exc:
        raise RuntimeError(f"Bootstrap failed at MasterOrchestrator: {exc}") from exc

    logger.info("VGA bootstrap complete — all singletons initialized (%d components)", len(registry))
    return registry


# Module-level registry populated at startup
_registry: dict | None = None


def get_registry() -> dict:
    """Return the initialized singleton registry. Raises if bootstrap has not run."""
    if _registry is None:
        raise RuntimeError("VGA bootstrap has not been called. Call initialize_all_singletons() first.")
    return _registry


def run_bootstrap() -> dict:
    """Public entry point called by FastAPI lifespan."""
    global _registry
    _registry = initialize_all_singletons()
    return _registry
