"""
SystemGuard — context manager wrapping every stage execution.
Provides structured lifecycle logging, double-entry protection, and GPU cleanup on failure.
Spec: VGA Engine Template Spec v17.2 §4; FR-220–FR-228
"""
from __future__ import annotations

import logging
import time
from types import TracebackType
from typing import Optional, Type

from vga.core.exceptions import (
    ArchitectureGuardViolationError,
    AudioQualityError,
    AutoregressiveViolationError,
    CLIPValidationError,
    CompositionPlanValidationError,
    CriticalPipelineError,
    IdentityCumulativeDriftError,
    ImmutableContextViolationError,
    SVICFGViolationError,
    TemporalBufferError,
    VGABaseError,
    VRAMViolationError,
)
from vga.models.enums import FailureSeverity

logger = logging.getLogger(__name__)

# Exceptions that always result in CRITICAL (pipeline halt)
_CRITICAL_EXCEPTIONS = (
    TemporalBufferError,
    AutoregressiveViolationError,
    ImmutableContextViolationError,
    ArchitectureGuardViolationError,
    VRAMViolationError,
    SVICFGViolationError,
    CriticalPipelineError,
)

# Exceptions that are DEGRADED (retry up to 3 times)
_DEGRADED_EXCEPTIONS = (
    CLIPValidationError,
    AudioQualityError,
    IdentityCumulativeDriftError,
    CompositionPlanValidationError,
)


class SystemGuard:
    """Context manager providing stage isolation and lifecycle enforcement.

    Usage (inside execute_stage only):
        with SystemGuard(stage_id="S-01", scene_id="sc_001") as guard:
            result = agent.run(input_data, context)
    """

    _active_stages: set[str] = set()   # class-level guard against re-entry

    def __init__(self, stage_id: str, scene_id: str, job_id: str = "") -> None:
        self.stage_id = stage_id
        self.scene_id = scene_id
        self.job_id = job_id
        self._start_time: float = 0.0
        self._entered = False

    def __enter__(self) -> "SystemGuard":
        if self.stage_id in SystemGuard._active_stages:
            raise ArchitectureGuardViolationError(
                f"SystemGuard double-entry detected for stage {self.stage_id}. "
                f"Stage is already executing — concurrent execution is forbidden."
            )
        SystemGuard._active_stages.add(self.stage_id)
        self._entered = True
        self._start_time = time.monotonic()
        logger.info(
            "SystemGuard ENTER stage=%s scene=%s job=%s",
            self.stage_id, self.scene_id, self.job_id,
        )
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> bool:
        elapsed = time.monotonic() - self._start_time
        if self._entered:
            SystemGuard._active_stages.discard(self.stage_id)

        if exc_val is None:
            logger.info(
                "SystemGuard EXIT OK stage=%s elapsed=%.2fs",
                self.stage_id, elapsed,
            )
            return False   # don't suppress

        severity = self.classify_failure(exc_val)
        logger.error(
            "SystemGuard EXIT FAILURE stage=%s severity=%s error=%s elapsed=%.2fs",
            self.stage_id, severity.value, str(exc_val), elapsed,
        )

        if severity == FailureSeverity.CRITICAL:
            self._cleanup_gpu()

        return False   # always re-raise

    @staticmethod
    def classify_failure(exc: BaseException) -> FailureSeverity:
        """Map exception type to FailureSeverity level."""
        if isinstance(exc, _CRITICAL_EXCEPTIONS):
            return FailureSeverity.CRITICAL
        if isinstance(exc, _DEGRADED_EXCEPTIONS):
            return FailureSeverity.DEGRADED
        if isinstance(exc, VGABaseError):
            return FailureSeverity.WARNING
        return FailureSeverity.CRITICAL   # unknown exceptions → critical

    def _cleanup_gpu(self) -> None:
        """Release GPU resources after a CRITICAL failure."""
        try:
            import gc
            import torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("SystemGuard: GPU cleanup completed after CRITICAL failure")
        except Exception as e:
            logger.warning("SystemGuard: GPU cleanup failed: %s", e)
