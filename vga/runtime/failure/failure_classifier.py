"""
FailureClassifier — classifies exceptions into CRITICAL/DEGRADED/WARNING severity.
Spec: VGA Runtime Spec v17.2 §failure/failure_classifier.py
"""
from __future__ import annotations

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

_CRITICAL = (
    TemporalBufferError,
    AutoregressiveViolationError,
    ImmutableContextViolationError,
    ArchitectureGuardViolationError,
    VRAMViolationError,
    SVICFGViolationError,
    CriticalPipelineError,
)

_DEGRADED = (
    CLIPValidationError,
    AudioQualityError,
    IdentityCumulativeDriftError,
    CompositionPlanValidationError,
)


def classify(exc: BaseException) -> FailureSeverity:
    """Map an exception to its FailureSeverity level."""
    if isinstance(exc, _CRITICAL):
        return FailureSeverity.CRITICAL
    if isinstance(exc, _DEGRADED):
        return FailureSeverity.DEGRADED
    if isinstance(exc, VGABaseError):
        return FailureSeverity.WARNING
    return FailureSeverity.CRITICAL   # unknown exceptions → critical
