"""
VGA Exception Hierarchy.
All exceptions derive from VGABaseError.
Import exceptions from HERE — never define inline in other files.
Spec: VGA File Responsibility Spec v17.2 §core/exceptions.py
"""
from __future__ import annotations


class VGABaseError(Exception):
    """Base class for all VGA pipeline exceptions."""

    def __init__(self, message: str, stage_id: str | None = None, **kwargs):
        self.stage_id = stage_id
        self.details = kwargs
        super().__init__(message)


# ─── v15.0 and earlier exceptions (retained) ─────────────────────────────────

class CriticalPipelineError(VGABaseError):
    """Unrecoverable pipeline failure — system must halt."""


class PipelineAbortError(VGABaseError):
    """Pipeline aborted by operator or HRG rejection."""


class SchemaValidationError(VGABaseError):
    """Pydantic schema validation failed for a stage input or output."""


class ModelLoadError(VGABaseError):
    """Model failed to load from disk or registry."""


class ModelUnloadError(VGABaseError):
    """Model failed to unload cleanly from VRAM."""


class VRAMViolationError(VGABaseError):
    """Attempt to load a second heavy model while one is already in VRAM."""


class CLIPValidationError(VGABaseError):
    """CLIP identity score below required threshold. RULE-92."""


class SLAViolationError(VGABaseError):
    """Stage exceeded its SLA time budget."""


class HRGTimeoutError(VGABaseError):
    """Human review gate did not receive a response within the timeout window."""


class HRGRejectionError(VGABaseError):
    """Human reviewer rejected the stage output — pipeline must regenerate."""


class RetryExhaustedError(VGABaseError):
    """All retry attempts for a stage have been exhausted."""


class ArchitectureGuardViolationError(VGABaseError):
    """Direct agent.run() call detected — use execute_stage() only. RULE-106."""


class LoRALoadError(VGABaseError):
    """LoRA weights failed to load or apply."""


class OutputValidationError(VGABaseError):
    """Stage output failed schema or quality validation. RULE-90."""


class ContextEvolutionError(VGABaseError):
    """context.evolve() was not called or returned an invalid context."""


# ─── v17.0 NEW exceptions ────────────────────────────────────────────────────

class CompositionPlanValidationError(VGABaseError):
    """CompositionPlan schema validation failed or plan missing before generation. RULE-88."""


class TemporalBufferError(VGABaseError):
    """TemporalBuffer constraint violated (size ≠ 5, resolution mismatch, device error). RULE-86."""

    def __init__(
        self,
        message: str,
        frame_count: int | None = None,
        required: int = 5,
        **kwargs,
    ):
        self.frame_count = frame_count
        self.required = required
        super().__init__(message, **kwargs)


class SVICFGViolationError(VGABaseError):
    """SVI CFG value is outside the allowed range [5.0, 6.0]. RULE-86, FR-936."""

    def __init__(self, cfg_value: float, **kwargs):
        self.cfg_value = cfg_value
        super().__init__(
            f"SVI CFG {cfg_value:.2f} outside [5.0, 6.0] — color banding risk",
            **kwargs,
        )


class AutoregressiveViolationError(VGABaseError):
    """Batch SVI generation or single-frame conditioning attempted. RULE-87."""


class TemporalSegmentFailureError(VGABaseError):
    """All retries exhausted for a temporal segment. RULE-87."""

    def __init__(self, scene_id: str, segment_id: int, **kwargs):
        self.scene_id = scene_id
        self.segment_id = segment_id
        super().__init__(
            f"Temporal segment {segment_id} in scene {scene_id} failed after all retries",
            **kwargs,
        )


class IdentityCumulativeDriftError(VGABaseError):
    """Cumulative identity drift exceeded threshold — full phase regeneration required. RULE-95."""

    def __init__(self, drift_score: float, threshold: float, **kwargs):
        self.drift_score = drift_score
        self.threshold = threshold
        super().__init__(
            f"Identity cumulative drift {drift_score:.4f} exceeds threshold {threshold:.4f}",
            **kwargs,
        )


class IdentityReferenceCorruptionError(VGABaseError):
    """char_identity_ref was recomputed or mutated mid-pipeline. RULE-95."""


class AudioQualityError(VGABaseError):
    """SNR < 10 dB or peaks > 0 dBFS detected in audio output. RULE-99."""

    def __init__(
        self,
        snr_db: float | None = None,
        peak_db: float | None = None,
        **kwargs,
    ):
        self.snr_db = snr_db
        self.peak_db = peak_db
        snr_str = f"{snr_db:.1f}" if snr_db is not None else "N/A"
        peak_str = f"{peak_db:.1f}" if peak_db is not None else "N/A"
        msg = f"Audio quality failure: SNR={snr_str}dB (min 10), peak={peak_str}dBFS (max 0)"
        super().__init__(msg, **kwargs)


class MissingPredecessorOutputError(VGABaseError):
    """Stage tried to run without required predecessor output. RULE-90."""

    def __init__(self, stage_id: str, required_output: str, **kwargs):
        super().__init__(
            f"Stage {stage_id} requires '{required_output}' from predecessor stage",
            **kwargs,
        )


class SVISchedulerViolationError(VGABaseError):
    """Static LoRA weight used in SVI generation instead of dynamic schedule. RULE-86."""


class CrossModalAlignmentError(VGABaseError):
    """Video-audio alignment error exceeds ±0.10 s tolerance. FR-972."""


class ImmutableContextViolationError(VGABaseError):
    """Dict-based context was used instead of ImmutableContext. RULE-108."""
