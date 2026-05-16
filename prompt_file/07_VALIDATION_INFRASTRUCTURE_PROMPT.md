# Prompt 07: Validation Infrastructure
**Category:** Validation  
**Files:**
- `vga/validation/clip_validator.py`
- `vga/validation/composition_validator.py`
- `vga/validation/audio_quality_validator.py`
- `vga/validation/cross_modal_alignment_validator.py`
- `vga/validation/timing_validator.py`
**Spec:** `01_VGA_SRD_v17.2.md` §3.71–3.72, SEC-059 through SEC-064

## CLIPValidator
```python
class CLIPValidator:
    """
    Validates character identity via CLIP cosine similarity.
    RULE-92: ≥ 0.93 at every validation point.
    RULE-95: ALWAYS uses frozen char_identity_ref from ImmutableContext.identity_state.embedding_vector.
    NEVER recomputes the reference embedding mid-pipeline.
    """
    
    def score(self, frame: Tensor, reference_embedding: ndarray) -> float:
        """
        Compute CLIP cosine similarity between frame and frozen reference.
        
        Args:
            frame: video frame or image tensor
            reference_embedding: FROZEN embedding from S-05 (ImmutableContext)
        Returns: float in [0.0, 1.0]
        """
        # Encode frame with CLIP image encoder
        frame_embedding = self._encode_image(frame)
        # Cosine similarity with frozen reference
        return float(F.cosine_similarity(frame_embedding, reference_embedding_tensor, dim=-1))
    
    def validate_or_raise(self, frame, reference_embedding, stage_id: str) -> float:
        """Score and raise CLIPValidationError if below threshold."""
        score = self.score(frame, reference_embedding)
        if score < settings.CLIP_IDENTITY_THRESHOLD:
            raise CLIPValidationError(
                f"CLIP score {score:.3f} < {settings.CLIP_IDENTITY_THRESHOLD} at {stage_id}"
            )
        return score
```

## CompositionValidator
```python
class CompositionPlanValidator:
    """
    Full Pydantic schema assertion for CompositionPlan.
    RULE-88: CompositionPlan ALL 6 fields required.
    Replaces the former null-check with full schema validation.
    """
    
    def validate(self, plan: Any) -> CompositionPlanSchema:
        """Validate and return typed CompositionPlan. Raises on any violation."""
        return CompositionPlanSchema.model_validate(plan)
    
    def assert_in_context(self, context: ImmutableContext) -> None:
        """Assert CompositionPlan present and valid in context."""
        if context.composition_plan is None:
            raise CompositionPlanValidationError("CompositionPlan missing (RULE-88)")
        self.validate(context.composition_plan.model_dump())  # re-validate
```

## AudioQualityValidator (NEW v17.0)
```python
class AudioQualityValidator:
    """
    RULE-99: SNR ≥ 10 dB + peaks ≤ 0 dBFS (no clipping).
    FR-970–FR-971: Hard requirements after AudioMixingAgent.
    SLA: complete in ≤ 5 seconds per scene (NFR-171).
    """
    
    def validate(self, audio_path: str, scene_id: str) -> AudioQualityRecord:
        """
        Compute SNR and peak level. Raise AudioQualityError on violation.
        Uses torchaudio or librosa for analysis.
        """
        import torchaudio
        waveform, sr = torchaudio.load(audio_path)
        
        snr_db = self._compute_snr(waveform)
        peak_db = self._compute_peak_db(waveform)
        clipping = peak_db > settings.MAX_PEAK_DBFS
        
        record = AudioQualityRecord(
            scene_id=scene_id,
            snr_db=snr_db,
            peak_db=peak_db,
            clipping_detected=clipping,
            snr_passed=snr_db >= settings.MIN_SNR_DB,
            clipping_passed=not clipping,
        )
        
        if not record.snr_passed or not record.clipping_passed:
            raise AudioQualityError(snr_db=snr_db, peak_db=peak_db)
        
        return record
    
    def _compute_snr(self, waveform) -> float:
        """Signal-to-noise ratio in dB."""
        signal_power = waveform.pow(2).mean()
        # Estimate noise from quiet segments
        ...
    
    def _compute_peak_db(self, waveform) -> float:
        """Peak level in dBFS."""
        peak = waveform.abs().max()
        return 20 * torch.log10(peak + 1e-8).item()
```

## CrossModalAlignmentValidator (NEW v17.0)
```python
class CrossModalAlignmentValidator:
    """
    FR-972: Video ↔ Audio duration alignment within ±0.10s.
    FR-973: Segment boundaries identical between audio and video.
    Used at S-12 (LipSync) and S-13 (AmbientAudio).
    """
    
    def validate_cross_modal(
        self,
        video_path: str,
        audio_path: str,
        scene_id: str,
        segment_id: str,
    ) -> CrossModalAlignmentRecord:
        """
        Compute sync score and validate alignment.
        Raises CrossModalAlignmentError if duration mismatch > ±0.10s.
        """
        video_duration = self._get_video_duration(video_path)
        audio_duration = self._get_audio_duration(audio_path)
        error = abs(video_duration - audio_duration)
        
        record = CrossModalAlignmentRecord(
            scene_id=scene_id,
            segment_id=segment_id,
            video_duration_s=video_duration,
            audio_duration_s=audio_duration,
            alignment_error_s=error,
            within_tolerance=error <= settings.TIMING_TOLERANCE_S,
        )
        
        if not record.within_tolerance:
            raise CrossModalAlignmentError(
                f"Alignment error {error:.3f}s > {settings.TIMING_TOLERANCE_S}s tolerance"
            )
        return record
```

## Acceptance Criteria
- [ ] `CLIPValidator.score(frame, reference)` always uses reference as second arg (not recomputed)
- [ ] `CompositionPlanValidator.assert_in_context()` raises if plan is None
- [ ] `AudioQualityValidator.validate()` raises `AudioQualityError` when SNR < 10dB
- [ ] `CrossModalAlignmentValidator.validate_cross_modal()` raises when duration diff > 0.10s
- [ ] All validators complete within SLA time constraints
