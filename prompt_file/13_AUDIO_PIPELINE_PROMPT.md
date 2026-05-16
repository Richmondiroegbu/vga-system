# Prompt 13: Audio Pipeline (S-11 through S-15)
**Category:** Audio Pipeline  
**Files:**
- `vga/agents/dialogue_agent.py` (S-11)
- `vga/agents/lip_sync_agent.py` (S-12)
- `vga/agents/ambient_audio_agent.py` (S-13)
- `vga/agents/music_agent.py` (S-14)
- `vga/agents/audio_mixing_agent.py` (S-15) — CRITICAL
**Spec:** `01_VGA_SRD_v17.2.md` §3.72, §11.1 (Audio Directive v17)

## DialogueAgent (S-11)
```python
class DialogueAgent(BaseAgent):
    """
    S-11: Generate segment-aligned speech using CosyVoice3-0.5B.
    RULE-96: timing_error ≤ 0.10s.
    Output: per-segment WAV files with timing metadata.
    """
    def run(self, input_data, context: ImmutableContext):
        # For each segment: synthesize + validate timing
        for seg_data in input_data.segments:
            audio_path, actual_duration = self.cosyvoice_wrapper.synthesize(
                text=seg_data.dialogue,
                target_duration_s=seg_data.target_duration_s,
            )
            timing_error = abs(actual_duration - seg_data.target_duration_s)
            if timing_error > settings.TIMING_TOLERANCE_S:
                raise TimingValidationError(f"Timing error {timing_error:.3f}s > 0.10s")
        # HRG-9 checkpoint
        ...
```

## LipSyncAgent (S-12) — identity validation post-sync
```python
class LipSyncAgent(BaseAgent):
    """
    S-12: Lip sync video segments to dialogue using LatentSync-1.6.
    phoneme_alignment ≥ 0.80.
    RULE-97: identity_delta ≤ 0.03 after lip sync.
    RULE-89: CLIPValidator on synced frames.
    CrossModalAlignmentValidator at this stage (FR-972, RULE-110).
    IdentityStateTracker.update() per segment (NEW v17.0).
    """
    def run(self, input_data, context: ImmutableContext):
        synced_segments = []
        for i, (video_seg, audio_seg) in enumerate(zip(input_data.video_segments, input_data.audio_segments)):
            synced = self.latentsync_wrapper.sync(video_seg.path, audio_seg.path)
            
            # CLIP identity validation post-sync (RULE-89)
            synced_keyframe = self._extract_keyframe(synced)
            clip_score = self.clip_validator.score(synced_keyframe, context.identity_state.embedding_vector)
            
            # Identity delta check (RULE-97)
            prior_clip_score = input_data.pre_sync_clip_scores[i]
            delta = abs(clip_score - prior_clip_score)
            if delta > settings.LIPSYNC_IDENTITY_DELTA_THRESHOLD:
                raise CLIPValidationError(f"Lip sync identity delta {delta:.4f} > 0.03 (RULE-97)")
            
            # Update IdentityStateTracker
            new_identity = context.identity_state.update_drift(delta)
            context = context.evolve(identity_state=new_identity)
            
            # CrossModal validation (RULE-110)
            self.cross_modal_validator.validate_cross_modal(
                synced.path, audio_seg.path, context.scene_id, f"segment_{i}"
            )
            synced_segments.append(synced)
        
        # HRG-10 checkpoint (identity delta + phoneme alignment)
        self.hrg_controller.checkpoint(HRGCheckpoint.HRG_10_LIPSYNC_QA, context, {
            "segments": synced_segments,
            "identity_delta_per_segment": [s.identity_delta for s in synced_segments],
        })
        return synced_segments, context
```

## AudioMixingAgent (S-15) — MOST CRITICAL AUDIO STAGE
```python
class AudioMixingAgent(BaseAgent):
    """
    S-15: Mix dialogue + ambient + music with priority hierarchy.
    Priority (RULE-98): Dialogue (0dB) > Ambient (−12dB) > Music (−18dB)
    Ducking: ambient/music at −6dB during dialogue.
    RULE-99: SNR ≥ 10dB + peaks ≤ 0dBFS (HARD REQUIREMENTS).
    Audio validation MUST complete before HRG-11.
    """
    
    def run(self, input_data, context: ImmutableContext):
        # Mix with priority levels
        mixed = self._mix_audio(
            dialogue=input_data.dialogue_audio,
            ambient=input_data.ambient_audio,
            music=input_data.music_audio,
        )
        
        # SNR + clipping validation (RULE-99, SEC-064)
        audio_record = self.audio_quality_validator.validate(
            mixed.path, context.scene_id
        )
        # Raises AudioQualityError if SNR < 10dB or peak > 0dBFS
        
        # CrossModal validation (FR-972)
        for i, video_seg in enumerate(input_data.video_segments):
            self.cross_modal_validator.validate_cross_modal(
                video_seg.path, mixed.segment_paths[i],
                context.scene_id, f"segment_{i}",
            )
        
        # HRG-11 checkpoint with SNR badge and clipping status
        self.hrg_controller.checkpoint(
            HRGCheckpoint.HRG_11_FINAL_AUDIO_QA,
            context,
            {"audio_record": audio_record, "mixed": mixed},
        )
        return mixed, context
    
    def _mix_audio(self, dialogue, ambient, music) -> "MixedAudio":
        """
        Mix with pydub. Priority hierarchy (RULE-98):
        - Dialogue: 0 dB (no attenuation)
        - Ambient: −12 dB (−6 dB during dialogue = ducking)
        - Music: −18 dB (−6 dB during dialogue = ducking)
        
        Apply automatic normalization to prevent clipping (peaks ≤ 0 dBFS).
        """
        from pydub import AudioSegment
        ...
```

## Acceptance Criteria
- [ ] `DialogueAgent`: timing_error > 0.10s raises `TimingValidationError`
- [ ] `LipSyncAgent`: identity_delta > 0.03 raises `CLIPValidationError`
- [ ] `LipSyncAgent`: calls `IdentityStateTracker.update()` per segment
- [ ] `LipSyncAgent`: calls `CrossModalAlignmentValidator` (RULE-110)
- [ ] `AudioMixingAgent`: SNR < 10dB raises `AudioQualityError`
- [ ] `AudioMixingAgent`: peak > 0dBFS raises `AudioQualityError`
- [ ] `AudioMixingAgent`: writes `AudioQualityRecord` with snr_db, peak_db fields
- [ ] HRG-11 display includes SNR badge (✅/❌) and clipping status
