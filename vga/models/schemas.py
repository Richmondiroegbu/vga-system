"""
VGA Data Contracts — All Pydantic v2 Schemas.
schema_version: "v6.0" on all v17.2 artifacts.
Spec: VGA Data Contracts Interface Specification v17.2
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ─── Composition (NEW v17.0) ──────────────────────────────────────────────────

VALID_CAMERA_ANGLES = {
    "extreme close-up", "close-up", "medium close-up", "medium shot",
    "medium wide shot", "wide shot", "extreme wide shot", "overhead",
    "low angle", "high angle", "dutch angle", "eye level",
}


class CameraAngleGroup(BaseModel):
    """One group of segments sharing the same camera angle reference image.

    Used to implement multi-reference I2V: different reference images drive
    different segment ranges, enabling motivated camera angle changes mid-video.

    Example:
        angle_groups = [
            CameraAngleGroup(image_path="angle1_front.jpg", camera_angle="medium shot",
                             segments=[1, 2], transition_mode="none"),
            CameraAngleGroup(image_path="angle2_side.jpg", camera_angle="medium wide shot",
                             segments=[3, 4], transition_mode="blend"),
            CameraAngleGroup(image_path="angle3_closeup.jpg", camera_angle="close-up",
                             segments=[5, 6], transition_mode="hard_cut"),
        ]

    transition_mode: how to transition INTO this group from the previous group.
        "none"      — first group; no transition needed.
        "hard_cut"  — Strategy A: switch reference + raise denoising_strength to 0.90.
        "blend"     — Strategy C: pixel-space cosine blend of conditioning frames (α 0→0.25).
    """

    image_path: str
    camera_angle: str
    segments: List[int]       # 1-indexed segment numbers belonging to this group
    transition_mode: str = "none"    # TransitionMode value
    schema_version: str = "v6.0"

    @field_validator("segments")
    @classmethod
    def validate_segments_nonempty(cls, v: List[int]) -> List[int]:
        if not v:
            raise ValueError("CameraAngleGroup.segments must not be empty")
        return v

    @field_validator("transition_mode")
    @classmethod
    def validate_transition_mode(cls, v: str) -> str:
        valid = {"none", "hard_cut", "blend"}
        if v not in valid:
            raise ValueError(f"transition_mode {v!r} must be one of {valid}")
        return v


class CompositionPlanSchema(BaseModel):
    """Mandatory output of SceneCompositionAgent (S-04). RULE-88. schema_version v6.0."""

    scene_id: str
    camera_angle: str        # "medium shot", "close-up", etc.
    camera_motion: str       # "slow dolly forward", "static", "pan left", etc.
    character_positions: List[Dict[str, Any]]  # [{character_id, position, facing}]
    focus_subject: str       # "main_character"
    lighting_style: str      # "low-key dramatic", "soft natural", etc.
    motion_vector: str       # "forward_slow", "stationary", "right_medium", etc.
    angle_groups: Optional[List[CameraAngleGroup]] = None  # multi-reference I2V groups
    schema_version: str = "v6.0"

    @field_validator("camera_angle")
    @classmethod
    def validate_camera_angle(cls, v: str) -> str:
        if v.lower() not in VALID_CAMERA_ANGLES:
            raise ValueError(
                f"Invalid camera_angle: {v!r}. Must be one of {VALID_CAMERA_ANGLES}"
            )
        return v.lower()

    @field_validator("character_positions")
    @classmethod
    def validate_character_positions(cls, v: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not v:
            raise ValueError("character_positions must contain at least one entry")
        for entry in v:
            if "character_id" not in entry:
                raise ValueError("Each character_positions entry must have 'character_id'")
        return v


# ─── Temporal Buffer (NEW v17.0) ─────────────────────────────────────────────

class TemporalBufferRecord(BaseModel):
    """Logged after each TemporalBuffer update. OR-034. schema_version v6.0."""

    segment_id: str
    scene_id: str
    frame_count: int = Field(5, ge=5, le=5)   # MUST be exactly 5 (RULE-86)
    timestamps: List[float] = Field(min_length=5, max_length=5)
    scene_id_ref: str
    schema_version: str = "v6.0"


# ─── Motion & Identity (NEW v17.0) ───────────────────────────────────────────

class MotionStateRecord(BaseModel):
    """Logged per segment by MotionStateTracker. OR-035."""

    segment_id: str
    scene_id: str
    velocity_magnitude: float
    direction: str           # forward/backward/left/right/stationary
    velocity_vector: Optional[List[float]] = None
    schema_version: str = "v6.0"


class IdentityStateRecord(BaseModel):
    """Logged per stage transition by IdentityStateTracker. OR-036."""

    stage_id: str
    scene_id: str
    drift_score: float
    cumulative_drift: float
    drift_history: List[float]
    threshold_exceeded: bool
    schema_version: str = "v6.0"


# ─── Audio Quality (NEW v17.0) ───────────────────────────────────────────────

class AudioQualityRecord(BaseModel):
    """Logged after AudioMixingAgent (S-15). OR-037. RULE-99."""

    scene_id: str
    snr_db: float
    peak_db: float
    clipping_detected: bool
    snr_passed: bool        # snr_db >= 10.0
    clipping_passed: bool   # peak_db <= 0.0
    schema_version: str = "v6.0"


# ─── Cross-Modal Alignment (NEW v17.0) ───────────────────────────────────────

class CrossModalAlignmentRecord(BaseModel):
    """Logged after cross-modal validation. FR-972."""

    scene_id: str
    segment_id: str
    video_duration_s: float
    audio_duration_s: float
    alignment_error_s: float
    within_tolerance: bool   # abs(alignment_error_s) <= 0.10
    schema_version: str = "v6.0"


# ─── SVI Generation (NEW v17.0) ──────────────────────────────────────────────

class SVILoraSchedule(BaseModel):
    """Dynamic LoRA weights per noise phase. RULE-86, FR-932–FR-934."""

    high_noise_weight: float = 0.6   # t > 0.67*T
    mid_noise_weight: float = 0.5    # 0.33*T < t <= 0.67*T
    low_noise_weight: float = 0.4    # t <= 0.33*T


class SVIGenerationRecord(BaseModel):
    """Logged per segment generated by TemporalEngine. schema_version v6.0."""

    segment_id: str
    scene_id: str
    cfg: float
    steps: int
    lora_schedule: SVILoraSchedule
    clip_score: float
    continuity_score: float
    retry_count: int = 0
    schema_version: str = "v6.0"

    @field_validator("cfg")
    @classmethod
    def validate_cfg(cls, v: float) -> float:
        from vga.config.settings import settings
        if not (settings.SVI_CFG_MIN <= v <= settings.SVI_CFG_MAX):
            raise ValueError(
                f"SVI CFG {v} outside allowed range [{settings.SVI_CFG_MIN}, {settings.SVI_CFG_MAX}]"
            )
        return v


# ─── Narrative Schemas ───────────────────────────────────────────────────────

class CharacterDescription(BaseModel):
    """Character visual description for script and identity design."""

    character_id: Optional[str] = None   # Qwen uses 'name' not 'character_id'
    name: str = "main_character"
    age_range: str = ""
    appearance: str = ""
    emotional_arc: str = ""
    role: str = ""                        # Qwen sometimes uses 'role' instead of other fields
    description: str = ""                 # Qwen sometimes uses 'description'

    @model_validator(mode="after")
    def ensure_character_id(self) -> "CharacterDescription":
        """Auto-generate character_id from name if Qwen didn't provide it."""
        if not self.character_id:
            self.character_id = self.name.lower().replace(" ", "_").replace("'", "")
        return self


class SceneDescription(BaseModel):
    """Scene-level description from script."""

    scene_id: Optional[str] = None
    scene_number: int = 1
    title: str = ""
    description: str = ""              # Qwen sometimes uses scene_description/summary/content
    scene_description: str = ""        # Qwen variant 1
    summary: str = ""                  # Qwen variant 2
    content: str = ""                  # Qwen variant 3
    emotional_tone: str = "hopeful"
    duration_hint_s: Optional[float] = None
    # ── Action Density (motion evaluation requirement) ────────────────────────
    # 4+ specific, coherent physical action beats covering the first ~20 seconds.
    # Each string = one segment's worth of action (~4-5 seconds).
    # Used to drive segment action_description fields in S-02 SegmentPlanSchema
    # and to evaluate how naturally the character moves vs. real human motion.
    # Required on scene_number == 1 (the opening scene); optional on later scenes.
    # Format: each string must specify BODY PART + MOVEMENT + EMOTIONAL STATE.
    # Example: "She walks briskly down the corridor, arms pumping, jaw set — pauses
    #   abruptly at a door, hand gripping the handle, breath visibly quickening"
    opening_action_sequence: List[str] = Field(
        default_factory=list,
        description="≥4 specific coherent physical action beats, each covering one ~5s segment.",
    )

    @model_validator(mode="after")
    def normalise(self) -> "SceneDescription":
        """Normalise Qwen's varying field names into standard fields.

        Also warns (non-fatal) when opening_action_sequence is missing on
        scene 1 — the pipeline continues but S-09 video quality will be harder
        to evaluate for natural motion drift.
        """
        if not self.scene_id:
            self.scene_id = f"scene_{self.scene_number:03d}"
        # Action density: scene 1 should have ≥4 action beats
        if self.scene_number == 1 and len(self.opening_action_sequence) < 4:
            import logging as _logging
            _logging.getLogger(__name__).warning(
                "SceneDescription scene_001: opening_action_sequence has %d items "
                "(need ≥4 for motion quality evaluation). Script prompt may need re-run.",
                len(self.opening_action_sequence),
            )
        # Populate description from any synonym Qwen used
        if not self.description:
            self.description = (
                self.scene_description or self.summary or self.content
                or self.title or f"Scene {self.scene_number}"
            )
        return self


class ScriptSchema(BaseModel):
    """Output of ScriptAgent (S-01). schema_version v6.0."""

    job_id: str = ""
    title: str = "Untitled Story"
    logline: str = ""
    characters: List[CharacterDescription] = []
    scenes: List[SceneDescription] = []
    total_duration_estimate_s: float = 60.0
    schema_version: str = "v6.0"

    @model_validator(mode="after")
    def force_schema_version(self) -> "ScriptSchema":
        """Always use v6.0 regardless of what the LLM outputs."""
        self.schema_version = "v6.0"
        return self


class SegmentPlanSchema(BaseModel):
    """Plan for a single video segment within a scene."""

    segment_id: str
    scene_id: str
    segment_number: int
    duration_s: float = Field(ge=3.0, le=5.0)   # RULE: segments 3–5 seconds
    dialogue: Optional[str] = None
    action_description: str
    camera_instruction: str
    # FLF2V: optional path to a pre-generated end frame image for this segment.
    # When set, the SVI inference bridge injects it as an endpoint constraint,
    # forcing the model to arrive at this visual state by the segment's final frame.
    # Generated by EndFrameGenerator (S-07c) using FLUX.2-klein from action_description.
    end_frame_path: Optional[str] = None
    schema_version: str = "v6.0"


class ScenePlanSchema(BaseModel):
    """Output of ScenePlanner (S-02). schema_version v6.0."""

    job_id: str
    scene_id: str
    scene_number: int
    duration_s: float = Field(ge=10.0, le=30.0)  # RULE: scenes 10–30 seconds
    segments: List[SegmentPlanSchema]
    setting: str
    characters_present: List[str]
    emotional_beat: str
    schema_version: str = "v6.0"

    @model_validator(mode="after")
    def validate_segment_count(self) -> "ScenePlanSchema":
        total = sum(seg.duration_s for seg in self.segments)
        if abs(total - self.duration_s) > 1.0:
            raise ValueError(
                f"Segment durations sum {total:.1f}s doesn't match scene duration {self.duration_s:.1f}s"
            )
        return self


class IdentityDesignSchema(BaseModel):
    """Output of IdentityDesignAgent (S-03). schema_version v6.0."""

    job_id: str
    scene_id: str
    character_id: str
    character_identity: str          # detailed visual prompt for character
    environment_description: str     # scene environment prompt
    reference_strategy: str          # MANDATORY — how to maintain visual consistency
    negative_prompt: str
    schema_version: str = "v6.0"


# ─── Continuity & Video ───────────────────────────────────────────────────────

class ContinuityReport(BaseModel):
    """Output of ContinuityValidationAgent (S-10). schema_version v6.0."""

    scene_id: str
    overall_continuity_score: float
    motion_continuity: float
    lighting_continuity: float
    identity_continuity: float
    identity_per_segment: List[float]    # CLIP score per segment (NEW v17.0)
    segments_validated: int
    passed: bool                          # overall_continuity_score >= 0.90
    schema_version: str = "v6.0"


class VideoSegmentArtifact(BaseModel):
    """A generated video segment artifact."""

    segment_id: str
    scene_id: str
    segment_number: int
    file_path: str
    duration_s: float
    keyframe_path: str
    clip_score: Optional[float] = None
    continuity_score: Optional[float] = None
    schema_version: str = "v6.0"


# ─── HRG Display Data (NEW v17.0) ────────────────────────────────────────────

class HRG2DisplayData(BaseModel):
    """Data shown in HRG-2 (scene/segment plan review). NEW v17.0."""

    job_id: str
    scenes: List[ScenePlanSchema]
    total_segments: int
    total_duration_estimate_s: float
    schema_version: str = "v6.0"


class HRG4DisplayData(BaseModel):
    """Data shown in HRG-4 (CompositionPlan review). NEW v17.0."""

    scene_id: str
    composition_plan: CompositionPlanSchema
    editable_fields: List[str] = [
        "camera_angle", "camera_motion", "character_positions",
        "focus_subject", "lighting_style", "motion_vector"
    ]
    schema_version: str = "v6.0"


class HRG8DisplayData(BaseModel):
    """Data shown in HRG-8 (motion QA). Updated v17.0 with identity_per_segment."""

    scene_id: str
    video_segment_paths: List[str]
    continuity_score: float
    motion_state_summary: Optional[Dict[str, Any]] = None
    identity_per_segment: List[float]    # NEW v17.0 — CLIP per segment
    passed: bool
    schema_version: str = "v6.0"


class HRG10DisplayData(BaseModel):
    """Data shown in HRG-10 (lip sync QA). Updated v17.0 with identity_delta."""

    scene_id: str
    lip_sync_segment_paths: List[str]
    phoneme_alignment_score: float
    identity_delta_per_segment: List[float]   # NEW v17.0 — drift per segment
    all_within_threshold: bool                # all deltas <= 0.03
    schema_version: str = "v6.0"


class HRG11DisplayData(BaseModel):
    """Data shown in HRG-11 (final audio QA). Updated v17.0 with SNR/clipping."""

    scene_id: str
    audio_path: str
    snr_db: float                # NEW v17.0
    peak_db: float               # NEW v17.0
    clipping_detected: bool      # NEW v17.0
    snr_passed: bool
    clipping_passed: bool
    mixing_levels: Dict[str, float]
    schema_version: str = "v6.0"


# ─── API Response Schemas ─────────────────────────────────────────────────────

class TemporalBufferStatusResponse(BaseModel):
    """API response for temporal buffer status endpoint."""

    scene_id: str
    segment_id: Optional[str] = None
    frame_count: int
    is_initialized: bool
    last_update_timestamp: Optional[float] = None
    schema_version: str = "v6.0"


class IdentityStateResponse(BaseModel):
    """API response for identity state endpoint."""

    scene_id: str
    is_frozen: bool
    drift_score: float
    cumulative_drift: float
    threshold: float
    drift_history: List[float]
    schema_version: str = "v6.0"


class AudioValidationResponse(BaseModel):
    """API response for audio validation endpoint."""

    scene_id: str
    record: AudioQualityRecord
    validation_passed: bool
    schema_version: str = "v6.0"


class CompositionPlanUpdateRequest(BaseModel):
    """Request body for HRG-4 CompositionPlan update (human edits during review)."""

    scene_id: str
    camera_angle: Optional[str] = None
    camera_motion: Optional[str] = None
    character_positions: Optional[List[Dict[str, Any]]] = None
    focus_subject: Optional[str] = None
    lighting_style: Optional[str] = None
    motion_vector: Optional[str] = None


# ─── Pipeline Report ─────────────────────────────────────────────────────────

class RuleComplianceRecord(BaseModel):
    """Per-rule compliance status for the pipeline report."""

    rule_id: str
    description: str
    compliant: bool
    details: Optional[str] = None


class PipelineReport(BaseModel):
    """Final pipeline report. Output of QualityAgent (S-16c). schema_version v6.0."""

    job_id: str
    scene_id: str
    success: bool
    total_duration_s: float

    # Stage SLA records
    stage_durations: Dict[str, float]

    # Composition (v17.0)
    composition_plan_summary: Optional[Dict[str, Any]] = None

    # Temporal (v17.0)
    temporal_engine_health: Optional[Dict[str, Any]] = None
    motion_state_summary: Optional[Dict[str, Any]] = None
    identity_per_segment_video: Optional[List[float]] = None

    # Identity (v17.0)
    identity_state_final: Optional[IdentityStateRecord] = None
    identity_delta_per_segment_sync: Optional[List[float]] = None

    # Audio (v17.0)
    audio_quality_summary: Optional[AudioQualityRecord] = None
    cross_modal_alignment_summary: Optional[CrossModalAlignmentRecord] = None

    # HRG (v17.0)
    hrg_checkpoint_count: int = 11
    hrg_outcomes: Dict[str, str] = {}   # checkpoint → "approved"/"rejected"

    # Rule compliance (v17.0)
    rule_compliance: List[RuleComplianceRecord] = []

    # Artifacts
    output_video_path: Optional[str] = None
    schema_version: str = "v6.0"
