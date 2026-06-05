"""
VGA v17.2 Configuration Settings.
Single source of truth for ALL system constants.
Spec: VGA SRD v17.2 §5, §16; VGA Coding Standards RULE-01 through RULE-05.
"""
from __future__ import annotations

import os
from pathlib import Path

from pydantic_settings import BaseSettings


class VGASettings(BaseSettings):
    """All VGA runtime constants. Loaded from /workspace/.env_vga on RunPod."""

    # === Core Identity ===
    SCHEMA_VERSION: str = "v6.0"
    SYSTEM_VERSION: str = "17.2.0"
    MISSION: str = "Inspire audiences by telling stories of people who overcame adversity"

    # === Workspace Paths ===
    WORKSPACE_ROOT: Path = Path("/workspace")
    MODELS_DIR: Path = Path("/workspace/models")
    LORAS_DIR: Path = Path("/workspace/loras")
    ASSETS_DIR: Path = Path("/workspace/assets")
    OUTPUT_DIR: Path = Path("/workspace/output")
    LOGS_DIR: Path = Path("/workspace/logs")
    STATE_DIR: Path = Path("/workspace/state")
    HRG_DIR: Path = Path("/workspace/hrg")

    # === Model Paths ===
    QWEN_MODEL_PATH: Path = Path("/workspace/models/qwen")
    FLUX2_MODEL_PATH: Path = Path("/workspace/models/flux2")
    ZIMAGE_MODEL_PATH: Path = Path("/workspace/models/zimage")
    WAN22_MODEL_PATH: Path = Path("/workspace/models/wan22")
    SVI_MODEL_PATH: Path = Path("/workspace/models/svi")
    LATENTSYNC_PATH: Path = Path("/workspace/LatentSync")
    COSYVOICE_PATH: Path = Path("/workspace/CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B")
    MUSICGEN_MODEL_PATH: Path = Path("/workspace/models/musicgen")
    MMAUDIO_PATH: Path = Path("/workspace/MMAudio")
    CLIP_MODEL_PATH: Path = Path("/workspace/auxiliary/clip")
    SVI_REPO_PATH: Path = Path("/workspace/Stable-Video-Infinity")

    # === LoRA Paths ===
    SVI_HIGH_NOISE_PATH: Path = Path(
        "/workspace/loras/svi/SVI_Wan2.2-I2V-A14B_high_noise_lora_v2.0_pro.safetensors"
    )
    SVI_LOW_NOISE_PATH: Path = Path(
        "/workspace/loras/svi/SVI_Wan2.2-I2V-A14B_low_noise_lora_v2.0_pro.safetensors"
    )
    CONSISTENCY_LORA_PATH: Path = Path("/workspace/loras/consistency")
    # Verified correct filename for FLUX.2-klein-4B (f2k = FLUX.2-klein, 4B variant).
    # Full repo has 3.47 GB; bootstrap downloads only this 385 MB file.
    CONSISTENCY_LORA_FILENAME: str = "f2k_4B_consist_20260314.safetensors"

    # === SVI Environment ===
    SVI_REPO_BRANCH: str = "svi_wan22"
    SVI_WAN22_PYTHON: str = "/usr/bin/python3"  # system Python with DiffSynth installed
    # RTX 5090 is Blackwell sm_120. PyTorch 2.7.1+cu128 is confirmed working
    # for SVI (flash_attn==2.8.0.post2 tested with this combination).
    # PyTorch 2.8.0 also works. Minimum is 2.7.0 (first stable sm_120 release).
    SVI_WAN22_TORCH_VERSION: str = "2.7.1"
    SVI_WAN22_CUDA_INDEX: str = "cu128"

    # === SVI Noise Schedule ===
    # sigma_shift controls the noise schedule distribution.
    # For base Wan2.2 I2V (S-08): 5.0 — official Wan2.2 recommendation for I2V.
    # For SVI continuation (S-09+): 7.0 — SVI 2.0 Pro community recommendation.
    # Using 5.0 (I2V value) for SVI continuation misaligns the schedule against the
    # SVI LoRA's training distribution, producing systematically softer/hazier frames.
    SVI_SIGMA_SHIFT_I2V: float = 5.0           # S-08 base I2V generation
    SVI_SIGMA_SHIFT_CONTINUATION: float = 7.0  # S-09+ SVI autoregressive continuation

    # === SVI Persistent Server (speed optimisation) ===
    # SVIWrapper starts vga_svi_server.py as a background daemon on first segment
    # call. All subsequent segments POST to the warm server instead of spawning a
    # fresh subprocess (saves 3-5 min cold load per segment).
    SVI_SERVER_PORT: int = 8765

    # GPU-resident DiT mode: keeps both DiTs on GPU VRAM instead of offloading.
    # RTX PRO 6000 96GB + BF16: T5(9.3GB) + DiT-high(~15GB) + DiT-low(~15GB) = ~39GB.
    # All models fit GPU-resident — no CPU offloading required.
    SVI_GPU_RESIDENT_DITS: bool = True

    # === Wan2.2 BF16 Base Model (RTX PRO 6000 upgrade) ===
    # Full BF16 precision — eliminates FP8 quantization artifacts, better temporal coherence.
    # Download: huggingface-cli download Wan-AI/Wan2.2-I2V-A14B --local-dir /workspace/models/wan22_bf16
    WAN22_BF16_MODEL_PATH: Path = Path("/workspace/models/wan22_bf16")
    # "bf16" = WAN22_BF16_MODEL_PATH, standard BF16 loader, no apply_vram_management (RTX PRO 6000).
    # "fp8" = WAN22_MODEL_PATH, custom FP8 split-block loader (legacy 32GB path).
    SVI_WAN22_PRECISION: str = "bf16"

    # === Identity Thresholds ===
    CLIP_IDENTITY_THRESHOLD: float = 0.93        # RULE-92: minimum CLIP score everywhere
    CLIP_SCENE_EXPANDED_MINIMUM: float = 0.55   # floor for scene-expanded images (wide shot vs close-up portrait embedding naturally scores lower)
    CLIP_VIDEO_SEGMENT_MINIMUM: float = 0.75    # soft floor for SVI video segment keyframes (warning-only)
    CLIP_I2V_HARD_FLOOR: float = 0.15          # hard floor for I2V segments (S-08, S-09); scene frames naturally score low vs close-up ref
    CLIP_S08_I2V_MINIMUM: float = 0.15         # alias for backward compat
    CLIP_DRIFT_THRESHOLD: float = 0.02           # RULE-93: max drift in image refinement
    IDENTITY_CUMULATIVE_DRIFT_THRESHOLD: float = 0.15   # Full regeneration trigger
    LIPSYNC_IDENTITY_DELTA_THRESHOLD: float = 0.03      # RULE-97
    IDENTITY_MAX_PHASE_REGENERATIONS: int = 1

    # === Temporal Engine (RULE-86, RULE-87) ===
    TEMPORAL_BUFFER_SIZE: int = 5                # MUST be 5, always
    TEMPORAL_MAX_RETRIES_PER_SEGMENT: int = 3
    SEGMENT_CONTINUITY_MIN: float = 0.85

    # === SVI CFG + Steps (FR-936, FR-937) ===
    SVI_CFG_MIN: float = 5.0
    SVI_CFG_MAX: float = 8.0   # raised from 6.0; vita-epfl recommends 7.0 for non-distill SVI
    SVI_CFG_DEFAULT: float = 7.0
    STEPS_CRITICAL: int = 50     # production quality — keep high for final renders
    STEPS_STANDARD: int = 30    # reduced from 50; BF16 GPU-resident on RTX PRO 6000 = faster per step

    # === SVI LoRA Scheduling (FR-932–FR-934) ===
    LORA_WEIGHT_HIGH_NOISE: float = 0.6          # t > HIGH_NOISE_FRACTION * T
    LORA_WEIGHT_MID_NOISE: float = 0.5           # MID_NOISE_FRACTION < t <= HIGH_NOISE_FRACTION
    LORA_WEIGHT_LOW_NOISE: float = 0.4           # t <= MID_NOISE_FRACTION * T
    HIGH_NOISE_FRACTION: float = 0.67
    MID_NOISE_FRACTION: float = 0.33

    # === Camera Angle Transition (multi-reference I2V) ===
    # Strategy A (hard cut): model gets maximum freedom to adopt new viewpoint.
    TRANSITION_HARD_CUT_DS: float = 0.90
    # Strategy C (pixel-space blend): softer — old motion partially guides model.
    TRANSITION_BLEND_DS: float = 0.80
    # Max blend alpha on the final conditioning frame (cosine ramp 0→this value).
    # Research confirms safe range is ≤ 0.30 for semantically similar source/dest.
    TRANSITION_BLEND_MAX_ALPHA: float = 0.25

    # === FLF2V — First-Last Frame to Video (Phase 1: concat_mask on existing weights) ===
    # Conditions segment generation on a pre-chosen end frame, constraining the model
    # to arrive at a target visual state. Solves static character and narrative drift.
    # Phase 1: injects end_image via WanVideoSviPipeline kwarg (graceful fallback if
    # pipeline does not support it). No new model weights required.
    FLF2V_ENABLED: bool = False          # feature flag — enable after pod validation
    FLF2V_END_FRAME_MODEL: str = "flux2" # "flux2" | "wan_t2i" — end frame generator
    # Z-Image-Turbo polish on each FLUX-generated end frame (S-07c).
    # denoise=0.10 — same conservative strength as S-07 character refinement.
    # Sharpens edges/texture without altering scene composition or identity.
    FLF2V_ZIMAGE_POLISH: bool = True
    FLF2V_ZIMAGE_DENOISE: float = 0.10
    # WanCutLastSlot: number of frames to trim from tail of a FLF2V segment before
    # using it as SVI continuation conditioning for the next segment.
    # The end-frame-locked tail region causes motion-steering conflicts at the seam.
    # Wan2.2 VAE temporal stride = 4, so one temporal slot = 4 pixel-space frames.
    FLF2V_WANCUT_SLOT_FRAMES: int = 4

    # === Motion ===
    MOTION_STATIONARY_THRESHOLD: float = 0.02

    # === Scene Composition ===
    COMPOSITION_MAX_RETRIES: int = 3
    SLA_COMPOSITION_MAX_S: float = 15.0

    # === Audio Quality (RULE-99) ===
    MIN_SNR_DB: float = 10.0
    MAX_PEAK_DBFS: float = 0.0
    HEADROOM_DB: float = 1.0
    AUDIO_QUALITY_MAX_RETRIES: int = 3

    # === Audio Mixing Levels ===
    DIALOGUE_LEVEL_DB: float = 0.0
    AMBIENT_LEVEL_DB: float = -12.0
    MUSIC_LEVEL_DB: float = -18.0
    DUCKING_DB: float = -6.0     # ambient/music reduction during dialogue

    # === Cross-Modal Alignment (FR-972) ===
    CROSS_MODAL_SYNC_THRESHOLD: float = 0.9
    TIMING_TOLERANCE_S: float = 0.10

    # === HRG ===
    HRG_CHECKPOINT_COUNT: int = 11
    HRG_REVIEW_ENABLED: bool = True
    HRG_APPROVAL_TIMEOUT_SECONDS: int = 300

    # === Continuity Scoring Weights ===
    CONTINUITY_MOTION_WEIGHT: float = 0.40
    CONTINUITY_LIGHTING_WEIGHT: float = 0.30
    CONTINUITY_IDENTITY_WEIGHT: float = 0.30

    # === Video / Image Quality ===
    BASE_IMAGE_COUNT: int = 6
    SCENE_DURATION_MIN_S: float = 10.0
    SCENE_DURATION_MAX_S: float = 30.0
    SEGMENT_DURATION_MIN_S: float = 3.0
    SEGMENT_DURATION_MAX_S: float = 5.0
    IMAGE_CLIP_SCORE_MIN: float = 0.93

    # === FLUX.2-klein Settings ===
    FLUX_IDENTITY_LORA_WEIGHT_MIN: float = 0.4
    FLUX_IDENTITY_LORA_WEIGHT_MAX: float = 0.7
    FLUX_CFG: float = 1.0
    FLUX_STEPS: int = 4

    # === Z-Image-Turbo Settings ===
    ZIMAGE_DENOISE_MIN: float = 0.05
    ZIMAGE_DENOISE_MAX: float = 0.15
    ZIMAGE_CFG: float = 5.0

    # === VRAM ===
    VRAM_ENFORCE_HARD_LIMIT: bool = True
    VRAM_FREE_RATIO_MIN: float = 0.90    # After GPU cleanup, assert ≥ this

    # === Retry / Backoff ===
    MAX_RETRIES: int = 3
    BACKOFF_SECONDS: list = [5, 15, 45]

    # === MMAudio Model Selection ===
    # Switched from large_44k_v2 (4.12 GB) to medium_44k (2.49 GB) to save disk space.
    # medium_44k still outputs 44.1kHz audio — same sample rate as large.
    # Ambient audio is mixed at -12dB under dialogue so quality difference is inaudible.
    MMAUDIO_MODEL_NAME: str = "medium_44k"
    MMAUDIO_SAMPLE_RATE: int = 44100       # 44.1kHz for all _44k model variants

    # === Immutable Context ===
    IMMUTABLE_CONTEXT_ENFORCE: bool = True

    # === SLA KPIs ===
    SLA_SCRIPT_MAX_S: float = 30.0
    SLA_SCENE_PLAN_MAX_S: float = 10.0
    SLA_IDENTITY_DESIGN_MAX_S: float = 20.0
    SLA_BASE_IMAGE_MAX_S: float = 60.0
    SLA_SEGMENT_GEN_MAX_S: float = 120.0
    SLA_SEGMENT_GEN_CRITICAL_MAX_S: float = 300.0
    SLA_LIPSYNC_MAX_S: float = 30.0
    SLA_AUDIO_MIX_MAX_S: float = 10.0
    SLA_EXPORT_MAX_S: float = 30.0

    model_config = {
        "env_file": "/workspace/.env_vga",
        "env_file_encoding": "utf-8",
        "case_sensitive": True,
        "extra": "ignore",
    }


# Singleton instance — import this everywhere
settings = VGASettings()
