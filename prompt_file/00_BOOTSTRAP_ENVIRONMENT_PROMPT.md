# Prompt 00: Bootstrap Environment Setup
**Category:** Infrastructure  
**File:** `bootstrap_pipeline.py` + `session_controller.py` + `download_all_models.sh`  
**Spec Reference:** RunPod_Model_Download_Specification_v6.5.md, VGA_Model_Stack_Setup_Guide_v7.2.md  
**Dependencies:** None (this is Phase 0 — executes before any code build)  
**Runtime:** On RunPod RTX 4090 pod (bootstrapped once per new pod)

---

## Context

The VGA system runs on a **RunPod RTX 4090** pod. The bootstrap layer is the foundation that all pipeline code depends on. It must:

1. Install all system dependencies (apt, pip, CUDA-aligned torch stack)
2. Create the workspace directory structure
3. Download all AI models (~170 GB total across 14 models)
4. Write the `.env_vga` environment file
5. Write a `run_manifest.json` for the session controller to detect completion

The bootstrap is **idempotent** (safe to re-run), **resumable** (continues after interruption), and **deterministic** (same output every time).

The session controller (`session_controller.py`) runs **locally on Windows** and manages the RunPod pod lifecycle (resume/stop), triggers bootstrap via SSH, and drives the pipeline via HTTP API after bootstrap completes.

---

## Implementation Requirements

### bootstrap_pipeline.py

Implement the complete `bootstrap_pipeline.py` with these 5 phases:

#### Phase 1 — System Preparation
```python
# System packages (via apt-get):
# git, ffmpeg, libgl1, libglib2.0-0, libsm6, libxext6, libxrender-dev
# curl, wget, ca-certificates, sox, libsox-dev

# Directory structure (create all, idempotent):
WORKSPACE_DIRS = [
    "/workspace/app",
    "/workspace/models/qwen", "/workspace/models/flux2",
    "/workspace/models/zimage", "/workspace/models/wan22",
    "/workspace/models/svi", "/workspace/models/svi/version-2.0",
    "/workspace/models/latentsync", "/workspace/models/cosyvoice",
    "/workspace/models/musicgen", "/workspace/models/mmaudio",
    "/workspace/loras/identity/character_main",
    "/workspace/loras/style/cinematic",
    "/workspace/loras/consistency", "/workspace/loras/svi",
    "/workspace/auxiliary/clip",
    "/workspace/identity/char_A", "/workspace/identity/char_B",
    "/workspace/motion", "/workspace/hrg/checkpoints", "/workspace/hrg/approvals",
    "/workspace/cache/huggingface", "/workspace/state", "/workspace/logs",
    "/workspace/LatentSync", "/workspace/CosyVoice",
    "/workspace/MMAudio", "/workspace/Stable-Video-Infinity",
]

# Disk check: abort if < 90 GB free; warn if < 110 GB free
```

#### Phase 2 — Dependency Installation
```python
PYTORCH_INDEX = "https://download.pytorch.org/whl/cu124"

# Install order (CRITICAL — all PyTorch from same CUDA index):
# 1. pip upgrade
# 2. torch>=2.5.1 torchvision torchaudio --index-url PYTORCH_INDEX
# 3. diffusers==0.30.0, transformers==4.45.0, accelerate==0.34.0
#    sentencepiece, "protobuf>=3.20.0,<5.0.0"
# 4. peft==0.12.0, safetensors==0.4.3
# 5. xformers --index-url PYTORCH_INDEX
# 6. ninja (for flash-attn build speedup)
# 7. flash-attn>=2.6.0 --no-build-isolation (if GPU compute >= 8.0)
# 8. open-clip-torch>=2.24.0, Pillow>=10, numpy>=1.24, opencv-python>=4.9
#    pydub>=0.25, ffmpeg-python>=0.2
# 9. psutil>=5.9, requests>=2.31, fastapi>=0.115, uvicorn>=0.30
#    pydantic>=2.0, scipy, omegaconf, einops, mediapipe, face-alignment
#    decord, soundfile, python-dotenv>=1.0, streamlit, unsloth, bitsandbytes
# 10. audiocraft (for MusicGen)
# 11. diffusers from source: pip install git+https://github.com/huggingface/diffusers.git
#     (required for ZImagePipeline + Flux2KleinPipeline)
```

#### Phase 3 — Environment Configuration
```python
# Write /workspace/.env_vga with ALL variables including:
# HF_HOME, HUGGINGFACE_HUB_CACHE, HF_HUB_ENABLE_HF_TRANSFER=1
# HF_XET_HIGH_PERFORMANCE=1, HUGGING_FACE_HUB_TOKEN
# VGA_MODELS_DIR, VGA_LORAS_DIR, VGA_ASSETS_DIR
# LORA_IDENTITY_REPO, LORA_STYLE_REPO
# SVI_LORA_REPO=vita-video-gen/svi-model
# SVI_HIGH_NOISE_FILE=version-2.0/SVI_Wan2.2-I2V-A14B_high_noise_lora_v2.0_pro.safetensors
# SVI_LOW_NOISE_FILE=version-2.0/SVI_Wan2.2-I2V-A14B_low_noise_lora_v2.0_pro.safetensors
# SVI_REPO_BRANCH=svi_wan22
# SVI_WAN22_PYTHON=/opt/conda/envs/svi_wan22/bin/python
# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# CUDA_VISIBLE_DEVICES=0
# QUALITY_CLIP_THRESHOLD=0.93
# QUALITY_MAX_RETRIES=3
# SNR_MIN_DB=10, AUDIO_PEAK_MAX_DBFS=0
# VRAM_ENFORCE_HARD_LIMIT=true
# IMMUTABLE_CONTEXT_ENFORCE=true
# IDENTITY_DRIFT_THRESHOLD=0.15
# CROSS_MODAL_SYNC_THRESHOLD=0.9
# HRG_REVIEW_ENABLED=true, HRG_APPROVAL_TIMEOUT_SECONDS=300
# TEMPORAL_BUFFER_SIZE=5
# SVI_CFG_MIN=5.0, SVI_CFG_MAX=6.0
# SCHEMA_VERSION=v6.0
```

#### Phase 4 — Model Downloads
```python
# Asset registry with (key, repo_id, local_dir, type, est_gb, validation_files):
ASSETS = [
    # Auxiliary first (small, needed by many)
    ("clip",         "openai/clip-vit-large-patch14",          "/workspace/auxiliary/clip",          "hf_snapshot", 1, ["config.json"]),
    ("consistency",  "lrzjason/Consistance_Edit_Lora",          "/workspace/loras/consistency",        "hf_snapshot", 1, []),
    
    # SVI LoRAs — targeted include downloads (NOT full repo)
    ("svi_high_noise", "vita-video-gen/svi-model",
     "/workspace/models/svi",  "hf_include", 2,
     ["version-2.0/SVI_Wan2.2-I2V-A14B_high_noise_lora_v2.0_pro.safetensors"],
     "version-2.0/SVI_Wan2.2-I2V-A14B_high_noise_lora_v2.0_pro.safetensors"),
    
    ("svi_low_noise", "vita-video-gen/svi-model",
     "/workspace/models/svi", "hf_include", 2,
     ["version-2.0/SVI_Wan2.2-I2V-A14B_low_noise_lora_v2.0_pro.safetensors"],
     "version-2.0/SVI_Wan2.2-I2V-A14B_low_noise_lora_v2.0_pro.safetensors"),
    
    # TTS + audio (medium size)
    ("cosyvoice3",   "FunAudioLLM/Fun-CosyVoice3-0.5B-2512",   
     "/workspace/CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B", "hf_snapshot", 10, ["cosyvoice3.yaml"]),
    ("musicgen",     "facebook/musicgen-medium",                 "/workspace/models/musicgen",          "hf_snapshot", 2, []),
    ("latentsync",   "ByteDance/LatentSync-1.6",                 "/workspace/LatentSync/checkpoints",   "hf_snapshot", 8, ["latentsync_unet.pt", "whisper/tiny.pt"]),
    
    # Image models
    ("zimage",       "Tongyi-MAI/Z-Image-Turbo",                 "/workspace/models/zimage",            "hf_snapshot", 16, []),
    ("qwen",         "unsloth/Qwen2.5-14B-Instruct-unsloth-bnb-4bit", "/workspace/models/qwen",        "hf_snapshot", 8, []),
    
    # Video models (large — download last)
    ("wan22",        "nalexand/Wan2.2-I2V-A14B-FP8",            "/workspace/models/wan22",             "hf_snapshot", 30, []),
    ("flux2",        "black-forest-labs/FLUX.2-klein-4B",        "/workspace/models/flux2",             "hf_snapshot", 24, ["model_index.json"]),
]

# Git clones:
GIT_REPOS = [
    ("CosyVoice", "https://github.com/FunAudioLLM/CosyVoice.git", "/workspace/CosyVoice", None),
    ("MMAudio", "https://github.com/hkchengrex/MMAudio.git", "/workspace/MMAudio", None),
    ("LatentSync", "https://github.com/bytedance/LatentSync.git", "/workspace/LatentSync", None),
    ("Wan2.2_nalexand", "https://github.com/nalexand/Wan2.2.git", "/workspace/Wan2.2_nalexand", None),
    ("SVI", "https://github.com/vita-epfl/Stable-Video-Infinity.git", "/workspace/Stable-Video-Infinity", "svi_wan22"),
]

# After git clones, install MMAudio: pip install -e /workspace/MMAudio
# After clones, install CosyVoice: pip install -r /workspace/CosyVoice/requirements.txt
# After clones, install LatentSync: pip install -r /workspace/LatentSync/requirements.txt

# Create SVI LoRA symlinks:
# /workspace/loras/svi/SVI_Wan2.2-I2V-A14B_high_noise_lora_v2.0_pro.safetensors
#   → /workspace/models/svi/version-2.0/SVI_Wan2.2-I2V-A14B_high_noise_lora_v2.0_pro.safetensors
# (same for low_noise)

# Download retry: MAX_RETRIES=3, BACKOFF=[5,15,45]
# Rolling disk headroom guard: MIN_FREE_GB=15 before each large asset
```

#### Phase 5 — Validation
```python
# Multi-layer validation (all 8 layers):
# Layer 1: Critical paths exist (CLAUDE.md, spec files, bootstrap itself)
# Layer 2: Torch + CUDA functional (torch.cuda.is_available())
# Layer 3: Model files exist with min size
# Layer 4: SVI LoRA exact filenames verified
# Layer 5: Environment variables set in .env_vga
# Layer 6: Git repos cloned to expected paths
# Layer 7: Python imports successful (clip, diffusers, audiocraft, etc.)
# Layer 8: Runtime load test (load one model, run dummy inference, unload)
```

#### Manifest Writing
```python
# /workspace/state/run_manifest.json — written:
# - At START: {"validation": "in_progress", ...} (sentinel for crash detection)
# - On SUCCESS: {
#     "validation": "passed",
#     "torch_version": torch.__version__,
#     "models": {asset_key: {"path": ..., "validated": True} for each asset},
#     "execute_stage_enforced": True,
#     "immutable_context_enforced": True,
#     "hrg_checkpoints_active": 11,
#     "temporal_loop_enforced": True,
#     "spec_version": "v6.5",
#     "schema_version": "v6.0",
#     "timestamp": "ISO-8601"
#   }
# - On FAILURE: {"validation": "failed", "error": "...", "timestamp": ...}
```

### session_controller.py

Implement the complete `session_controller.py` with:

**Responsibilities (strict):**
- Resume/stop RunPod pod via GraphQL API (`podResume`, `podStop` mutations only)
- SSH connection management (using paramiko)
- Auto-detect bootstrap completion via `run_manifest.json` content validation (not just existence)
- Inject env vars explicitly into bootstrap invocation (never rely on implicit SSH env)
- Start VGA API via uvicorn and verify `/health` with retry-safe restart loop
- Drive pipeline via HTTP API endpoints only
- Handle HRG pause → human review → pod resume → API restart → `/system/resume` lifecycle
- Structured JSON event logging with run correlation ID
- Bootstrap timeout guard: 1-hour hard limit

**NOT responsible for:**
- Creating or terminating pods
- Installing dependencies
- Writing `.env_vga`
- Downloading models
- Calling VGA code directly (API only)

**Key implementation details:**
```python
# build_bootstrap_env_exports() — builds the export string for SSH bootstrap invocation
# Includes all env vars: HUGGING_FACE_HUB_TOKEN, LORA_IDENTITY_REPO, LORA_STYLE_REPO,
# SVI_HIGH_NOISE_FILE, SVI_LOW_NOISE_FILE, SVI_REPO_BRANCH, etc.

# manifest_valid(manifest: dict) → bool
# Validates: "validation" == "passed" AND "torch_version" exists AND "models" key exists
# (NOT just file existence — content validation required)

# _api_start_with_retry() — starts uvicorn, verifies /health
# Each attempt: kill previous process, restart uvicorn, wait, check /health
# 3 attempts max, backoff [5, 15, 45] seconds

# RunPod GraphQL endpoint: https://api.runpod.io/graphql
# Only mutations: podResume and podStop (NEVER podTerminate)
```

### download_all_models.sh

Generate a standalone bash script that downloads all models in dependency order:
1. CLIP (smallest, needed first)
2. Consistance LoRA
3. SVI LoRAs (targeted --include, NOT full repo)
4. CosyVoice3
5. MusicGen medium
6. MMAudio (weights)
7. LatentSync-1.6
8. Qwen2.5-14B
9. Z-Image-Turbo
10. Wan2.2-I2V-A14B-FP8 (nalexand)
11. FLUX.2-klein-4B (largest, last)

---

## Acceptance Criteria

- [ ] `python3 bootstrap_pipeline.py` runs to completion with exit code 0
- [ ] `/workspace/state/run_manifest.json` contains `"validation": "passed"`
- [ ] SVI LoRA files exist at exact paths: `/workspace/models/svi/version-2.0/SVI_Wan2.2-I2V-A14B_{high,low}_noise_lora_v2.0_pro.safetensors`
- [ ] `python -c "import torch; assert torch.cuda.is_available()"` succeeds
- [ ] `python -c "from diffusers import ZImagePipeline"` succeeds
- [ ] Bootstrap is idempotent: second run completes without re-downloading existing files
- [ ] Session controller reads manifest content (not just existence) before skipping bootstrap
- [ ] Bootstrap timeout: session controller aborts after 3600 seconds with clear error

---

## Notes

- The SVI svi_wan22 branch requires PyTorch 2.7.1+cu128 — **separate conda env** — bootstrap installs main cu124 stack only; SVI env setup is documented separately
- LatentSync-1.6 requires HF token (gated model) — use `huggingface-cli login` or `HUGGING_FACE_HUB_TOKEN` env var
- vita-video-gen/svi-model also requires HF token
- CosyVoice requires sox + libsox-dev (apt) — bootstrap must install these
- ffmpeg must be < 7 for MMAudio compatibility (conda install -c conda-forge 'ffmpeg<7')
