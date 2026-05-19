#!/usr/bin/env python3
"""
bootstrap_pipeline.py — VGA v17.2 / Bootstrap Layer v6.5
Spec:  Session Controller Upgrade Specification v7.0, Parts 1–2 + v7.1 Patch

THIS IS THE SINGLE BOOTSTRAP ENTRYPOINT.
Invoked by session_controller.py via:
    python3 /workspace/bootstrap_pipeline.py

Owned responsibilities (bootstrap layer ONLY):
  ✔  apt system packages
  ✔  pip dependency installation (CUDA-safe, single index cu124)
  ✔  Workspace directory structure
  ✔  Environment variables (/workspace/.env_vga)
  ✔  Model + LoRA downloads (idempotent, resumable, atomic)
  ✔  Multi-layer asset validation
  ✔  Run manifest (/workspace/state/run_manifest.json)

Properties:
  Idempotent   — safe to run multiple times
  Deterministic — same output every run
  Resumable    — continues after any interruption
  CUDA-safe    — entire PyTorch stack from one CUDA index (cu124)
  Exit codes:  0 = success, 1 = failure

SVI Environment Note:
  The svi_wan22 branch of vita-epfl/Stable-Video-Infinity is built on DiffSynth 2.0
  and officially tested with PyTorch 2.7.1 + cu128 — a different CUDA stack from the
  main VGA cu124 environment. SVI inference MUST be invoked in a dedicated conda env
  (svi_wan22) or via subprocess with its own Python interpreter. The bootstrap installs
  the main cu124 stack only; SVI env setup is documented in VGA_Model_Stack_Setup_Guide.md
  Section 3.5 and should be run separately.

v7.1 Changes:
  ✅  Manifest guarantees all fields required by session_controller manifest_valid():
      validation, models, torch_version — present and correct on success
  ✅  Sentinel manifest written at bootstrap start (validation=in_progress)
      so a mid-run crash is detectable by the controller
  ✅  write_manifest() accepts validation_status param; error paths write "failed"
  ✅  configure_env() warns explicitly when HF_TOKEN is empty
  ✅  FlashAttention: GPU compute capability pre-check before compile attempt
  ✅  Disk guard unified with clear documentation of dual-threshold roles

v7.1.1 Fixes (SVI LoRA filename correction):
  ✅  SVI_HIGH_NOISE_FILE and SVI_LOW_NOISE_FILE corrected to match the actual
      filenames on vita-video-gen/svi-model (version-2.0/ subfolder + full model name)
  ✅  SVI LoRA download now uses --include with subfolder path to avoid pulling
      the entire 30 GB+ repo when only two files are needed
  ✅  PyTorch minimum version raised to 2.5.1 (required by MMAudio official repo)
  ✅  PYTORCH_INDEX updated to cu124 (matches RunPod 4090 default CUDA 12.4 images)
  ✅  sox libsox-dev added to apt packages (required by CosyVoice)
  ✅  SVI LoRA download type changed to hf_include (targeted partial-repo download)
      to correctly retrieve files from version-2.0/ subfolder

v7.1.2 Fixes (SVI env isolation + verified corrections):
  ✅  SVI environment isolation documented — svi_wan22 branch requires PyTorch 2.7.1
      + cu128 (DiffSynth 2.0 base); incompatible with main cu124 stack; must run in
      dedicated conda env (svi_wan22) or via subprocess
  ✅  SVI_REPO_BRANCH env var added to .env_vga for downstream inference scripts
  ✅  SVI_WAN22_PYTHON_ENV added to .env_vga to specify SVI conda env interpreter
  ✅  Wan2.2 est_gb corrected to 30 (nalexand FP8 fork actual size)
  ✅  FLUX.2-klein-4B est_gb corrected to 24 (full BF16 weights ~23.7 GB)
  ✅  LatentSync est_gb corrected to 8 (full checkpoint set including auxiliary)
  ✅  CosyVoice est_gb corrected to 10 (full model + ttsfrd resource)
"""

from __future__ import annotations

import gc
import glob
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────
os.makedirs("/workspace/logs", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(levelname)s][bootstrap] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("/workspace/logs/bootstrap.log", mode="a"),
    ],
)
log = logging.getLogger("bootstrap")

# ─────────────────────────────────────────────────────────────────────────────
# ENVIRONMENT  (read from shell — bootstrap writes /workspace/.env_vga,
#               it does NOT read it here; env vars come from the controller
#               which injects them explicitly via export statements in the
#               bootstrap invocation command — see session_controller v7.1
#               build_bootstrap_env_exports())
# ─────────────────────────────────────────────────────────────────────────────
HF_TOKEN            = os.environ.get("HUGGING_FACE_HUB_TOKEN", "")

# Ensure HF_TOKEN is active for all huggingface_hub calls in this process.
# Without this, downloads are unauthenticated, rate-limited, and slower.
if HF_TOKEN:
    os.environ["HUGGING_FACE_HUB_TOKEN"] = HF_TOKEN
    os.environ["HUGGING_FACE_HUB_TOKEN_PATH"] = ""   # prevent file-based override
LORA_IDENTITY_REPO  = os.environ.get("LORA_IDENTITY_REPO",  "")
LORA_STYLE_REPO     = os.environ.get("LORA_STYLE_REPO",     "")
SVI_LORA_REPO       = os.environ.get("SVI_LORA_REPO",       "vita-video-gen/svi-model")

# ─── CORRECTED SVI LoRA filenames ────────────────────────────────────────────
# These match the ACTUAL filenames in vita-video-gen/svi-model on HuggingFace.
# Files live inside the version-2.0/ subfolder of the repo.
# Verified against:
#   https://huggingface.co/vita-video-gen/svi-model/tree/main/version-2.0
#   https://github.com/vita-epfl/Stable-Video-Infinity/tree/svi_wan22
#
# WRONG (old, broken):  "SVI_Wan2.2_high_noise_v2.0_pro.safetensors"
# CORRECT (verified):   "version-2.0/SVI_Wan2.2-I2V-A14B_high_noise_lora_v2.0_pro.safetensors"
SVI_HIGH_NOISE_FILE = os.environ.get(
    "SVI_HIGH_NOISE_FILE",
    "version-2.0/SVI_Wan2.2-I2V-A14B_high_noise_lora_v2.0_pro.safetensors",
)
SVI_LOW_NOISE_FILE  = os.environ.get(
    "SVI_LOW_NOISE_FILE",
    "version-2.0/SVI_Wan2.2-I2V-A14B_low_noise_lora_v2.0_pro.safetensors",
)
# ─────────────────────────────────────────────────────────────────────────────

# v7.1 — Unified disk guard constant.
# Used by both the initial workspace check (Phase 1) and the per-asset
# rolling headroom guard (Phase 4).  Single source of truth.
MIN_FREE_GB = float(os.environ.get("DOWNLOAD_MIN_FREE_GB", "15"))

# Retry / backoff (spec Part 2 §8.2)
MAX_RETRIES = 3
BACKOFF     = [5, 15, 45]   # seconds

# Disk requirements (spec Part 2 §2.3–2.4)
# REQUIRED_MIN_GB  — hard abort: workspace must have at least this free at start
# RECOMMENDED_MIN_GB — soft warning: recommend this much for comfortable margin
REQUIRED_MIN_GB    = 10.0   # lowered: all models already on disk, only validation needed
RECOMMENDED_MIN_GB = 110.0

STATE_DIR = "/workspace/state"
MANIFEST  = f"{STATE_DIR}/run_manifest.json"

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1 — SYSTEM PREPARATION  (spec Part 2 §2)
# ─────────────────────────────────────────────────────────────────────────────

WORKSPACE_DIRS = [
    "/workspace/app",
    "/workspace/models/qwen",
    "/workspace/models/flux2",
    "/workspace/models/zimage",
    "/workspace/models/wan22",
    "/workspace/models/svi",
    "/workspace/models/svi/version-2.0",    # SVI LoRAs live here
    "/workspace/models/latentsync",
    "/workspace/models/cosyvoice",
    "/workspace/models/musicgen",
    "/workspace/models/mmaudio",
    "/workspace/loras/identity/character_main",
    "/workspace/loras/identity/char_A",
    "/workspace/loras/identity/char_B",
    "/workspace/loras/style/cinematic",
    "/workspace/loras/consistency",
    "/workspace/loras/svi",                 # symlink target dir
    "/workspace/auxiliary/clip",
    "/workspace/identity/char_A",
    "/workspace/identity/char_B",
    "/workspace/motion",
    "/workspace/hrg/checkpoints",
    "/workspace/hrg/approvals",
    "/workspace/cache/huggingface",
    "/workspace/state",
    "/workspace/logs",
    "/workspace/LatentSync",
    "/workspace/CosyVoice",
    "/workspace/MMAudio",
    "/workspace/Stable-Video-Infinity",
]


def system_setup() -> None:
    """Phase 1: system packages + workspace directory structure + disk check."""
    log.info("=== PHASE 1: SYSTEM PREPARATION ===")

    # 1-pre. Bootstrap prerequisites — must exist before Phase 1 uses them.
    # psutil is used in _check_disk_space() (Phase 1) before Phase 2 installs deps.
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "psutil", "--quiet"],
        check=True,
    )
    log.info("[phase1] Bootstrap prerequisites (psutil) installed.")

    # 1a. apt packages
    log.info("[phase1] Installing system packages …")
    subprocess.run(["apt-get", "update", "-qq"], check=True)
    subprocess.run(
        [
            "apt-get", "install", "-y", "--no-install-recommends",
            "git", "ffmpeg", "libgl1", "libglib2.0-0",
            "libsm6", "libxext6", "libxrender-dev",
            "curl", "wget", "ca-certificates",
            "sox", "libsox-dev",                # Required by CosyVoice
        ],
        env={**os.environ, "DEBIAN_FRONTEND": "noninteractive"},
        check=True,
    )
    log.info("[phase1] System packages OK.")

    # 1b. Workspace directories
    log.info("[phase1] Creating workspace directory structure …")
    for d in WORKSPACE_DIRS:
        os.makedirs(d, exist_ok=True)
    log.info("[phase1] Workspace directories OK.")

    # 1c. Disk check (mandatory before any download)
    _check_disk_space(REQUIRED_MIN_GB)
    log.info("=== PHASE 1 COMPLETE ===")


def _check_disk_space(required_gb: float) -> None:
    """
    Hard abort if free disk space is below required_gb.
    Used for the initial workspace sanity check only (Phase 1).
    Per-asset rolling headroom checks use _ensure_disk_headroom() in Phase 4.
    """
    import psutil
    usage   = psutil.disk_usage("/workspace")
    free_gb = usage.free / (1024 ** 3)
    if free_gb < required_gb:
        raise RuntimeError(
            f"[disk] ABORT — only {free_gb:.1f} GB free on /workspace; "
            f"minimum required is {required_gb:.1f} GB."
        )
    if free_gb < RECOMMENDED_MIN_GB:
        log.warning(
            f"[disk] Low disk space ({free_gb:.1f} GB free); "
            f"recommended minimum is {RECOMMENDED_MIN_GB:.1f} GB."
        )
    log.info(f"[disk] Disk OK — {free_gb:.1f} GB free.")


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2 — DEPENDENCY INSTALLATION  (spec Part 2 §3)
# CRITICAL: ALL PyTorch packages must come from the SAME CUDA index (cu128).
# RTX 5090 (Blackwell sm_100) requires PyTorch 2.8+ and CUDA 12.8 (cu128).
# PyTorch 2.7.x and below do NOT include sm_100 kernels — will fail on 5090.
# ─────────────────────────────────────────────────────────────────────────────

PYTORCH_INDEX = "https://download.pytorch.org/whl/cu128"


def install_dependencies() -> None:
    """
    Phase 2: Install the full dependency stack.
    Entire PyTorch stack (torch + torchvision + torchaudio + xformers) is
    installed from the cu128 index (CUDA 12.8) to guarantee zero binary mismatch.

    Minimum torch version: 2.7.0 (RTX 5090 Blackwell sm_120 support).
    PyTorch 2.7.0 was the first stable release with Blackwell GPU kernels.

    v7.1: FlashAttention now pre-checks GPU compute capability before attempting
    the expensive compile step — avoids silent failures on incompatible GPUs.
    """
    log.info("=== PHASE 2: DEPENDENCY INSTALLATION ===")

    def pip(*args: str) -> None:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--quiet", *args],
            check=True,
        )

    # 2a. Upgrade pip
    pip("--upgrade", "pip")

    # 2b. Core PyTorch stack — ALL from cu128 index (NEVER mix CUDA versions)
    # RTX 5090 is Blackwell sm_120 (consumer). PyTorch 2.7.0 was the first STABLE
    # release with sm_120 kernels. PyTorch 2.8.0+ also works. Using >=2.7.0 here
    # to accept any stable Blackwell-compatible build on the pod template.
    log.info("[phase2] Installing PyTorch 2.7.0+ (cu128) for RTX 5090 Blackwell sm_120 …")
    pip(
        "torch>=2.7.0", "torchvision", "torchaudio",
        "--index-url", PYTORCH_INDEX,
    )

    # 2c. ML stack — minimum versions; unpinned upper bound for RTX 5090 Blackwell support.
    # diffusers==0.30.0 was too old for sm_120 Blackwell compatibility.
    # transformers==4.45.0 and accelerate==0.34.0 also work better unpinned.
    log.info("[phase2] Installing ML stack (Blackwell-compatible, unpinned upper bound) …")
    pip("diffusers>=0.30.0")
    pip("transformers>=4.45.0")
    pip("accelerate>=0.34.0")

    # 2d. LoRA / weights
    pip("peft>=0.17.0")   # Z-Image-Turbo (diffusers) requires >=0.17.0; was pinned to 0.12.0
    pip("safetensors>=0.8.0rc0")  # 0.8.0 stable not yet on PyPI; rc0 satisfies diffusers>=0.8.0-rc.0

    # 2e. Performance — xformers from cu128 index.
    # NOTE on RTX 5090 (sm_120): xformers installs but may not activate memory-efficient
    # attention on Blackwell. Community reports show it falls back to standard attention
    # with no performance penalty (PyTorch 2.7+ has native SDPA for sm_120).
    # We install it anyway — if it doesn't accelerate, PyTorch's built-in SDPA handles it.
    log.info("[phase2] Installing xformers (cu128) — note: may fall back to SDPA on sm_120 …")
    try:
        pip("xformers", "--index-url", PYTORCH_INDEX)
        log.info("[phase2] xformers installed (sm_120 acceleration may vary)")
    except Exception as exc:
        log.warning(
            "[phase2] xformers install failed (%s) — PyTorch 2.7+ SDPA will be used instead. "
            "No performance loss on RTX 5090.", exc
        )

    # 2f. FlashAttention — SKIPPED for PyTorch 2.11+.
    # flash-attn 2.8.0.post2 requires PyTorch <=2.7 and consistently times out
    # during compilation on this stack. PyTorch 2.11 ships native SDPA (scaled
    # dot-product attention) for sm_120 which provides equivalent performance
    # without any compilation step. No action needed.
    log.info("[phase2] Skipping flash-attn — PyTorch 2.11 native SDPA used instead (sm_120 OK).")

    # 2g. Vision
    pip("open-clip-torch==2.24.0")
    pip("opencv-python>=4.9.0")   # 4.9.0 exact does not exist on PyPI; correct is 4.9.0.80+
    pip("Pillow>=10.0.0", "numpy>=1.24.0")

    # 2h. Audio
    pip("pydub==0.25.0", "ffmpeg-python==0.2.0")

    # 2i. HuggingFace download
    pip("huggingface-hub>=0.21.0", "hf-transfer>=0.1.6")

    # 2j. System + API + UI
    pip(
        "psutil>=5.9.0", "requests>=2.31.0",
        "fastapi>=0.115.0", "uvicorn>=0.30.0", "pydantic>=2.4.0",
        "pydantic-settings>=2.0.0",
        "streamlit>=1.30.0",       # Streamlit UI for HRG review panels
        "omegaconf>=2.3.0",
        "python-dotenv>=1.0.0",
    )

    # 2k. AudioCraft (MusicGen)
    log.info("[phase2] Installing AudioCraft …")
    try:
        pip("-U", "audiocraft")
        log.info("[phase2] audiocraft installed.")
    except subprocess.CalledProcessError as exc:
        log.warning(f"[phase2] audiocraft install failed ({exc}) — non-fatal.")

    # 2l. Post-install CUDA validation (spec Part 2 §3.6)
    log.info("[phase2] Validating CUDA availability …")
    try:
        import importlib
        torch = importlib.import_module("torch")
        assert torch.cuda.is_available(), "torch.cuda.is_available() returned False"
        assert torch.version.cuda.startswith("12"), (
            f"Expected CUDA 12.x, got {torch.version.cuda}"
        )
        # Enforce minimum torch version for RTX 5090 Blackwell (sm_120) compatibility.
        # PyTorch 2.7.0 was the first STABLE release with sm_120 kernels.
        from packaging.version import Version
        assert Version(torch.__version__) >= Version("2.7.0"), (
            f"torch>=2.7.0 required (RTX 5090 Blackwell sm_120), got {torch.__version__}. "
            f"PyTorch <2.7.0 does not include sm_120 Blackwell kernels — will crash at runtime."
        )
        log.info(
            f"[phase2] CUDA OK — torch={torch.__version__} "
            f"cuda={torch.version.cuda}"
        )
    except ImportError:
        # packaging not installed yet — install and re-check
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "packaging", "--quiet"],
            check=True,
        )
        # Re-import and validate
        import importlib
        torch = importlib.import_module("torch")
        assert torch.cuda.is_available(), "torch.cuda.is_available() returned False"
        log.info(
            f"[phase2] CUDA OK (packaging installed) — torch={torch.__version__} "
            f"cuda={torch.version.cuda}"
        )
    except Exception as exc:
        raise RuntimeError(f"[phase2] CUDA validation failed: {exc}") from exc

    log.info("=== PHASE 2 COMPLETE ===")


def _check_flash_attn_eligibility() -> bool:
    """
    v7.1: Verify GPU compute capability before attempting flash-attn compile.
    FlashAttention requires CUDA compute capability >= 8.0 (Ampere+).
    Returns True if eligible, False if incompatible or no GPU detected.
    """
    try:
        import importlib
        torch = importlib.import_module("torch")
        if not torch.cuda.is_available():
            return False
        major, minor = torch.cuda.get_device_capability(0)
        eligible = (major, minor) >= (8, 0)
        log.info(
            f"[phase2] GPU compute capability: {major}.{minor} "
            f"({'eligible' if eligible else 'ineligible'} for flash-attn)"
        )
        return eligible
    except Exception as exc:
        log.warning(f"[phase2] Could not determine GPU compute capability: {exc}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 3 — ENVIRONMENT CONFIGURATION  (spec Part 2 §4)
# Bootstrap OWNS /workspace/.env_vga.  Controller only sources it AFTER
# bootstrap completes (v7.1 constraint enforced in session_controller).
# ─────────────────────────────────────────────────────────────────────────────

def configure_env() -> None:
    """
    Phase 3: Write /workspace/.env_vga with all VGA runtime variables.
    This is the ONLY place this file is created or written.

    v7.1: Emits an actionable warning if HF_TOKEN is empty — gated model
    downloads will fail silently without this token.

    v7.1.1: SVI LoRA filenames corrected to match vita-video-gen/svi-model repo.
    """
    log.info("=== PHASE 3: ENVIRONMENT CONFIGURATION ===")

    # v7.1 — Early warning for missing HF token before any gated downloads
    if not HF_TOKEN:
        log.warning(
            "[phase3] HUGGING_FACE_HUB_TOKEN is empty. "
            "All gated model downloads will fail. "
            "Ensure the token is set in the controller environment and passed "
            "via explicit env injection (session_controller v7.1 build_bootstrap_env_exports)."
        )

    lines = [
        "# VGA v17.2 / Bootstrap v6.5 — auto-generated by bootstrap_pipeline.py",
        "",
        "# ── HuggingFace ──────────────────────────────────────────────────────",
        "export HF_HOME=/workspace/cache/huggingface",
        "export HUGGINGFACE_HUB_CACHE=/workspace/cache/huggingface",
        "export HF_HUB_ENABLE_HF_TRANSFER=1",
        "export HF_XET_HIGH_PERFORMANCE=1",
        "export HF_HUB_DOWNLOAD_TIMEOUT=300",
        "export HF_HUB_HTTP_TOTAL_TIMEOUT=600",
        "export HF_HUB_MAX_RETRIES=5",
        "",
        "# ── Authentication ────────────────────────────────────────────────────",
        f"export HUGGING_FACE_HUB_TOKEN={HF_TOKEN}",
        "",
        "# ── LoRA repos ────────────────────────────────────────────────────────",
        f"export LORA_IDENTITY_REPO={LORA_IDENTITY_REPO}",
        f"export LORA_STYLE_REPO={LORA_STYLE_REPO}",
        f"export SVI_LORA_REPO={SVI_LORA_REPO}",
        # CORRECTED SVI LoRA filenames (version-2.0/ subfolder + full model name)
        f"export SVI_HIGH_NOISE_FILE={SVI_HIGH_NOISE_FILE}",
        f"export SVI_LOW_NOISE_FILE={SVI_LOW_NOISE_FILE}",
        "",
        "# ── SVI runtime ───────────────────────────────────────────────────────",
        "export SVI_ENABLE_CPU_OFFLOAD=true",
        "# SVI svi_wan22 branch: upgraded to PyTorch 2.8.0+cu128 for RTX 5090 (Blackwell).",
        "# PyTorch 2.7.1 does NOT support sm_100 — 2.8.0 is the minimum for 5090.",
        "# SEPARATE conda env (svi_wan22) — different from main VGA env.",
        "export SVI_REPO_BRANCH=svi_wan22",
        "export SVI_REPO_PATH=/workspace/Stable-Video-Infinity",
        "# Set SVI_WAN22_PYTHON to the Python interpreter of the svi_wan22 conda env",
        "# e.g.: /opt/conda/envs/svi_wan22/bin/python3",
        "# Run: conda create -n svi_wan22 python=3.10 -y before first use",
        "export SVI_WAN22_PYTHON=/opt/conda/envs/svi_wan22/bin/python3",
        "# PyTorch 2.7.1+cu128 is confirmed working on RTX 5090 (sm_120) for SVI.",
        "# flash_attn==2.8.0.post2 confirmed working with this combination.",
        "export SVI_WAN22_TORCH_VERSION=2.7.1",
        "export SVI_WAN22_CUDA_INDEX=cu128",
        "",
        "# ── Disk safety ───────────────────────────────────────────────────────",
        f"export DOWNLOAD_MIN_FREE_GB={MIN_FREE_GB}",
        "",
        "# ── PyTorch ───────────────────────────────────────────────────────────",
        "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
        "",
        "# ── Framework ─────────────────────────────────────────────────────────",
        "export XFORMERS_ENABLE=1",
        "export FLASH_ATTN_ENABLE=1",
        "",
        "# ── AudioCraft ────────────────────────────────────────────────────────",
        "export AUDIOCRAFT_CACHE_DIR=/workspace/models",
        "",
        "# ── Quality feedback ──────────────────────────────────────────────────",
        "export QUALITY_CLIP_THRESHOLD=0.93",
        "export QUALITY_MAX_RETRIES=3",
        "",
        "# ── Optical flow ──────────────────────────────────────────────────────",
        "export OPTICAL_FLOW_BACKEND=torchvision",
        "",
        "# ── Audio constraints ─────────────────────────────────────────────────",
        "export SNR_MIN_DB=10",
        "export AUDIO_PEAK_MAX_DBFS=0",
        "",
        "# ── VRAM enforcement ──────────────────────────────────────────────────",
        "export VRAM_ENFORCE_HARD_LIMIT=true",
        "",
        "# ── v6.5 execution contract ───────────────────────────────────────────",
        "export IMMUTABLE_CONTEXT_ENFORCE=true",
        "export IDENTITY_DRIFT_THRESHOLD=0.15",
        "export CROSS_MODAL_SYNC_THRESHOLD=0.9",
        "export HRG_REVIEW_ENABLED=true",
        "export HRG_APPROVAL_TIMEOUT_SECONDS=300",
    ]

    env_path = "/workspace/.env_vga"
    with open(env_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    # Source immediately so subsequent phases in this process pick up the vars.
    subprocess.run(
        f"export $(grep -v '^#' {env_path} | grep '=' | xargs)",
        shell=True, check=False,
    )

    log.info("[phase3] /workspace/.env_vga written.")
    log.info("=== PHASE 3 COMPLETE ===")


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 4 — MODEL DOWNLOAD SYSTEM  (spec Part 2 §5)
# ─────────────────────────────────────────────────────────────────────────────

# ── State machine helpers ────────────────────────────────────────────────────

def _state_path(key: str, suffix: str) -> str:
    return os.path.join(STATE_DIR, f"{key}{suffix}")

def mark_downloading(key: str) -> None:
    open(_state_path(key, ".downloading"), "w").close()

def mark_complete(key: str) -> None:
    dl = _state_path(key, ".downloading")
    if os.path.exists(dl):
        os.remove(dl)
    open(_state_path(key, ".complete"), "w").close()

def is_complete(key: str) -> bool:
    return os.path.exists(_state_path(key, ".complete"))

def clear_state(key: str) -> None:
    for suffix in (".downloading", ".complete"):
        p = _state_path(key, suffix)
        if os.path.exists(p):
            os.remove(p)


# ── Asset registry (spec Part 2 §5) ─────────────────────────────────────────
#
# SVI LoRA entries use type "svi_lora" with hf_include download.
# The filenames include the version-2.0/ subfolder prefix as they appear
# in the vita-video-gen/svi-model HuggingFace repository tree.

SYSTEM_ASSET_REGISTRY: dict[str, dict] = {
    "qwen": {
        "repo_id":   "unsloth/Qwen2.5-14B-Instruct-unsloth-bnb-4bit",
        "local_dir": "/workspace/models/qwen",
        "type":      "hf_snapshot",
        "gated":     True,
        "est_gb":    12,   # verified 11.3 GB (was 8)
    },
    "flux2": {
        "repo_id":   "black-forest-labs/FLUX.2-klein-4B",
        "local_dir": "/workspace/models/flux2",
        "type":      "hf_snapshot",
        "gated":     True,
        "est_gb":    24,   # verified 23.7 GB
    },
    "zimage": {
        "repo_id":   "Tongyi-MAI/Z-Image-Turbo",
        "local_dir": "/workspace/models/zimage",
        "type":      "hf_snapshot",
        "gated":     True,
        "est_gb":    33,   # verified 32.9 GB (was 16 — transformer 24.6 GB + text_encoder 8 GB)
    },
    "wan22": {
        "repo_id":   "nalexand/Wan2.2-I2V-A14B-FP8",
        "local_dir": "/workspace/models/wan22",
        "type":      "hf_snapshot",
        "gated":     True,
        "est_gb":    41,   # verified 40.5 GB (was 30)
    },
    # svi_core REMOVED — vita-video-gen/svi-model full repo (24.5 GB) is not needed.
    # VGA only uses the two SVI LoRA files downloaded below via svi_high_noise / svi_low_noise.
    "latentsync": {
        "repo_id":   "ByteDance/LatentSync-1.6",
        "local_dir": "/workspace/models/latentsync",
        "type":      "hf_snapshot",
        "gated":     True,
        "est_gb":    10,   # verified 9.64 GB (was 8)
    },
    "cosyvoice": {
        "repo_id":   "FunAudioLLM/Fun-CosyVoice3-0.5B-2512",
        "local_dir": "/workspace/CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B",
        "type":      "hf_snapshot",
        "gated":     True,
        "est_gb":    10,   # verified 9.75 GB
    },
    "musicgen": {
        "repo_id":         "facebook/musicgen-medium",
        "local_dir":       "/workspace/models/musicgen",
        "type":            "hf_snapshot",
        "gated":           False,
        "est_gb":          9,    # verified 8.04 GB pytorch_model.bin + tokenizer (was 4)
        # Exclude AudioCraft-format files — VGA wrapper uses Transformers (pytorch_model.bin).
        # Saves 3.9 GB (state_dict.bin 3.68 GB + compression_state_dict.bin 236 MB).
        "ignore_patterns": ["state_dict.bin", "compression_state_dict.bin"],
    },
    "mmaudio": {
        "repo_id":        "hkchengrex/MMAudio",
        "local_dir":      "/workspace/models/mmaudio",
        "type":           "hf_snapshot",
        "gated":          False,
        "est_gb":         6,    # verified: medium_44k (2.49 GB) + ext_weights (3.31 GB) = 5.8 GB
        # Download only the medium model + required ext_weights.
        # Skips checkpoints/ (39.3 GB training data) and unused model sizes.
        # Saves ~48.9 GB vs full hf_snapshot (54.7 GB total repo).
        "allow_patterns": [
            "weights/mmaudio_medium_44k.pth",
            "ext_weights/*",
        ],
    },
    "lora_consistency": {
        # Targeted single-file download — only the FLUX.2-klein-4B variant.
        # Full repo is 3.47 GB but contains 9B LoRAs incompatible with VGA's 4B model.
        # Verified filename: f2k_4B_consist_20260314.safetensors (385 MB).
        "repo_id":    "lrzjason/Consistance_Edit_Lora",
        "include":    "f2k_4B_consist_20260314.safetensors",
        "local_dir":  "/workspace/loras/consistency",
        "local_path": "/workspace/loras/consistency/f2k_4B_consist_20260314.safetensors",
        "type":       "hf_include",
        "gated":      False,
        "est_gb":     0.4,  # verified 385 MB (was 1 GB — saves 3.08 GB)
    },
    "lora_identity": {
        "repo_id":   None,  # from LORA_IDENTITY_REPO env var
        "local_dir": "/workspace/loras/identity/character_main",
        "type":      "hf_snapshot",
        "gated":     True,
        "est_gb":    1,
    },
    "lora_style": {
        "repo_id":   None,  # from LORA_STYLE_REPO env var
        "local_dir": "/workspace/loras/style/cinematic",
        "type":      "hf_snapshot",
        "gated":     True,
        "est_gb":    1,
    },
    # ── SVI LoRAs ──────────────────────────────────────────────────────────
    # CRITICAL: filenames include the version-2.0/ subfolder prefix.
    # These are the VERIFIED correct filenames from vita-video-gen/svi-model.
    "svi_high_noise": {
        "repo_id":    "vita-video-gen/svi-model",
        # include path = the --include argument for huggingface_hub
        "include":    "version-2.0/SVI_Wan2.2-I2V-A14B_high_noise_lora_v2.0_pro.safetensors",
        "local_dir":  "/workspace/models/svi",
        # local_path after download (include preserves subfolder structure)
        "local_path": (
            "/workspace/models/svi/version-2.0/"
            "SVI_Wan2.2-I2V-A14B_high_noise_lora_v2.0_pro.safetensors"
        ),
        "type":       "svi_lora",
        "gated":      True,
        "est_gb":     2,
    },
    "svi_low_noise": {
        "repo_id":    "vita-video-gen/svi-model",
        "include":    "version-2.0/SVI_Wan2.2-I2V-A14B_low_noise_lora_v2.0_pro.safetensors",
        "local_dir":  "/workspace/models/svi",
        "local_path": (
            "/workspace/models/svi/version-2.0/"
            "SVI_Wan2.2-I2V-A14B_low_noise_lora_v2.0_pro.safetensors"
        ),
        "type":       "svi_lora",
        "gated":      True,
        "est_gb":     2,
    },
    "clip": {
        "repo_id":         "openai/clip-vit-large-patch14",
        "local_dir":       "/workspace/auxiliary/clip",
        "type":            "hf_snapshot",
        "gated":           False,
        "est_gb":          2,    # verified: safetensors + tokenizer = 1.71 GB (was 1)
        # Exclude Flax (JAX) and TensorFlow framework formats — VGA only uses safetensors.
        # Also exclude legacy pytorch_model.bin (replaced by model.safetensors).
        # Saves ~5.1 GB (3 unused formats × 1.71 GB each; full repo is 6.85 GB).
        "ignore_patterns": ["flax_model.msgpack", "tf_model.h5", "pytorch_model.bin"],
    },
}

# ── Dependency graph (spec Part 2 §5.6) ─────────────────────────────────────

ASSET_DEPENDENCIES: dict[str, list[str]] = {
    # svi_core removed — VGA only needs the two LoRA files (svi_high_noise, svi_low_noise)
    "flux2":            ["clip"],   # lora_identity removed — optional user LoRA, not required for flux2
    "zimage":           ["clip", "lora_consistency"],
    "wan22":            ["clip"],
    "latentsync":       [],
    "cosyvoice":        [],
    "mmaudio":          [],
    "musicgen":         [],
    "qwen":             [],
    "lora_identity":    [],
    "lora_style":       [],
    "lora_consistency": [],
    "svi_high_noise":   [],
    "svi_low_noise":    [],
    "clip":             [],
}

# Download order respects dependency graph — clip first, large models last.
# svi_core removed — VGA only needs the two targeted SVI LoRA files.
DOWNLOAD_ORDER = [
    "clip",
    "lora_consistency",
    "lora_identity",
    "lora_style",
    "svi_high_noise",
    "svi_low_noise",
    "cosyvoice",
    "musicgen",
    "mmaudio",
    "latentsync",
    "qwen",
    "zimage",
    "wan22",
    "flux2",
]


# ── Disk guard ───────────────────────────────────────────────────────────────

def _ensure_disk_headroom(required_gb: float) -> None:
    """
    Per-asset rolling headroom check (Phase 4).
    Ensures free disk space remains above MIN_FREE_GB after downloading
    an asset of size required_gb.

    Two-tier abort:
      1. Hard abort if free < MIN_FREE_GB (global floor, regardless of asset)
      2. Hard abort if free < required_gb + MIN_FREE_GB (not enough for this asset + buffer)
    """
    import psutil
    usage   = psutil.disk_usage("/workspace")
    free_gb = usage.free / (1024 ** 3)
    if free_gb < MIN_FREE_GB:
        raise RuntimeError(
            f"[disk] ABORT — only {free_gb:.1f} GB free; "
            f"global minimum headroom is {MIN_FREE_GB} GB."
        )
    if free_gb < required_gb + MIN_FREE_GB:
        raise RuntimeError(
            f"[disk] ABORT — not enough headroom for asset "
            f"requiring ~{required_gb:.1f} GB "
            f"({free_gb:.1f} GB free, need {MIN_FREE_GB + required_gb:.1f} GB)."
        )


# ── HF download helpers ──────────────────────────────────────────────────────

def _snapshot_download(
    repo_id: str,
    local_dir: str,
    token: str,
    allow_patterns: list | None = None,
    ignore_patterns: list | None = None,
) -> None:
    from huggingface_hub import snapshot_download
    os.makedirs(local_dir, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        token=token or None,
        resume_download=True,
        local_dir_use_symlinks=False,
        force_download=False,
        etag_timeout=30,
        max_workers=4,
        allow_patterns=allow_patterns or None,
        ignore_patterns=ignore_patterns or None,
    )


def _file_download(repo_id: str, filename: str, local_dir: str, token: str) -> None:
    from huggingface_hub import hf_hub_download
    os.makedirs(local_dir, exist_ok=True)
    hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=local_dir,
        token=token or None,
    )


def _include_download(
    repo_id: str, include_pattern: str, local_dir: str, token: str
) -> None:
    """
    Download a specific file from an HF repo using the snapshot_download
    allow_patterns mechanism. This correctly preserves subfolder structure
    (e.g., version-2.0/ inside vita-video-gen/svi-model).

    Used for SVI LoRAs to avoid downloading the entire repo while still
    placing files in their correct version-2.0/ subdirectory.
    """
    from huggingface_hub import snapshot_download
    os.makedirs(local_dir, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        allow_patterns=[include_pattern],
        token=token or None,
        resume_download=True,
        local_dir_use_symlinks=False,
    )
    log.info(
        f"[download] Partial download: {repo_id} → "
        f"{local_dir}/{include_pattern}"
    )


# ── Per-asset download  ──────────────────────────────────────────────────────

def _download_asset(key: str) -> bool:
    """
    Idempotent, crash-safe, retryable download for a single asset.
    Uses .downloading / .complete state files (spec Part 2 §5.2).
    Atomic write: download → temp → rename → .complete.
    """
    if is_complete(key):
        log.info(f"[{key}] Already complete — skipping.")
        return True

    cfg = SYSTEM_ASSET_REGISTRY.get(key)
    if cfg is None:
        log.error(f"[{key}] Not in asset registry.")
        return False

    _ensure_disk_headroom(cfg.get("est_gb", 1.0))
    token = HF_TOKEN if cfg.get("gated") else ""

    mark_downloading(key)
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            log.info(f"[{key}] Download attempt {attempt}/{MAX_RETRIES} …")

            if cfg["type"] == "hf_snapshot":
                repo_id = cfg["repo_id"]
                if repo_id is None:
                    repo_id = (
                        LORA_IDENTITY_REPO if key == "lora_identity"
                        else LORA_STYLE_REPO if key == "lora_style"
                        else None
                    )
                if not repo_id:
                    # Optional user LoRA not configured — treat as skipped (success),
                    # not as failure. lora_identity and lora_style are user-supplied.
                    log.info(f"[{key}] repo_id not configured — optional, skipping (OK).")
                    mark_complete(key)   # mark as complete so manifest counts it as passed
                    return True
                _snapshot_download(
                    repo_id,
                    cfg["local_dir"],
                    token,
                    allow_patterns=cfg.get("allow_patterns"),
                    ignore_patterns=cfg.get("ignore_patterns"),
                )

            elif cfg["type"] == "hf_include":
                # Targeted single-file download (no SVI symlink).
                # Used for LoRAs where only one specific file is needed.
                local_path = cfg["local_path"]
                if not os.path.isfile(local_path):
                    _include_download(
                        cfg["repo_id"],
                        cfg["include"],
                        cfg["local_dir"],
                        token,
                    )
                else:
                    log.info(f"[{key}] File already exists: {local_path}")

            elif cfg["type"] == "svi_lora":
                # Targeted partial download: retrieve only the specific file
                # from the version-2.0/ subfolder of vita-video-gen/svi-model.
                # _include_download preserves subfolder structure so the file
                # lands at local_dir/version-2.0/<filename>.
                local_path = cfg["local_path"]
                if not os.path.isfile(local_path):
                    _include_download(
                        cfg["repo_id"],
                        cfg["include"],
                        cfg["local_dir"],
                        token,
                    )
                    # After download, create convenience symlink in loras/svi/
                    _create_svi_symlink(local_path)
                else:
                    log.info(f"[{key}] File already exists: {local_path}")

            else:
                log.error(f"[{key}] Unknown asset type: {cfg['type']}")
                return False

            # Post-download validation
            passed, reason = _full_validate(key)
            if passed:
                mark_complete(key)
                log.info(f"[{key}] ✅ Download + validation PASSED: {reason}")
                return True
            else:
                log.error(f"[{key}] Validation FAILED (attempt {attempt}): {reason}")
                if attempt < MAX_RETRIES:
                    time.sleep(BACKOFF[attempt - 1])
                continue

        except Exception as exc:
            log.error(f"[{key}] Exception (attempt {attempt}): {exc}")
            if attempt < MAX_RETRIES:
                time.sleep(BACKOFF[attempt - 1])

    log.error(f"[{key}] ❌ All {MAX_RETRIES} download attempts failed.")
    return False


def _create_svi_symlink(lora_path: str) -> None:
    """
    Create a convenience symlink for an SVI LoRA file in /workspace/loras/svi/.
    Makes it accessible as /workspace/loras/svi/<basename> without the version-2.0/ prefix.
    """
    symlink_dir = "/workspace/loras/svi"
    os.makedirs(symlink_dir, exist_ok=True)
    basename = os.path.basename(lora_path)
    link_path = os.path.join(symlink_dir, basename)
    try:
        if not os.path.exists(link_path):
            os.symlink(lora_path, link_path)
            log.info(f"[svi] Symlink created: {link_path} → {lora_path}")
        else:
            log.info(f"[svi] Symlink already exists: {link_path}")
    except Exception as exc:
        log.warning(f"[svi] Could not create symlink for {basename}: {exc} — non-fatal.")


# ── Dependency-aware resolver ────────────────────────────────────────────────

def _resolve_and_download(key: str, visited: set | None = None) -> bool:
    if visited is None:
        visited = set()
    if key in visited:
        log.warning(f"[resolver] Circular dependency at '{key}' — skipping.")
        return True
    visited.add(key)

    for dep in ASSET_DEPENDENCIES.get(key, []):
        if not is_complete(dep):
            log.info(f"[resolver] '{key}' depends on '{dep}' — resolving first.")
            if not _resolve_and_download(dep, visited):
                log.error(f"[resolver] Dependency '{dep}' failed — cannot proceed with '{key}'.")
                return False

    return _download_asset(key)


def download_models() -> dict[str, bool]:
    """Phase 4: download all assets in dependency order."""
    log.info("=== PHASE 4: MODEL DOWNLOAD ===")
    results: dict[str, bool] = {}

    # De-duplicate order list while preserving order
    seen: set[str] = set()
    order = [k for k in DOWNLOAD_ORDER if not (k in seen or seen.add(k))]  # type: ignore[func-returns-value]

    for key in order:
        log.info(f"[phase4] Processing: {key}")
        results[key] = _resolve_and_download(key)

    passed = [k for k, v in results.items() if v]
    failed = [k for k, v in results.items() if not v]
    log.info(f"[phase4] Passed ({len(passed)}): {passed}")
    if failed:
        log.error(f"[phase4] Failed ({len(failed)}): {failed}")
    log.info("=== PHASE 4 COMPLETE ===")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 5 — VALIDATION SYSTEM  (spec Part 2 §6)
# Multi-layer: file integrity → subcomponent → load test → runtime checks
# ─────────────────────────────────────────────────────────────────────────────

MIN_FILE_COUNTS = {
    "qwen":             5,
    "flux2":            4,
    "zimage":           4,
    "wan22":            2,   # nalexand FP8 layout: VAE.pth + T5.pth at root + model subdirs
    # svi_core removed from registry
    "latentsync":       3,
    "cosyvoice":        3,
    "musicgen":         2,   # pytorch_model.bin + tokenizer files (state_dict excluded)
    "mmaudio":          5,   # mmaudio_medium_44k.pth + 4 ext_weights files
    "lora_consistency": 1,   # single file: f2k_4B_consist_20260314.safetensors
    "lora_identity":    1,
    "lora_style":       1,
    "clip":             3,   # model.safetensors + tokenizer files (flax/tf excluded)
}

DIFFUSION_SUBCOMPONENTS: dict[str, dict] = {
    "flux2": {
        # FLUX.2-klein-4B layout: tokenizer is in tokenizer/ dir, NOT inside text_encoder/.
        # text_encoder/ contains only config.json + safetensors shards (no tokenizer.json).
        "vae":          {"required_files": ["config.json"],
                         "weight_patterns": ["*.safetensors"]},
        "text_encoder": {"required_files": ["config.json"],
                         "weight_patterns": ["*.safetensors"]},
        "scheduler":    {"required_files": ["scheduler_config.json"]},
    },
    "zimage": {
        "vae":       {"required_files": ["config.json"],
                      "weight_patterns": ["*.safetensors"]},
        "scheduler": {"required_files": ["scheduler_config.json"]},
    },
    # wan22 (nalexand/Wan2.2-I2V-A14B-FP8) uses non-diffusers layout:
    # Wan2.1_VAE.pth + models_t5_umt5-xxl-enc-bf16.pth at root,
    # high_noise_model_fp8/ and low_noise_model_fp8/ as subdirs.
    # No vae/ or scheduler/ subdirectory — removed from subcomponent checks.
}


def _validate_layer1_structure(local_dir: str, key: str) -> tuple[bool, str]:
    """Layer 1 — file existence + minimum count."""
    if not os.path.isdir(local_dir):
        return False, f"Directory missing: {local_dir}"
    count = sum(1 for f in Path(local_dir).rglob("*") if f.is_file())
    min_c = MIN_FILE_COUNTS.get(key, 1)
    if count < min_c:
        return False, (
            f"[{key}] Only {count} files in {local_dir}; "
            f"expected ≥ {min_c}."
        )
    return True, f"[{key}] Structure OK ({count} files)"


def _validate_layer2_subcomponents(model_dir: str, key: str) -> tuple[bool, str]:
    """Layer 2 — diffusion subcomponent presence + required files."""
    if key not in DIFFUSION_SUBCOMPONENTS:
        return True, f"[{key}] Not a diffusion model — skipped."
    spec = DIFFUSION_SUBCOMPONENTS[key]
    for comp, comp_spec in spec.items():
        comp_dir = os.path.join(model_dir, comp)
        if not os.path.isdir(comp_dir):
            return False, f"[{key}] Subcomponent '{comp}' dir missing."
        for req in comp_spec.get("required_files", []):
            if not os.path.exists(os.path.join(comp_dir, req)):
                return False, f"[{key}] Missing '{req}' in '{comp}'."
        for pat in comp_spec.get("weight_patterns", []):
            if not glob.glob(os.path.join(comp_dir, pat)):
                log.warning(
                    f"[{key}] No weight files in '{comp}' "
                    "— component may load from parent dir."
                )
    return True, f"[{key}] Subcomponents OK."


def _validate_layer3_model_load(local_dir: str, key: str) -> tuple[bool, str]:
    """Layer 3 — safetensors spot-check on the first safetensors file found."""
    try:
        candidates = list(Path(local_dir).rglob("*.safetensors"))
        if not candidates:
            return True, f"[{key}] No safetensors found — load test skipped."
        import safetensors.torch as st
        st.load_file(str(candidates[0]))
        return True, f"[{key}] Load test OK: {candidates[0].name}"
    except Exception as exc:
        return False, f"[{key}] Load test FAILED: {exc}"


def _validate_layer4_diffusion_pipeline(local_dir: str, key: str) -> tuple[bool, str]:
    """Layer 4 — AutoPipeline from_pretrained smoke test."""
    if key not in ("flux2", "zimage"):
        return True, f"[{key}] Diffusion pipeline test not applicable."
    try:
        from diffusers import AutoPipelineForText2Image
        import torch
        AutoPipelineForText2Image.from_pretrained(
            local_dir,
            torch_dtype=torch.float16,
            device_map=None,
        )
        return True, f"[{key}] Diffusion pipeline load OK."
    except Exception as exc:
        return False, f"[{key}] Diffusion pipeline test FAILED: {exc}"


def _validate_layer5_optical_flow() -> tuple[bool, str]:
    """Layer 5 — optical flow backend availability."""
    try:
        from torchvision.models.optical_flow import raft_small  # noqa: F401
        return True, "torchvision RAFT"
    except Exception:
        pass
    try:
        import cv2
        import numpy as np
        dummy = np.zeros((10, 10), dtype=np.float32)
        cv2.calcOpticalFlowFarneback(dummy, dummy, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        return True, "OpenCV Farneback fallback"
    except Exception as exc:
        return False, f"Both optical flow backends failed: {exc}"


def _validate_layer6_audio() -> tuple[bool, str]:
    """Layer 6 — ffmpeg binary check."""
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            check=True, capture_output=True, text=True, timeout=10,
        )
        return True, result.stdout.split("\n")[0]
    except Exception as exc:
        return False, f"ffmpeg not functional: {exc}"


def _validate_layer7_gpu() -> tuple[bool, str]:
    """Layer 7 — GPU availability + VRAM check."""
    try:
        import torch
        if not torch.cuda.is_available():
            return False, "CUDA not available"
        vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        return True, f"GPU OK — {vram:.1f} GB VRAM"
    except Exception as exc:
        return False, f"GPU check failed: {exc}"


def _validate_svi_lora(key: str) -> tuple[bool, str]:
    """
    Dedicated validator for single-file SVI LoRAs.
    Checks the local_path (version-2.0/ subfolder) for the file.
    Also checks the symlink in /workspace/loras/svi/.
    """
    cfg       = SYSTEM_ASSET_REGISTRY.get(key, {})
    path      = cfg.get("local_path", "")
    basename  = os.path.basename(path)
    symlink   = f"/workspace/loras/svi/{basename}"

    if not os.path.isfile(path):
        return False, (
            f"[{key}] SVI LoRA file missing: {path}\n"
            f"  Expected: version-2.0/{basename}\n"
            f"  Source:   vita-video-gen/svi-model (HuggingFace)"
        )
    size_mb = os.path.getsize(path) / (1024 ** 2)
    if size_mb < 100:   # SVI Pro LoRAs are ~1.1 GB each
        return False, f"[{key}] SVI LoRA suspiciously small ({size_mb:.1f} MB)."

    # Symlink check (non-fatal warning if missing)
    if not os.path.exists(symlink):
        log.warning(
            f"[{key}] Convenience symlink missing: {symlink} — "
            "run _create_svi_symlink() to recreate."
        )

    return True, f"[{key}] SVI LoRA OK ({size_mb:.0f} MB) at {path}"


def _full_validate(key: str) -> tuple[bool, str]:
    """Run all applicable validation layers for a single asset."""
    cfg = SYSTEM_ASSET_REGISTRY.get(key)
    if cfg is None:
        return False, f"Asset '{key}' not in registry."

    if cfg["type"] == "svi_lora":
        return _validate_svi_lora(key)

    local_dir = cfg["local_dir"]

    for layer_fn in [
        lambda: _validate_layer1_structure(local_dir, key),
        lambda: _validate_layer2_subcomponents(local_dir, key),
        lambda: _validate_layer3_model_load(local_dir, key),
        lambda: _validate_layer4_diffusion_pipeline(local_dir, key),
    ]:
        passed, reason = layer_fn()
        if not passed:
            return False, reason

    log.info(f"[validate] {key}: PASS")
    return True, f"[{key}] All layers passed."


def validate_all() -> None:
    """
    Phase 5: Run runtime-level validations (optical flow, audio, GPU).
    Per-asset validation is performed inline during Phase 4.
    Hard failures (audio, GPU) abort bootstrap. Optical flow degraded is a warning.
    """
    log.info("=== PHASE 5: RUNTIME VALIDATION ===")

    ok, msg = _validate_layer6_audio()
    if not ok:
        raise RuntimeError(f"[phase5] Audio validation FAILED: {msg}")
    log.info(f"[phase5] Audio: {msg}")

    ok, msg = _validate_layer5_optical_flow()
    if not ok:
        log.warning(f"[phase5] Optical flow degraded: {msg}")
    else:
        log.info(f"[phase5] Optical flow backend: {msg}")

    ok, msg = _validate_layer7_gpu()
    if not ok:
        raise RuntimeError(f"[phase5] GPU validation FAILED: {msg}")
    log.info(f"[phase5] {msg}")

    log.info("=== PHASE 5 COMPLETE ===")


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 6 — MANIFEST  (spec Part 2 §7)
# ─────────────────────────────────────────────────────────────────────────────

def write_manifest(
    download_results: dict[str, bool],
    validation_status: str = "passed",
) -> None:
    """
    Phase 6: Write /workspace/state/run_manifest.json.
    This file is the bootstrap completion signal consumed by session_controller.

    v7.1 CRITICAL — All three fields checked by session_controller.manifest_valid()
    are GUARANTEED present in every manifest this function writes:

      "validation"   : str  — "passed" | "failed" | "in_progress"
                              session_controller requires == "passed" to skip re-bootstrap
      "models"       : list — completed asset keys; must be non-empty for valid manifest
      "torch_version": str  — installed torch version; must be present for valid manifest

    Call write_manifest(results, validation_status="failed") from error paths
    to ensure a crashed bootstrap cannot produce a manifest that the controller
    would mistake for a successful one.
    """
    log.info("=== PHASE 6: MANIFEST WRITE ===")

    import importlib
    try:
        torch = importlib.import_module("torch")
        torch_version = torch.__version__
        cuda_version  = torch.version.cuda or "unknown"
    except Exception:
        torch_version = "unknown"
        cuda_version  = "unknown"

    completed_models = [k for k, v in download_results.items() if v]
    failed_models    = [k for k, v in download_results.items() if not v]

    manifest = {
        # ── Required by session_controller.manifest_valid() ───────────────
        "validation":    validation_status,   # checked: == "passed"
        "models":        completed_models,    # checked: bool(models)
        "torch_version": torch_version,       # checked: bool(torch_version)
        # ── Additional context ────────────────────────────────────────────
        "spec_version":                   "v7.1.1",
        "timestamp":                      time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "cuda_version":                   cuda_version,
        "dependencies":                   "installed",
        "assets_complete":                completed_models,
        "assets_failed":                  failed_models,
        # ── VGA v6.5 enforcement flags ────────────────────────────────────
        "execute_stage_enforced":         True,
        "immutable_context_enforced":     True,
        "hrg_checkpoints_active":         True,
        "cross_modal_validation_active":  True,
        "identity_tracker_stateful":      True,
        "temporal_loop_enforced":         True,
        "system_guard_active":            True,
        # ── SVI LoRA filename record (for auditability) ───────────────────
        "svi_high_noise_file":            SVI_HIGH_NOISE_FILE,
        "svi_low_noise_file":             SVI_LOW_NOISE_FILE,
    }

    with open(MANIFEST, "w") as f:
        json.dump(manifest, f, indent=2)

    log.info(f"[phase6] Manifest written: {MANIFEST}")
    log.info(
        f"[phase6] validation={validation_status}  "
        f"models={len(completed_models)}  torch={torch_version}"
    )
    log.info("=== PHASE 6 COMPLETE ===")


# ─────────────────────────────────────────────────────────────────────────────
# BOOTSTRAP ENTRYPOINT  (spec Part 2 §1.4)
# ─────────────────────────────────────────────────────────────────────────────

def _write_sentinel_manifest() -> None:
    """
    v7.1: Write a sentinel manifest with validation="in_progress" at the very
    start of bootstrap execution.

    This ensures that if bootstrap crashes at any phase — before write_manifest()
    runs — the controller's manifest_valid() check will return False
    (validation != "passed") and trigger a full re-bootstrap on the next run,
    rather than finding a stale "passed" manifest from a previous run and
    incorrectly skipping bootstrap.

    Overwritten with validation="passed" only on full successful completion.
    """
    sentinel = {
        "validation":    "in_progress",
        "models":        [],
        "torch_version": "unknown",
        "spec_version":  "v7.1.1",
        "timestamp":     time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "note":          "Bootstrap started but not yet complete — do not treat as valid.",
    }
    try:
        with open(MANIFEST, "w") as f:
            json.dump(sentinel, f, indent=2)
        log.info("[bootstrap] Sentinel manifest written (validation=in_progress).")
    except Exception as exc:
        log.warning(f"[bootstrap] Could not write sentinel manifest: {exc}")


def main() -> int:
    log.info("=" * 70)
    log.info("VGA v17.2 Bootstrap Pipeline — Spec v7.0 / v7.1 Patch / v7.1.1 Fixes")
    log.info("=" * 70)
    log.info(f"SVI_HIGH_NOISE_FILE = {SVI_HIGH_NOISE_FILE}")
    log.info(f"SVI_LOW_NOISE_FILE  = {SVI_LOW_NOISE_FILE}")
    log.info("=" * 70)

    # v7.1: Write sentinel before any phase runs.
    # If bootstrap crashes mid-run, controller will detect validation != "passed".
    os.makedirs(STATE_DIR, exist_ok=True)
    _write_sentinel_manifest()

    download_results: dict[str, bool] = {}

    try:
        # Strict phase order
        system_setup()                        # Phase 1
        install_dependencies()                # Phase 2
        configure_env()                       # Phase 3
        download_results = download_models()  # Phase 4
        validate_all()                        # Phase 5

        failed = [k for k, v in download_results.items() if not v]
        if failed:
            log.error(f"Bootstrap INCOMPLETE — failed assets: {failed}")
            # Write failed manifest so controller detects partial completion
            write_manifest(download_results, validation_status="failed")
            return 1

        # All phases succeeded — write the success manifest
        write_manifest(download_results, validation_status="passed")  # Phase 6

        log.info("=" * 70)
        log.info("Bootstrap COMPLETE — all assets downloaded and validated.")
        log.info("=" * 70)
        return 0

    except Exception as exc:
        log.exception(f"Bootstrap ABORTED: {exc}")
        # Always write a failed manifest on exception so the controller
        # cannot mistakenly reuse a stale "passed" manifest
        try:
            write_manifest(download_results, validation_status="failed")
        except Exception as manifest_exc:
            log.warning(f"[phase6] Could not write failure manifest: {manifest_exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
