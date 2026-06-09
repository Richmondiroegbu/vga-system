#!/usr/bin/env python3
"""
test_minimal_bootstrap.py — Minimal model downloader for image/video quality testing.

Downloads ONLY the three models needed for the image+video diagnostic test:
  1. FLUX.2-klein-4B        (image generation)
  2. Z-Image-Turbo          (image refinement)
  3. WAN2.2-I2V-A14B        (video generation — BF16 or FP8 based on your GPU)
  4. SVI LoRAs              (high + low noise, required by WAN2.2 inference)

Skips everything else: Qwen, LatentSync, CosyVoice, MusicGen, MMAudio, CLIP.

Usage:
    # A6000 48GB or any 48GB+ GPU — use BF16 (production quality, matches real pipeline)
    python3 scripts/test_minimal_bootstrap.py --precision bf16 --hf-token YOUR_TOKEN

    # RTX 5090 32GB — use FP8 (fits in 32GB VRAM)
    python3 scripts/test_minimal_bootstrap.py --precision fp8 --hf-token YOUR_TOKEN

After this completes, run:
    python3 scripts/test_image_video.py --precision bf16   (or fp8)
"""
from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("/workspace/logs/test_bootstrap.log", mode="a"),
    ],
)
log = logging.getLogger("test_bootstrap")

os.makedirs("/workspace/logs", exist_ok=True)
os.environ.setdefault("PIP_BREAK_SYSTEM_PACKAGES", "1")

# ── Directory layout ─────────────────────────────────────────────────────────
DIRS = [
    "/workspace/models/flux2",
    "/workspace/models/zimage",
    "/workspace/models/wan22",
    "/workspace/models/wan22_bf16",
    "/workspace/models/svi/version-2.0",
    "/workspace/loras/svi",
    "/workspace/loras/consistency",
    "/workspace/output/test",
    "/workspace/logs",
    "/workspace/state",
    "/workspace/scripts",
]

# ── Model registry ───────────────────────────────────────────────────────────
MODELS = {
    "flux2": {
        "repo_id":   "black-forest-labs/FLUX.2-klein-4B",
        "local_dir": "/workspace/models/flux2",
        "est_gb":    24,
        "gated":     True,
    },
    "zimage": {
        "repo_id":   "Tongyi-MAI/Z-Image-Turbo",
        "local_dir": "/workspace/models/zimage",
        "est_gb":    33,
        "gated":     True,
    },
    # BF16 — for 48GB+ GPUs (A6000, RTX 6000 Ada, A100, etc.)
    "wan22_bf16": {
        "repo_id":   "Wan-AI/Wan2.2-I2V-A14B",
        "local_dir": "/workspace/models/wan22_bf16",
        "est_gb":    30,
        "gated":     True,
        "precision": "bf16",
    },
    # FP8 — for 32GB GPUs (RTX 5090, RTX 3090, etc.)
    "wan22_fp8": {
        "repo_id":   "nalexand/Wan2.2-I2V-A14B-FP8",
        "local_dir": "/workspace/models/wan22",
        "est_gb":    41,
        "gated":     True,
        "precision": "fp8",
    },
}

SVI_LORAS = [
    {
        "repo_id":    "vita-video-gen/svi-model",
        "include":    "version-2.0/SVI_Wan2.2-I2V-A14B_high_noise_lora_v2.0_pro.safetensors",
        "local_dir":  "/workspace/models/svi",
        "local_path": "/workspace/models/svi/version-2.0/SVI_Wan2.2-I2V-A14B_high_noise_lora_v2.0_pro.safetensors",
        "symlink":    "/workspace/loras/svi/SVI_Wan2.2-I2V-A14B_high_noise_lora_v2.0_pro.safetensors",
        "label":      "SVI high-noise LoRA",
    },
    {
        "repo_id":    "vita-video-gen/svi-model",
        "include":    "version-2.0/SVI_Wan2.2-I2V-A14B_low_noise_lora_v2.0_pro.safetensors",
        "local_dir":  "/workspace/models/svi",
        "local_path": "/workspace/models/svi/version-2.0/SVI_Wan2.2-I2V-A14B_low_noise_lora_v2.0_pro.safetensors",
        "symlink":    "/workspace/loras/svi/SVI_Wan2.2-I2V-A14B_low_noise_lora_v2.0_pro.safetensors",
        "label":      "SVI low-noise LoRA",
    },
]

# Consistance_Edit_LoRA — single file download (385 MB).
# Used by FLUX at S-06 for identity reinforcement.
# Only the 4B variant (f2k) is needed — repo also has a 9B file, skip it.
CONSISTENCY_LORA = {
    "repo_id":    "lrzjason/Consistance_Edit_Lora",
    "include":    "f2k_4B_consist_20260314.safetensors",
    "local_dir":  "/workspace/loras/consistency",
    "local_path": "/workspace/loras/consistency/f2k_4B_consist_20260314.safetensors",
    "label":      "Consistance_Edit_LoRA (FLUX identity)",
}

# ── Pip dependencies (minimal — only what's needed for FLUX + Z-Image + SVI) ─
MAIN_DEPS = [
    "torch>=2.7.0 torchvision --index-url https://download.pytorch.org/whl/cu128",
    "diffusers>=0.30.0",
    "transformers>=4.45.0",
    "accelerate>=0.34.0",
    "peft>=0.17.0",
    "safetensors>=0.4.0",
    "Pillow>=10.0.0",
    "numpy>=1.24.0",
    "opencv-python>=4.9.0",
    "huggingface-hub>=0.21.0",
    "hf-transfer>=0.1.6",
    "psutil",
]

SVI_ENV_DEPS = [
    # DiffSynth-Studio — the SVI pipeline lives here
    "git+https://github.com/modelscope/DiffSynth-Studio.git@main",
    # safetensors needed inside SVI env too
    "safetensors>=0.4.0",
    "opencv-python>=4.9.0",
    "Pillow>=10.0.0",
    "numpy>=1.24.0",
]


def pip_install(packages: str, python: str = sys.executable) -> bool:
    """Install pip packages. Returns True on success."""
    cmd = [python, "-m", "pip", "install", "--quiet"] + packages.split()
    log.info("pip install %s ...", packages.split()[0])
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        log.error("pip install failed: %s", result.stderr.decode()[:300])
        return False
    return True


def hf_snapshot(repo_id: str, local_dir: str, token: str,
                 allow_patterns: list | None = None) -> bool:
    """Download a full HF repo snapshot."""
    try:
        from huggingface_hub import snapshot_download
        os.makedirs(local_dir, exist_ok=True)
        snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            token=token or None,
            resume_download=True,
            local_dir_use_symlinks=False,
            allow_patterns=allow_patterns or None,
        )
        return True
    except Exception as exc:
        log.error("Download failed %s: %s", repo_id, exc)
        return False


def hf_include(repo_id: str, include: str, local_dir: str, token: str) -> bool:
    """Download a single file from an HF repo (preserves subfolder structure)."""
    try:
        from huggingface_hub import snapshot_download
        os.makedirs(local_dir, exist_ok=True)
        snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            allow_patterns=[include],
            token=token or None,
            resume_download=True,
            local_dir_use_symlinks=False,
        )
        return True
    except Exception as exc:
        log.error("Download failed %s / %s: %s", repo_id, include, exc)
        return False


def make_symlink(src: str, dst: str) -> None:
    """Create convenience symlink dst → src (non-fatal if it fails)."""
    try:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if not os.path.exists(dst):
            os.symlink(src, dst)
            log.info("  Symlink: %s", dst)
    except Exception as exc:
        log.warning("  Symlink skipped (%s): %s", dst, exc)


def check_disk(required_gb: float) -> None:
    import psutil
    free = psutil.disk_usage("/workspace").free / 1e9
    if free < required_gb + 10:
        raise RuntimeError(
            f"Only {free:.0f} GB free — need ~{required_gb + 10:.0f} GB "
            f"(model {required_gb:.0f} GB + 10 GB buffer). Free up space first."
        )
    log.info("Disk: %.0f GB free (need ~%.0f GB)", free, required_gb)


def setup_svi_env(svi_python: str) -> bool:
    """Install DiffSynth + dependencies into the svi_wan22 env."""
    if not Path(svi_python).exists():
        log.error(
            "SVI Python not found: %s\n"
            "  Create the env first:\n"
            "    conda create -n svi_wan22 python=3.10 -y\n"
            "    conda activate svi_wan22\n"
            "    pip install torch==2.7.1 torchvision "
            "--index-url https://download.pytorch.org/whl/cu128",
            svi_python,
        )
        return False

    log.info("Setting up SVI env at %s ...", svi_python)
    for pkg in SVI_ENV_DEPS:
        if not pip_install(pkg, python=svi_python):
            log.error("SVI env setup failed at: %s", pkg)
            return False

    # Clone Stable-Video-Infinity (svi_wan22 branch) if not already present
    svi_repo = "/workspace/Stable-Video-Infinity"
    if not Path(svi_repo).exists():
        log.info("Cloning Stable-Video-Infinity (svi_wan22 branch)...")
        result = subprocess.run([
            "git", "clone",
            "--branch", "svi_wan22",
            "--depth", "1",
            "https://github.com/vita-epfl/Stable-Video-Infinity.git",
            svi_repo,
        ], capture_output=True)
        if result.returncode != 0:
            log.error("Git clone failed: %s", result.stderr.decode()[:300])
            return False
        log.info("SVI repo cloned → %s", svi_repo)
    else:
        log.info("SVI repo already exists at %s", svi_repo)

    log.info("SVI env ready ✓")
    return True


def write_env(precision: str) -> None:
    """Write a minimal /workspace/.env_vga for the test run."""
    lines = [
        "# Minimal .env_vga for image/video quality test",
        f"export WAN22_PRECISION={precision}",
        "export WAN22_BF16_DIR=/workspace/models/wan22_bf16",
        "export SVI_REPO_BRANCH=svi_wan22",
        "export SVI_WAN22_PYTHON=/opt/conda/envs/svi_wan22/bin/python3",
        "export SVI_GPU_RESIDENT=1",
        "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
        "export HF_HOME=/workspace/cache/huggingface",
        "export HF_HUB_ENABLE_HF_TRANSFER=1",
    ]
    Path("/workspace/.env_vga").write_text("\n".join(lines) + "\n")
    log.info(".env_vga written (precision=%s)", precision)


def mark_done(key: str) -> None:
    Path(f"/workspace/state/{key}.done").touch()

def is_done(key: str) -> bool:
    return Path(f"/workspace/state/{key}.done").exists()


def main() -> int:
    parser = argparse.ArgumentParser(description="Minimal VGA test bootstrap")
    parser.add_argument(
        "--precision", choices=["bf16", "fp8"], default="bf16",
        help="bf16 = 48GB+ GPU (A6000/A100).  fp8 = 32GB GPU (RTX 5090).",
    )
    parser.add_argument("--hf-token", default=os.environ.get("HUGGING_FACE_HUB_TOKEN", ""),
                        help="HuggingFace token for gated models")
    parser.add_argument(
        "--svi-python",
        default="/opt/conda/envs/svi_wan22/bin/python3",
        help="Path to the svi_wan22 conda env Python interpreter",
    )
    parser.add_argument("--skip-svi-env", action="store_true",
                        help="Skip svi_wan22 env setup (use if already configured)")
    args = parser.parse_args()

    token = args.hf_token
    if not token:
        log.warning("No HF token provided — gated model downloads will fail.")
        log.warning("Pass it with: --hf-token hf_xxxx")

    os.environ["HUGGING_FACE_HUB_TOKEN"] = token
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

    log.info("=" * 60)
    log.info("VGA Minimal Test Bootstrap  precision=%s", args.precision)
    log.info("=" * 60)

    # Phase 1: directories
    log.info("--- Phase 1: Workspace directories ---")
    Path("/workspace/state").mkdir(parents=True, exist_ok=True)
    for d in DIRS:
        Path(d).mkdir(parents=True, exist_ok=True)
    log.info("Directories OK")

    # Phase 2: pip dependencies (main env)
    log.info("--- Phase 2: pip dependencies (main env) ---")
    if not is_done("main_deps"):
        for pkg in MAIN_DEPS:
            if not pip_install(pkg):
                log.error("Dependency install failed — aborting.")
                return 1
        mark_done("main_deps")
        log.info("Main deps installed ✓")
    else:
        log.info("Main deps already installed (skip)")

    # Phase 3: download models
    log.info("--- Phase 3: Model downloads ---")

    # FLUX.2-klein-4B
    if not is_done("flux2"):
        log.info("[flux2] Downloading FLUX.2-klein-4B (~23.7 GB) ...")
        check_disk(24)
        if hf_snapshot(MODELS["flux2"]["repo_id"], MODELS["flux2"]["local_dir"], token):
            mark_done("flux2")
            log.info("[flux2] ✓")
        else:
            log.error("[flux2] FAILED"); return 1
    else:
        log.info("[flux2] Already downloaded (skip)")

    # Z-Image-Turbo
    if not is_done("zimage"):
        log.info("[zimage] Downloading Z-Image-Turbo (~32.9 GB) ...")
        check_disk(33)
        if hf_snapshot(MODELS["zimage"]["repo_id"], MODELS["zimage"]["local_dir"], token):
            mark_done("zimage")
            log.info("[zimage] ✓")
        else:
            log.error("[zimage] FAILED"); return 1
    else:
        log.info("[zimage] Already downloaded (skip)")

    # WAN2.2 — route by precision
    wan_key = "wan22_bf16" if args.precision == "bf16" else "wan22_fp8"
    wan_cfg = MODELS[wan_key]
    done_key = f"wan22_{args.precision}"

    if not is_done(done_key):
        log.info("[%s] Downloading WAN2.2 %s (~%d GB) ...",
                 done_key, args.precision.upper(), wan_cfg["est_gb"])
        check_disk(wan_cfg["est_gb"])
        if hf_snapshot(wan_cfg["repo_id"], wan_cfg["local_dir"], token):
            mark_done(done_key)
            log.info("[%s] ✓", done_key)
        else:
            log.error("[%s] FAILED", done_key); return 1
    else:
        log.info("[%s] Already downloaded (skip)", done_key)

    # Consistance_Edit_LoRA (FLUX identity reinforcement)
    lora_key = "consistency_lora"
    if not is_done(lora_key):
        if Path(CONSISTENCY_LORA["local_path"]).exists():
            log.info("[consistency_lora] Already exists (skip)")
            mark_done(lora_key)
        else:
            log.info("[consistency_lora] Downloading %s (~385 MB) ...", CONSISTENCY_LORA["label"])
            if hf_include(CONSISTENCY_LORA["repo_id"], CONSISTENCY_LORA["include"],
                          CONSISTENCY_LORA["local_dir"], token):
                mark_done(lora_key)
                log.info("[consistency_lora] ✓")
            else:
                log.error("[consistency_lora] FAILED"); return 1
    else:
        log.info("[consistency_lora] Already downloaded (skip)")

    # SVI LoRAs (high-noise + low-noise)
    for lora in SVI_LORAS:
        lora_key = Path(lora["local_path"]).stem
        if not is_done(lora_key):
            if Path(lora["local_path"]).exists():
                log.info("[%s] Already exists (skip)", lora["label"])
                make_symlink(lora["local_path"], lora["symlink"])
                mark_done(lora_key)
                continue
            log.info("[%s] Downloading ...", lora["label"])
            if hf_include(lora["repo_id"], lora["include"], lora["local_dir"], token):
                make_symlink(lora["local_path"], lora["symlink"])
                mark_done(lora_key)
                log.info("[%s] ✓", lora["label"])
            else:
                log.error("[%s] FAILED", lora["label"]); return 1
        else:
            log.info("[%s] Already downloaded (skip)", lora["label"])
            make_symlink(lora["local_path"], lora["symlink"])

    # Phase 4: svi_wan22 env setup
    log.info("--- Phase 4: SVI env setup ---")
    if args.skip_svi_env:
        log.info("Skipping SVI env setup (--skip-svi-env)")
    elif is_done("svi_env"):
        log.info("SVI env already configured (skip)")
    else:
        if setup_svi_env(args.svi_python):
            mark_done("svi_env")
        else:
            log.warning(
                "SVI env setup failed or interpreter not found.\n"
                "  You can still run the IMAGE-ONLY test:\n"
                "    python3 scripts/test_image_video.py --image-only --precision %s",
                args.precision,
            )

    # Phase 5: write .env_vga
    log.info("--- Phase 5: Write .env_vga ---")
    write_env(args.precision)

    log.info("=" * 60)
    log.info("Bootstrap COMPLETE — ready to test.")
    log.info("")
    log.info("Run the test:")
    log.info("  python3 /workspace/scripts/test_image_video.py --precision %s", args.precision)
    log.info("  python3 /workspace/scripts/test_image_video.py --precision %s --image-only", args.precision)
    log.info("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
