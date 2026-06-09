#!/usr/bin/env python3
"""
test_image_video.py — Direct image + video quality test. No pipeline stages.

Skips ALL VGA pipeline orchestration (no Qwen, no HRG, no agents, no context).
Goes directly: FLUX → Z-Image → WAN2.2 (SVI).

This is specifically for diagnosing the "over fried" (overexposed/oversaturated)
output issue after the two confirmed fixes:
  1. Z-Image guidance_scale: 5.0 → 0.0  (distilled model, CFG must be 0)
  2. sigma_shift: 7.0 → 5.0             (SVI Python API default, 7.0 had no source)

Usage:
    # Full test: images + video
    python3 scripts/test_image_video.py --precision bf16

    # Images only (skip WAN2.2 — useful to isolate FLUX/Z-Image first)
    python3 scripts/test_image_video.py --precision bf16 --image-only

    # RTX 5090 32GB
    python3 scripts/test_image_video.py --precision fp8

    # Custom options
    python3 scripts/test_image_video.py --precision bf16 --steps 20 --cfg 7.0 \\
        --prompt "cinematic portrait of a resilient young woman, warm light"

Outputs saved to /workspace/output/test/:
    01_flux.png             — raw FLUX output
    02_zimage_refined.png   — Z-Image refined (guidance=0.0, strength=0.05)
    03_segment.mp4          — WAN2.2 video from the refined image
    test_report.txt         — brightness/saturation stats + parameters used
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("/workspace/logs/test_image_video.log", mode="a"),
    ],
)
log = logging.getLogger("test_image_video")

OUT_DIR = Path("/workspace/output/test")
OUT_DIR.mkdir(parents=True, exist_ok=True)
Path("/workspace/logs").mkdir(parents=True, exist_ok=True)

# ── Model paths ───────────────────────────────────────────────────────────────
FLUX_PATH         = "/workspace/models/flux2"
ZIMAGE_PATH       = "/workspace/models/zimage"
WAN22_BF16_PATH   = "/workspace/models/wan22_bf16"
WAN22_FP8_PATH    = "/workspace/models/wan22"
CONSISTENCY_LORA  = "/workspace/loras/consistency/f2k_4B_consist_20260314.safetensors"
SVI_HIGH_LORA     = "/workspace/loras/svi/SVI_Wan2.2-I2V-A14B_high_noise_lora_v2.0_pro.safetensors"
SVI_LOW_LORA      = "/workspace/loras/svi/SVI_Wan2.2-I2V-A14B_low_noise_lora_v2.0_pro.safetensors"
SVI_INFERENCE     = "/workspace/scripts/vga_svi_inference.py"
SVI_PYTHON        = "/opt/conda/envs/svi_wan22/bin/python3"


# ── Stats helper ─────────────────────────────────────────────────────────────

def image_stats(img) -> dict:
    """Return mean brightness and mean HSV saturation for a PIL image."""
    import numpy as np
    import cv2
    arr = np.array(img.convert("RGB"))
    brightness = float(arr.mean())
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV).astype(np.float32)
    saturation = float(hsv[:, :, 1].mean())
    return {
        "brightness": round(brightness, 1),
        "saturation": round(saturation, 1),
        "status": _exposure_label(brightness, saturation),
    }


def _exposure_label(brightness: float, saturation: float) -> str:
    if brightness > 180:
        return "OVEREXPOSED"
    if brightness > 155:
        return "BRIGHT (borderline)"
    if brightness < 60:
        return "UNDEREXPOSED"
    if saturation > 140:
        return "OVERSATURATED"
    if saturation > 120:
        return "SATURATED (borderline)"
    return "OK"


# ── Pre-flight checks ─────────────────────────────────────────────────────────

def preflight(precision: str, image_only: bool, svi_python: str) -> None:
    """Verify required models exist before starting. Fail fast with clear messages."""
    errors = []

    if not Path(FLUX_PATH).exists():
        errors.append(f"FLUX model missing: {FLUX_PATH}")
    if not Path(ZIMAGE_PATH).exists():
        errors.append(f"Z-Image model missing: {ZIMAGE_PATH}")

    if not image_only:
        wan_path = WAN22_BF16_PATH if precision == "bf16" else WAN22_FP8_PATH
        if not Path(wan_path).exists():
            errors.append(f"WAN2.2 {precision.upper()} model missing: {wan_path}")
        if not Path(SVI_HIGH_LORA).exists():
            errors.append(f"SVI high-noise LoRA missing: {SVI_HIGH_LORA}")
        if not Path(SVI_LOW_LORA).exists():
            errors.append(f"SVI low-noise LoRA missing: {SVI_LOW_LORA}")
        if not Path(SVI_INFERENCE).exists():
            errors.append(f"SVI inference script missing: {SVI_INFERENCE}")
        if not Path(svi_python).exists():
            errors.append(
                f"SVI Python not found: {svi_python}\n"
                f"  Run bootstrap first or pass --svi-python /path/to/python"
            )

    if errors:
        log.error("Pre-flight FAILED — missing required files:")
        for e in errors:
            log.error("  ✗ %s", e)
        log.error("\nRun first: python3 scripts/test_minimal_bootstrap.py --precision %s", precision)
        sys.exit(1)

    log.info("Pre-flight OK — all required models found.")


# ── Step 1: FLUX image generation ────────────────────────────────────────────

def run_flux(prompt: str, seed: int = 42) -> "PIL.Image.Image":
    """Generate a base image with FLUX.2-klein-4B.

    Settings used (verified correct for distilled FLUX):
      guidance_scale = 1.0   (BFL spec for distilled model)
      num_inference_steps = 4
    """
    import torch
    from diffusers import DiffusionPipeline

    log.info("--- Step 1: FLUX image generation ---")
    log.info("  model: %s", FLUX_PATH)
    log.info("  guidance_scale=1.0  steps=4  seed=%d", seed)

    pipe = DiffusionPipeline.from_pretrained(FLUX_PATH, torch_dtype=torch.bfloat16)
    pipe.enable_model_cpu_offload()

    t0 = time.time()
    result = pipe(
        prompt=prompt,
        guidance_scale=1.0,
        num_inference_steps=4,
        generator=torch.Generator().manual_seed(seed),
    )
    elapsed = time.time() - t0

    img = result.images[0]
    out_path = OUT_DIR / "01_flux.png"
    img.save(str(out_path))
    stats = image_stats(img)

    log.info("  ✓ FLUX done in %.1fs  →  %s", elapsed, out_path.name)
    log.info("  Stats: brightness=%.1f  saturation=%.1f  [%s]",
             stats["brightness"], stats["saturation"], stats["status"])

    # Free VRAM before loading Z-Image
    del pipe, result
    import gc; gc.collect()
    if hasattr(torch.cuda, "empty_cache"):
        torch.cuda.empty_cache()

    return img, stats


# ── Step 2: Z-Image-Turbo refinement ─────────────────────────────────────────

def run_zimage(image, prompt: str, seed: int = 42) -> "PIL.Image.Image":
    """Refine the FLUX image with Z-Image-Turbo.

    Settings used (confirmed fix — guidance_scale must be 0.0 for distilled model):
      guidance_scale = 0.0   (official model card: "Guidance should be 0 for Turbo")
      strength = 0.05        (very light denoising — sharpens without changing content)
      num_inference_steps = 4
    """
    import torch
    from diffusers import AutoPipelineForImage2Image

    log.info("--- Step 2: Z-Image-Turbo refinement ---")
    log.info("  model: %s", ZIMAGE_PATH)
    log.info("  guidance_scale=0.0  strength=0.05  steps=4  seed=%d  [FIXED]", seed)

    pipe = AutoPipelineForImage2Image.from_pretrained(ZIMAGE_PATH, torch_dtype=torch.bfloat16)
    pipe.enable_model_cpu_offload()

    t0 = time.time()
    result = pipe(
        prompt=prompt,
        image=image,
        strength=0.05,
        guidance_scale=0.0,
        num_inference_steps=4,
        generator=torch.Generator().manual_seed(seed),
    )
    elapsed = time.time() - t0

    refined = result.images[0]
    out_path = OUT_DIR / "02_zimage_refined.png"
    refined.save(str(out_path))
    stats = image_stats(refined)

    log.info("  ✓ Z-Image done in %.1fs  →  %s", elapsed, out_path.name)
    log.info("  Stats: brightness=%.1f  saturation=%.1f  [%s]",
             stats["brightness"], stats["saturation"], stats["status"])

    del pipe, result
    import gc; gc.collect()
    if hasattr(torch.cuda, "empty_cache"):
        torch.cuda.empty_cache()

    return refined, stats


# ── Step 3: WAN2.2 video generation ──────────────────────────────────────────

def run_wan22(
    image_path: str,
    prompt: str,
    precision: str,
    steps: int,
    cfg: float,
    svi_python: str,
    seed: int = -1,
) -> str | None:
    """Generate a video segment from the refined image using WAN2.2/SVI.

    Calls vga_svi_inference.py in the svi_wan22 env via subprocess.
    This is exactly how the real pipeline runs it.

    Settings used:
      sigma_shift = 5.0      (FIXED — SVI Python API default; 7.0 had no source)
      denoising_strength = 1.0  (S-08 I2V bootstrap mode — full generation from image)
      cfg_scale = 7.0 (default, overridable with --cfg)
    """
    log.info("--- Step 3: WAN2.2 video generation ---")

    wan_path = WAN22_BF16_PATH if precision == "bf16" else WAN22_FP8_PATH
    log.info("  precision=%s  model=%s", precision.upper(), wan_path)
    log.info("  cfg=%.1f  steps=%d  sigma_shift=5.0  [FIXED]", cfg, steps)

    output_path = str(OUT_DIR / "03_segment.mp4")

    infer_config = {
        "init_image_path": image_path,
        "prompt": prompt,
        "cfg": cfg,
        "steps": steps,
        "lora_path_high": SVI_HIGH_LORA,
        "lora_path_low":  SVI_LOW_LORA,
        "lora_schedule": {
            "high_noise_weight": 0.6,
            "mid_noise_weight":  0.5,
            "low_noise_weight":  0.4,
        },
        "camera_motion": "static camera",
        "motion_vector": "natural gentle movement",
        "output_path": output_path,
        "sigma_shift": 5.0,           # FIXED from 7.0 — SVI Python API default
        "denoising_strength": 0.72,   # used in S-09 bootstrap continuation
        "num_overlap_frames": 5,
        "seed": seed,
        "wan22_precision": precision,
        "wan22_bf16_dir": WAN22_BF16_PATH,
        "ref_image_path": image_path,  # anchor = input image (keeps identity stable)
    }

    # Write config to temp file
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    ) as f:
        json.dump(infer_config, f, indent=2)
        config_path = f.name

    log.info("  Config: %s", config_path)
    log.info("  Running SVI inference (this takes several minutes) ...")

    env = os.environ.copy()
    env["WAN22_PRECISION"] = precision
    env["WAN22_BF16_DIR"] = WAN22_BF16_PATH
    env["SVI_GPU_RESIDENT"] = "1"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    t0 = time.time()
    try:
        result = subprocess.run(
            [svi_python, SVI_INFERENCE, "--config", config_path],
            env=env,
            timeout=1800,
        )
    except subprocess.TimeoutExpired:
        log.error("  WAN2.2 timed out after 30 minutes")
        return None
    finally:
        try:
            os.unlink(config_path)
        except OSError:
            pass

    elapsed = time.time() - t0

    if result.returncode != 0:
        log.error("  WAN2.2 inference FAILED (returncode=%d)", result.returncode)
        return None

    if not Path(output_path).exists():
        log.error("  Output file not written: %s", output_path)
        return None

    size_mb = Path(output_path).stat().st_size / 1e6
    log.info("  ✓ WAN2.2 done in %.1fs  →  %s (%.1f MB)", elapsed, output_path, size_mb)
    return output_path


# ── Report ────────────────────────────────────────────────────────────────────

def write_report(
    args: argparse.Namespace,
    prompt: str,
    flux_stats: dict,
    zimage_stats: dict,
    video_path: str | None,
) -> None:
    lines = [
        "VGA Image/Video Quality Test Report",
        "=" * 60,
        f"Date:        {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Precision:   {args.precision.upper()}",
        f"Steps:       {args.steps}",
        f"CFG:         {args.cfg}",
        f"Prompt:      {prompt}",
        f"Seed:        {args.seed}",
        "",
        "APPLIED FIXES (verified against primary sources):",
        "  Z-Image guidance_scale: 5.0 → 0.0  (official model card requirement)",
        "  sigma_shift: 7.0 → 5.0  (SVI Python API default; 7.0 had no source)",
        "",
        "IMAGE STATS",
        "-" * 40,
        f"  01_flux.png      brightness={flux_stats['brightness']:.1f}  "
        f"saturation={flux_stats['saturation']:.1f}  [{flux_stats['status']}]",
        f"  02_zimage.png    brightness={zimage_stats['brightness']:.1f}  "
        f"saturation={zimage_stats['saturation']:.1f}  [{zimage_stats['status']}]",
        "",
        "REFERENCE (healthy range):",
        "  Brightness : 90–150  (>160 = overexposed,  <70 = underexposed)",
        "  Saturation : 60–110  (>130 = oversaturated / 'over fried')",
        "",
        "VIDEO OUTPUT",
        "-" * 40,
        f"  {'03_segment.mp4  →  ' + str(video_path) if video_path else 'SKIPPED (--image-only) or FAILED'}",
        "",
        "HOW TO INTERPRET",
        "-" * 40,
        "  If FLUX stats are OK but Z-Image is worse → fixed CFG did not fully resolve it.",
        "  If both images are OK but video is over-fried → WAN2.2 is the source.",
        "  If video is OK → fixes resolved the problem, test with full pipeline next.",
        "",
        "NEXT STEP IF VIDEO IS STILL OVER-FRIED",
        "-" * 40,
        "  Test hypothesis: remove 'vivid colors, high contrast, sharp detail'",
        "  from the positive prompt in scripts/vga_svi_inference.py line ~676.",
        "  (unverified — no primary source confirms these terms cause overexposure",
        "   specifically in Wan2.2, but worth testing if sigma_shift fix alone didn't help)",
    ]
    report_path = OUT_DIR / "test_report.txt"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("Report written → %s", report_path)
    print("\n" + "\n".join(lines))


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="VGA direct image+video quality test")
    parser.add_argument(
        "--precision", choices=["bf16", "fp8"], default="bf16",
        help="bf16 = 48GB+ GPU (A6000).  fp8 = 32GB GPU (RTX 5090).",
    )
    parser.add_argument(
        "--image-only", action="store_true",
        help="Stop after Z-Image refinement — skip WAN2.2 video generation.",
    )
    parser.add_argument(
        "--steps", type=int, default=20,
        help="WAN2.2 inference steps. 20=fast test, 30=standard quality. Default: 20",
    )
    parser.add_argument(
        "--cfg", type=float, default=7.0,
        help="WAN2.2 CFG scale. Default: 7.0",
    )
    parser.add_argument(
        "--prompt", default=(
            "cinematic portrait of a determined young man, resilient expression, "
            "warm natural window light, medium shot, photorealistic"
        ),
        help="Text prompt for FLUX image generation.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility. Default: 42",
    )
    parser.add_argument(
        "--svi-python", default=SVI_PYTHON,
        help=f"Path to svi_wan22 conda env Python. Default: {SVI_PYTHON}",
    )
    args = parser.parse_args()

    log.info("=" * 60)
    log.info("VGA Direct Image+Video Test")
    log.info("  precision=%s  steps=%d  cfg=%.1f  image-only=%s",
             args.precision, args.steps, args.cfg, args.image_only)
    log.info("  output: %s", OUT_DIR)
    log.info("=" * 60)

    # Pre-flight checks
    preflight(args.precision, args.image_only, args.svi_python)

    refine_prompt = args.prompt + ", photorealistic, high detail, cinematic"

    # Step 1: FLUX
    flux_img, flux_stats = run_flux(args.prompt, seed=args.seed)

    # Step 2: Z-Image
    refined_img, zimage_stats = run_zimage(flux_img, refine_prompt, seed=args.seed)
    refined_path = str(OUT_DIR / "02_zimage_refined.png")

    # Step 3: WAN2.2 (unless --image-only)
    video_path = None
    if not args.image_only:
        video_path = run_wan22(
            image_path=refined_path,
            prompt=args.prompt,
            precision=args.precision,
            steps=args.steps,
            cfg=args.cfg,
            svi_python=args.svi_python,
            seed=args.seed,
        )
    else:
        log.info("--- Step 3: WAN2.2 SKIPPED (--image-only) ---")

    # Report
    write_report(args, args.prompt, flux_stats, zimage_stats, video_path)

    log.info("=" * 60)
    log.info("Test complete. Outputs in %s", OUT_DIR)
    log.info("  01_flux.png          — FLUX base image")
    log.info("  02_zimage_refined.png — Z-Image refined (guidance=0.0 FIXED)")
    if video_path:
        log.info("  03_segment.mp4       — WAN2.2 video (sigma_shift=5.0 FIXED)")
    log.info("  test_report.txt      — full stats + interpretation")
    log.info("=" * 60)

    return 0 if (args.image_only or video_path is not None) else 1


if __name__ == "__main__":
    sys.exit(main())
