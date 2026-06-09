#!/usr/bin/env python3
"""
test_image_video.py — Direct pipeline stage test: S-05 → S-07 → S-08 → S-09

Mirrors exactly the image-to-video spine of the full pipeline with NO orchestration
(no ImmutableContext, no HRG, no agents, no Qwen). Goes directly to model inference.

Stage mapping:
  S-05  BaseImageAgent      → FLUX.2-klein-4B        → character base image
  S-07  ImageRefinementAgent → Z-Image-Turbo          → refined character image
  S-08  VideoSegmentGenerator → WAN2.2 I2V (SVI)     → segment 001 (first, from image)
  S-09  TemporalEngine/SVI   → SVI continuation loop → segments 002, 003, ... (from video)

Confirmed fixes applied in this script:
  • Z-Image guidance_scale: 5.0 → 0.0  (official model card, distilled model)
  • sigma_shift: 7.0 → 5.0             (SVI Python API default, 7.0 had no source)

Usage:
    # Images only (examine S-05 + S-07 output first — cheapest test)
    python3 scripts/test_image_video.py --image-only

    # S-05 → S-07 → S-08 only (first video segment from image)
    python3 scripts/test_image_video.py --precision bf16 --segments 0

    # Full test: S-05 → S-07 → S-08 → S-09 x1 continuation segment
    python3 scripts/test_image_video.py --precision bf16

    # Full test with 2 continuation segments (S-09 x2)
    python3 scripts/test_image_video.py --precision bf16 --segments 2

    # RTX 5090 (32GB) — use fp8 precision
    python3 scripts/test_image_video.py --precision fp8

Outputs saved to /workspace/output/test/<RUN_ID>/:
    S05_base_image.png          — S-05: raw FLUX.2-klein-4B output
    S07_refined_image.png       — S-07: Z-Image-Turbo refined (guidance=0.0 FIXED)
    S08_segment_001.mp4         — S-08: WAN2.2 I2V first segment (from image)
    S09_segment_002.mp4         — S-09: SVI continuation segment 2
    S09_segment_003.mp4         — S-09: SVI continuation segment 3 (if --segments 2)
    test_report.txt             — brightness/saturation stats + parameters used

Then download to local machine:
    python3 scripts/download_test_outputs.py
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
import uuid
from pathlib import Path

# ── RUN ID ─────────────────────────────────────────────────────────────────────
# Each test run gets a unique subdirectory so multiple runs don't overwrite each other.
RUN_ID = time.strftime("%Y%m%d_%H%M%S")

OUT_DIR = Path(f"/workspace/output/test/{RUN_ID}")
OUT_DIR.mkdir(parents=True, exist_ok=True)
Path("/workspace/logs").mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f"/workspace/logs/test_image_video_{RUN_ID}.log", mode="w"),
    ],
)
log = logging.getLogger("test_image_video")

# ── Model paths ────────────────────────────────────────────────────────────────
FLUX_PATH        = "/workspace/models/flux2"
ZIMAGE_PATH      = "/workspace/models/zimage"
WAN22_BF16_PATH  = "/workspace/models/wan22_bf16"
WAN22_FP8_PATH   = "/workspace/models/wan22"
SVI_HIGH_LORA    = "/workspace/loras/svi/SVI_Wan2.2-I2V-A14B_high_noise_lora_v2.0_pro.safetensors"
SVI_LOW_LORA     = "/workspace/loras/svi/SVI_Wan2.2-I2V-A14B_low_noise_lora_v2.0_pro.safetensors"
SVI_INFERENCE    = "/workspace/scripts/vga_svi_inference.py"
SVI_PYTHON       = "/opt/conda/envs/svi_wan22/bin/python3"


# ── Stats helpers ──────────────────────────────────────────────────────────────

def image_stats(img) -> dict:
    """Mean brightness and HSV saturation for a PIL image."""
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


def _exposure_label(b: float, s: float) -> str:
    if b > 180:   return "OVEREXPOSED"
    if b > 155:   return "BRIGHT (borderline)"
    if b < 60:    return "UNDEREXPOSED"
    if s > 140:   return "OVERSATURATED"
    if s > 120:   return "SATURATED (borderline)"
    return "OK"


# ── Pre-flight ─────────────────────────────────────────────────────────────────

def preflight(precision: str, image_only: bool, segments: int, svi_python: str) -> None:
    """Verify all required models/scripts exist. Fail fast with clear messages."""
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
                f"SVI Python missing: {svi_python}\n"
                "  Run: python3 scripts/test_minimal_bootstrap.py first"
            )

    if errors:
        log.error("Pre-flight FAILED:")
        for e in errors:
            log.error("  ✗ %s", e)
        sys.exit(1)

    log.info("Pre-flight OK — all models found.")


# ── S-05: FLUX base image ──────────────────────────────────────────────────────

def run_s05_flux(prompt: str, seed: int) -> tuple:
    """S-05 equivalent: BaseImageAgent → FLUX.2-klein-4B → base character image.

    Params (BFL spec for distilled model):
      guidance_scale = 1.0
      num_inference_steps = 4
    """
    import torch
    from diffusers import DiffusionPipeline

    log.info("━━━ S-05: FLUX base image generation ━━━")
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
    out_path = OUT_DIR / "S05_base_image.png"
    img.save(str(out_path))
    stats = image_stats(img)

    log.info("  ✓ S-05 done in %.1fs → %s", elapsed, out_path.name)
    log.info("  brightness=%.1f  saturation=%.1f  [%s]",
             stats["brightness"], stats["saturation"], stats["status"])

    del pipe, result
    _free_vram(torch)
    return img, stats


# ── S-07: Z-Image refinement ───────────────────────────────────────────────────

def run_s07_zimage(image, prompt: str, seed: int) -> tuple:
    """S-07 equivalent: ImageRefinementAgent → Z-Image-Turbo → refined character image.

    Params (FIXED — confirmed by official Z-Image-Turbo model card):
      guidance_scale = 0.0   (DISTILLED model — guidance is baked into weights.
                               Using CFG > 0 applies it twice → oversaturation.)
      strength = 0.05        (light refinement — sharpens without changing content)
      num_inference_steps = 4
    """
    import torch
    from diffusers import AutoPipelineForImage2Image

    log.info("━━━ S-07: Z-Image-Turbo refinement ━━━")
    log.info("  model: %s", ZIMAGE_PATH)
    log.info("  guidance_scale=0.0 [FIXED from 5.0]  strength=0.05  steps=4  seed=%d", seed)

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
    out_path = OUT_DIR / "S07_refined_image.png"
    refined.save(str(out_path))
    stats = image_stats(refined)

    log.info("  ✓ S-07 done in %.1fs → %s", elapsed, out_path.name)
    log.info("  brightness=%.1f  saturation=%.1f  [%s]",
             stats["brightness"], stats["saturation"], stats["status"])

    del pipe, result
    _free_vram(torch)
    return refined, stats


# ── S-08: WAN2.2 first segment (I2V from image) ───────────────────────────────

def run_s08_wan22(
    refined_image_path: str,
    prompt: str,
    precision: str,
    steps: int,
    cfg: float,
    svi_python: str,
    seed: int,
) -> str | None:
    """S-08 equivalent: VideoSegmentGenerator → WAN2.2 I2V → segment 001.

    S-08 mode: init_image_path provided, no prev_segment_path.
    vga_svi_inference.py uses denoising_strength=1.0 for this mode (full I2V).

    Params:
      sigma_shift = 5.0   (FIXED from 7.0 — SVI Python API default)
      denoising_strength = 1.0  (full generation from still image)
    """
    log.info("━━━ S-08: WAN2.2 I2V first segment ━━━")
    wan_path = WAN22_BF16_PATH if precision == "bf16" else WAN22_FP8_PATH
    log.info("  precision=%s  model=%s", precision.upper(), wan_path)
    log.info("  cfg=%.1f  steps=%d  sigma_shift=5.0 [FIXED]  denoising_strength=1.0", cfg, steps)
    log.info("  input: %s", Path(refined_image_path).name)

    output_path = str(OUT_DIR / "S08_segment_001.mp4")

    config = {
        # S-08 mode: init_image_path set, no prev_segment_path
        "init_image_path": refined_image_path,
        "ref_image_path": refined_image_path,   # identity anchor
        "prompt": prompt,
        "cfg": cfg,
        "steps": steps,
        "lora_path_high": SVI_HIGH_LORA,
        "lora_path_low": SVI_LOW_LORA,
        "lora_schedule": {
            "high_noise_weight": 0.6,
            "mid_noise_weight":  0.5,
            "low_noise_weight":  0.4,
        },
        "camera_motion": "static camera",
        "motion_vector": "natural gentle movement",
        "output_path": output_path,
        "sigma_shift": 5.0,         # FIXED from 7.0
        "num_overlap_frames": 5,
        "seed": seed,
        "wan22_precision": precision,
        "wan22_bf16_dir": WAN22_BF16_PATH,
    }

    return _run_svi_inference(config, svi_python, precision, "S-08", output_path)


# ── S-09: SVI continuation segments ───────────────────────────────────────────

def run_s09_svi(
    prev_segment_path: str,
    refined_image_path: str,
    segment_num: int,
    prompt: str,
    precision: str,
    steps: int,
    cfg: float,
    svi_python: str,
    seed: int,
) -> str | None:
    """S-09 equivalent: TemporalEngine → SVI continuation → segment N.

    S-09 mode: prev_segment_path + ref_image_path (36-channel input to DiT):
      - last 5 frames of prev_segment → input_video → 16ch video latents
      - last frame of prev_segment    → input_image → 20ch (4 mask + 16 VAE)
      Total: 36ch ✓

    Params:
      denoising_strength = 0.72  (balanced continuation, tighter than 0.75)
      sigma_shift = 5.0          (FIXED from 7.0)
      num_overlap_frames = 5     (TEMPORAL_BUFFER_SIZE — pipeline constant)
    """
    log.info("━━━ S-09: SVI continuation segment %03d ━━━", segment_num)
    log.info("  prev_segment: %s", Path(prev_segment_path).name)
    log.info("  cfg=%.1f  steps=%d  sigma_shift=5.0 [FIXED]  denoising_strength=0.72", cfg, steps)

    output_path = str(OUT_DIR / f"S09_segment_{segment_num:03d}.mp4")

    config = {
        # S-09 mode: prev_segment_path set (both input_video AND input_image extracted from it)
        "prev_segment_path": prev_segment_path,
        "ref_image_path": refined_image_path,   # original S-07 anchor — prevents identity drift
        "prompt": prompt,
        "cfg": cfg,
        "steps": steps,
        "lora_path_high": SVI_HIGH_LORA,
        "lora_path_low": SVI_LOW_LORA,
        "lora_schedule": {
            "high_noise_weight": 0.6,
            "mid_noise_weight":  0.5,
            "low_noise_weight":  0.4,
        },
        "camera_motion": "static camera",
        "motion_vector": "natural gentle movement",
        "output_path": output_path,
        "sigma_shift": 5.0,           # FIXED from 7.0
        "denoising_strength": 0.72,   # S-09 continuation mode
        "num_overlap_frames": 5,      # TEMPORAL_BUFFER_SIZE
        "seed": -1,                   # random per segment (fixed seed → identical segments)
        "wan22_precision": precision,
        "wan22_bf16_dir": WAN22_BF16_PATH,
    }

    return _run_svi_inference(config, svi_python, precision, f"S-09 seg{segment_num:03d}", output_path)


# ── SVI subprocess runner ─────────────────────────────────────────────────────

def _run_svi_inference(
    config: dict,
    svi_python: str,
    precision: str,
    label: str,
    output_path: str,
) -> str | None:
    """Write config to temp file, call vga_svi_inference.py via subprocess."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    ) as f:
        json.dump(config, f, indent=2)
        config_path = f.name

    env = os.environ.copy()
    env["WAN22_PRECISION"] = precision
    env["WAN22_BF16_DIR"] = WAN22_BF16_PATH
    env["SVI_GPU_RESIDENT"] = "1"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    log.info("  Running SVI inference (takes several minutes) ...")
    t0 = time.time()
    try:
        proc = subprocess.run(
            [svi_python, SVI_INFERENCE, "--config", config_path],
            env=env,
            timeout=1800,
        )
    except subprocess.TimeoutExpired:
        log.error("  %s timed out after 30 minutes", label)
        return None
    finally:
        try:
            os.unlink(config_path)
        except OSError:
            pass

    elapsed = time.time() - t0

    if proc.returncode != 0:
        log.error("  %s FAILED (returncode=%d)", label, proc.returncode)
        return None

    if not Path(output_path).exists():
        log.error("  %s: output not written: %s", label, output_path)
        return None

    size_mb = Path(output_path).stat().st_size / 1e6
    log.info("  ✓ %s done in %.1fs → %s (%.1f MB)", label, elapsed,
             Path(output_path).name, size_mb)
    return output_path


def _free_vram(torch) -> None:
    import gc
    gc.collect()
    if hasattr(torch.cuda, "empty_cache"):
        torch.cuda.empty_cache()


# ── Report ─────────────────────────────────────────────────────────────────────

def write_report(
    args: argparse.Namespace,
    prompt: str,
    s05_stats: dict,
    s07_stats: dict,
    segments: list[str | None],
) -> None:
    lines = [
        "VGA Pipeline Stage Test Report",
        "=" * 60,
        f"Run ID:       {RUN_ID}",
        f"Date:         {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Precision:    {args.precision.upper()}",
        f"Steps:        {args.steps}",
        f"CFG:          {args.cfg}",
        f"Seed:         {args.seed}",
        f"Prompt:       {prompt}",
        "",
        "APPLIED FIXES (verified against primary sources):",
        "  [S-07] Z-Image guidance_scale: 5.0 → 0.0",
        "         Source: official Tongyi-MAI/Z-Image-Turbo model card",
        "         Reason: distilled model — CFG > 0 applies guidance twice → oversaturation",
        "  [S-08/S-09] sigma_shift: 7.0 → 5.0",
        "         Source: SVI Python API default (vita-epfl/Stable-Video-Infinity)",
        "         Reason: 7.0 was unverified forum speculation",
        "",
        "STAGE IMAGE STATS",
        "-" * 60,
        f"  {'Stage':<8}  {'File':<30}  {'Brightness':>10}  {'Saturation':>10}  {'Status'}",
        f"  {'-'*8}  {'-'*30}  {'-'*10}  {'-'*10}  {'-'*20}",
        f"  {'S-05':<8}  {'S05_base_image.png':<30}  "
        f"{s05_stats['brightness']:>10.1f}  {s05_stats['saturation']:>10.1f}  {s05_stats['status']}",
        f"  {'S-07':<8}  {'S07_refined_image.png':<30}  "
        f"{s07_stats['brightness']:>10.1f}  {s07_stats['saturation']:>10.1f}  {s07_stats['status']}",
        "",
        "REFERENCE RANGES (healthy output):",
        "  Brightness: 90–150  (>160 = overexposed,  <70 = underexposed)",
        "  Saturation: 60–110  (>130 = oversaturated / 'over fried')",
        "",
        "VIDEO SEGMENTS",
        "-" * 60,
    ]

    seg_labels = ["S-08 (first segment, I2V from image)"] + \
                 [f"S-09 continuation (segment {i+2})" for i in range(len(segments) - 1)]
    seg_files  = ["S08_segment_001.mp4"] + \
                 [f"S09_segment_{i+2:03d}.mp4" for i in range(len(segments) - 1)]

    for label, fname, path in zip(seg_labels, seg_files, segments):
        if path and Path(path).exists():
            mb = Path(path).stat().st_size / 1e6
            lines.append(f"  ✓ {label:<45}  {fname}  ({mb:.1f} MB)")
        else:
            lines.append(f"  ✗ {label:<45}  FAILED or SKIPPED")

    lines += [
        "",
        "HOW TO INTERPRET",
        "-" * 60,
        "  S-05 bright/saturated → FLUX is the source (unlikely, BFL spec is correct)",
        "  S-05 OK, S-07 worse   → Z-Image CFG fix did not fully resolve it",
        "  S-05 + S-07 OK, video over-fried → WAN2.2/SVI is the source",
        "  All OK                → fixes resolved the problem; test with full pipeline",
        "",
        "IF VIDEO IS STILL OVER-FRIED:",
        "  Unverified hypothesis: 'vivid colors, high contrast, sharp detail' in the",
        "  positive prompt (vga_svi_inference.py ~line 676) may amplify saturation.",
        "  Test by removing those terms and comparing outputs.",
        "",
        f"Output directory: /workspace/output/test/{RUN_ID}/",
        "Download to local: python3 scripts/download_test_outputs.py",
    ]

    report_path = OUT_DIR / "test_report.txt"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(lines))
    log.info("Report → %s", report_path)


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="VGA pipeline stage test: S-05 → S-07 → S-08 → S-09",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--precision", choices=["bf16", "fp8"], default="bf16",
        help="bf16 = 48GB+ GPU (A6000).  fp8 = 32GB GPU (RTX 5090). Default: bf16",
    )
    parser.add_argument(
        "--image-only", action="store_true",
        help="Run only S-05 and S-07 (FLUX + Z-Image). Skip all video generation.",
    )
    parser.add_argument(
        "--segments", type=int, default=1,
        help=(
            "Number of S-09 SVI continuation segments to generate AFTER the S-08 "
            "first segment. 0 = S-08 only. 1 = S-08 + 1 continuation (default). "
            "2 = S-08 + 2 continuations. Each segment takes ~5-15 min."
        ),
    )
    parser.add_argument(
        "--steps", type=int, default=20,
        help="Inference steps per segment. 20=fast test, 30=standard quality. Default: 20",
    )
    parser.add_argument(
        "--cfg", type=float, default=7.0,
        help="WAN2.2 CFG scale. Default: 7.0",
    )
    parser.add_argument(
        "--prompt",
        default=(
            "cinematic portrait of a determined young man, resilient expression, "
            "warm natural window light, medium shot, photorealistic"
        ),
        help="Text prompt used for all stages.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for S-05 and S-07 (reproducibility). Default: 42",
    )
    parser.add_argument(
        "--svi-python", default=SVI_PYTHON,
        help=f"Path to svi_wan22 conda env Python. Default: {SVI_PYTHON}",
    )
    args = parser.parse_args()

    log.info("=" * 60)
    log.info("VGA Pipeline Stage Test  [Run ID: %s]", RUN_ID)
    log.info("  Stages:    S-05 → S-07%s",
             "" if args.image_only else f" → S-08 → S-09 x{args.segments}")
    log.info("  Precision: %s  Steps: %d  CFG: %.1f",
             args.precision.upper(), args.steps, args.cfg)
    log.info("  Output:    %s", OUT_DIR)
    log.info("=" * 60)

    preflight(args.precision, args.image_only, args.segments, args.svi_python)

    refine_prompt = args.prompt + ", photorealistic, high detail, cinematic"

    # ── S-05: FLUX ──────────────────────────────────────────────────────────────
    base_img, s05_stats = run_s05_flux(args.prompt, seed=args.seed)

    # ── S-07: Z-Image ───────────────────────────────────────────────────────────
    _, s07_stats = run_s07_zimage(base_img, refine_prompt, seed=args.seed)
    del base_img  # free memory
    refined_path = str(OUT_DIR / "S07_refined_image.png")

    # ── Video stages ─────────────────────────────────────────────────────────────
    video_segments: list[str | None] = []

    if args.image_only:
        log.info("━━━ Video generation SKIPPED (--image-only) ━━━")
    else:
        # S-08: first segment (I2V from image)
        s08_path = run_s08_wan22(
            refined_image_path=refined_path,
            prompt=args.prompt,
            precision=args.precision,
            steps=args.steps,
            cfg=args.cfg,
            svi_python=args.svi_python,
            seed=args.seed,
        )
        video_segments.append(s08_path)

        if s08_path is None:
            log.error("S-08 failed — cannot run S-09 without a valid first segment.")
        else:
            # S-09: continuation segments
            prev_path = s08_path
            for i in range(args.segments):
                seg_num = i + 2   # segment 002, 003, ...
                s09_path = run_s09_svi(
                    prev_segment_path=prev_path,
                    refined_image_path=refined_path,
                    segment_num=seg_num,
                    prompt=args.prompt,
                    precision=args.precision,
                    steps=args.steps,
                    cfg=args.cfg,
                    svi_python=args.svi_python,
                    seed=args.seed,
                )
                video_segments.append(s09_path)
                if s09_path:
                    prev_path = s09_path  # next continuation uses this segment as input

    # ── Report ────────────────────────────────────────────────────────────────────
    write_report(args, args.prompt, s05_stats, s07_stats, video_segments)

    log.info("=" * 60)
    log.info("Test complete.")
    log.info("  Output dir:  %s", OUT_DIR)
    log.info("  To download: python3 scripts/download_test_outputs.py")
    log.info("=" * 60)

    failed = any(p is None for p in video_segments)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
