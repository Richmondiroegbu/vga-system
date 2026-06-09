#!/usr/bin/env python3
"""
diag_image_quality.py — VGA Image Quality Diagnostic

Generates reference images using FLUX and Z-Image-Turbo with BOTH the current
(potentially broken) settings and the corrected settings so you can compare
side-by-side before deciding whether to update the pipeline.

Run on the RunPod pod:
    cd /workspace
    python3 scripts/diag_image_quality.py --prompt "cinematic portrait of a determined young man"

Outputs (all in /workspace/output/diag/):
    01_flux_current.png        — FLUX with current settings (guidance=1.0, steps=4)
    02_zimage_current.png      — Z-Image on flux output, CURRENT guidance=5.0 (suspect)
    03_zimage_fixed.png        — Z-Image on flux output, FIXED guidance=0.0 (confirmed correct)
    04_zimage_fixed_stronger.png — Z-Image guidance=0.0, strength=0.10 (vs 0.05)
    diag_report.txt            — brightness / saturation stats for each output

CONFIRMED fixes already applied to the codebase:
    - Z-Image guidance_scale: 5.0 → 0.0  (confirmed by official model card)
    - sigma_shift: 7.0 → 5.0  (confirmed by SVI Python API default; 7.0 has no source)

Interpretation:
    - If 01 looks fine but 02 is oversaturated → Z-Image CFG was the culprit (confirmed)
    - If 01 already looks over-fried → problem is in FLUX (unlikely given BFL spec)
    - If 01–04 all look fine → "over fried" is coming from WAN2.2 alone
      → Next step: test WAN2.2 with sigma_shift=5.0 (already applied)
      → Also test hypothesis: remove "vivid colors, high contrast, sharp detail"
        from positive prompt in vga_svi_inference.py (unverified, needs testing)
"""
from __future__ import annotations

import argparse
import sys
import os
from pathlib import Path

OUT_DIR = Path("/workspace/output/diag")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def measure_stats(img) -> dict:
    """Measure mean brightness and mean saturation for a PIL image."""
    import numpy as np
    import cv2
    arr = np.array(img.convert("RGB"))
    brightness = float(arr.mean())
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV).astype(np.float32)
    saturation = float(hsv[:, :, 1].mean())
    return {"brightness": round(brightness, 1), "saturation": round(saturation, 1)}


def run_flux(prompt: str, seed: int = 42):
    """Run FLUX.2-klein-4B with current settings (guidance=1.0, steps=4)."""
    from diffusers import DiffusionPipeline
    import torch

    model_path = "/workspace/models/flux2"
    print(f"[FLUX] Loading from {model_path} ...")
    pipe = DiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.bfloat16)
    pipe.enable_model_cpu_offload()

    print(f"[FLUX] Generating: guidance_scale=1.0 steps=4 seed={seed}")
    result = pipe(
        prompt=prompt,
        guidance_scale=1.0,   # CORRECT for distilled FLUX
        num_inference_steps=4,
        generator=torch.Generator().manual_seed(seed),
    )
    img = result.images[0]
    out = OUT_DIR / "01_flux_current.png"
    img.save(str(out))
    stats = measure_stats(img)
    print(f"[FLUX] Saved → {out.name}  stats={stats}")
    return img, stats


def run_zimage(
    input_img,
    prompt: str,
    guidance_scale: float,
    strength: float,
    tag: str,
    seed: int = 42,
):
    """Run Z-Image-Turbo with given guidance_scale and strength."""
    from diffusers import AutoPipelineForImage2Image
    import torch

    model_path = "/workspace/models/zimage"
    print(f"[Z-Image] Loading from {model_path} ...")
    pipe = AutoPipelineForImage2Image.from_pretrained(model_path, torch_dtype=torch.bfloat16)
    pipe.enable_model_cpu_offload()

    print(f"[Z-Image] Generating: guidance={guidance_scale} strength={strength} tag={tag}")
    result = pipe(
        prompt=prompt,
        image=input_img,
        strength=strength,
        guidance_scale=guidance_scale,
        num_inference_steps=4,
        generator=torch.Generator().manual_seed(seed),
    )
    img = result.images[0]
    out = OUT_DIR / f"{tag}.png"
    img.save(str(out))
    stats = measure_stats(img)
    print(f"[Z-Image] Saved → {out.name}  stats={stats}")
    return img, stats


def write_report(results: list[dict]) -> None:
    """Write diag_report.txt with stats table and recommendations."""
    report_path = OUT_DIR / "diag_report.txt"
    lines = [
        "VGA Image Quality Diagnostic Report",
        "=" * 60,
        "",
        f"{'Image':<30} {'Brightness':>12} {'Saturation':>12}",
        "-" * 56,
    ]
    for r in results:
        lines.append(
            f"{r['tag']:<30} {r['stats']['brightness']:>12.1f} {r['stats']['saturation']:>12.1f}"
        )
    lines += [
        "",
        "Reference ranges (healthy output):",
        "  Brightness : 90–150  (>160 = overexposed, <70 = underexposed)",
        "  Saturation : 60–110  (>130 = oversaturated / 'over fried')",
        "",
        "Interpretation:",
        "  01_flux_current    — baseline FLUX output; if >150 bright or >120 sat → FLUX issue",
        "  02_zimage_current  — CURRENT pipeline (guidance=5.0); if much brighter/more saturated",
        "                       than 01 → Z-Image CFG is causing the 'over fried' look",
        "  03_zimage_fixed    — CORRECTED Z-Image (guidance=0.0); should match 01 closely",
        "  04_zimage_stronger — guidance=0.0, strength=0.10; slight refinement, still controlled",
        "",
        "If 01–04 all look healthy → 'over fried' comes from WAN2.2 alone.",
        "See WAN2.2 fixes in vga_svi_inference.py (prompt + sigma_shift + cfg).",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n[Report] Written → {report_path}")
    print("\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser(description="VGA Image Quality Diagnostic")
    parser.add_argument(
        "--prompt",
        default=(
            "cinematic portrait of a determined young man with expressive eyes, "
            "warm natural lighting, medium shot, photorealistic"
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    prompt = args.prompt
    seed = args.seed
    refine_prompt = prompt + ", photorealistic, high detail, cinematic"

    results = []

    # 1. FLUX baseline
    flux_img, flux_stats = run_flux(prompt, seed=seed)
    results.append({"tag": "01_flux_current", "stats": flux_stats})

    # 2. Z-Image with CURRENT broken settings (guidance=5.0)
    _, stats = run_zimage(flux_img, refine_prompt, guidance_scale=5.0, strength=0.05,
                          tag="02_zimage_current", seed=seed)
    results.append({"tag": "02_zimage_current", "stats": stats})

    # 3. Z-Image with FIXED settings (guidance=0.0)
    _, stats = run_zimage(flux_img, refine_prompt, guidance_scale=0.0, strength=0.05,
                          tag="03_zimage_fixed", seed=seed)
    results.append({"tag": "03_zimage_fixed", "stats": stats})

    # 4. Z-Image fixed + slightly stronger refinement
    _, stats = run_zimage(flux_img, refine_prompt, guidance_scale=0.0, strength=0.10,
                          tag="04_zimage_fixed_stronger", seed=seed)
    results.append({"tag": "04_zimage_fixed_stronger", "stats": stats})

    write_report(results)
    print(f"\n[Done] All outputs in {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
