#!/usr/bin/env python3
"""
VGA SVI Inference Bridge — runs inside svi_wan22 conda env.
Called by SVIWrapper via subprocess with a JSON config file.

Two operation modes
-------------------
  S-08 (Segment_1):  init_image_path provided → I2V from single character image.

  S-09+ (Segments 2..N): prev_segment_path provided → TRUE SVI CONTINUATION.
    The pipeline requires 36-channel input to its patch_embedding:
      - 16ch: video latents  (from input_video → WanVideoUnit_InputVideoEmbedder)
      - 16ch: image VAE      (from input_image → WanVideoUnit_ImageEmbedderVAE)
      -  4ch: temporal mask  (generated internally from input_image path)
      ─────────────────────────────────────────────────────────────────────────
      = 36ch total ✓

    BOTH input_image AND input_video must be provided together:
      input_image = last frame of prev segment  → supplies the 20ch `y` (mask+VAE)
      input_video = last N frames of prev segment → supplies the 16ch video latents
      denoising_strength = 0.60 → model starts 60% through the noise schedule,
        preserving coarse structure more tightly (reduces face warping vs 0.75)
        while still allowing new motion in the remaining 81-N frames.

    The first `num_overlap_frames` output frames closely replicate the end of
    the previous segment. Assembly trims them to prevent visual duplication.

    PREVIOUSLY BROKEN APPROACHES:
      • Only input_video (no input_image): 16ch only → "expected 36 channels" crash
      • Only input_image (no input_video): fresh I2V from a still frame each time
        → each segment is identical regeneration, quality fades, no motion continuity

Direct FP8 model loading: loads Wan2.2-I2V-A14B FP8 split-block format
by reconstructing proper key prefixes, bypassing ModelScope/HuggingFace.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import tempfile
from pathlib import Path

import torch

SVI_REPO = "/workspace/Stable-Video-Infinity"
if SVI_REPO not in sys.path:
    sys.path.insert(0, SVI_REPO)

WAN22_DIR = "/workspace/models/wan22"

# Default overlap: 5 frames shared between consecutive segments.
# Assembly trims the first 5 frames of each continuation segment.
# These 5 frames are also hard-replaced post-inference with the exact
# source frames for pixel-identical seams.
DEFAULT_OVERLAP_FRAMES = 5

# Wan2.2 VAE temporal stride = 4. With N input_video frames, the encoded latent
# has (N-1)//4 + 1 temporal frames. Passing >4 frames gives 2 latents, which
# mismatches the pipeline's internal concatenation → crash. 4 is the max that
# gives exactly 1 temporal latent, which the SVI pipeline expects for input_video.
_MAX_INPUT_VIDEO_FRAMES = 4


def verify_and_configure_attention() -> bool:
    """Verify FlashAttention-2 availability and configure PyTorch SDPA backend."""
    if hasattr(torch.backends.cuda, "enable_flash_sdp"):
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(False)
        print("  PyTorch SDPA: Flash+MemEfficient ENABLED, math (slow) disabled")
    else:
        print("  PyTorch SDPA: enable_flash_sdp not available (PyTorch too old?)")

    try:
        import flash_attn
        fa_version = getattr(flash_attn, "__version__", "unknown")
        print(f"  FlashAttention-2: INSTALLED (version={fa_version})")
        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability()
            gpu_name = torch.cuda.get_device_name(0)
            print(f"  GPU: {gpu_name} — compute capability {major}.{minor}")
            if (major, minor) >= (8, 0):
                print(f"  FlashAttention-2: GPU supports FA2 (sm_{major}{minor} >= sm_80) ✓")
            else:
                print(f"  FlashAttention-2: WARNING — sm_{major}{minor} < sm_80, FA2 may not work")
        return True
    except ImportError:
        print("  FlashAttention-2: NOT INSTALLED — using PyTorch SDPA (still fast on RTX 5090)")
        return False


def extract_last_frame(video_path: str, output_image_path: str) -> bool:
    """Extract last frame from a video file and save as JPEG (kept for diagnostics)."""
    try:
        import cv2
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, total - 1))
        ret, frame = cap.read()
        cap.release()
        if not ret:
            print(f"ERROR: Could not read frame from {video_path}")
            return False
        cv2.imwrite(output_image_path, frame)
        return True
    except Exception as e:
        print(f"ERROR extracting frame: {e}")
        return False


def extract_last_n_frames_as_pil(video_path: str, n: int) -> list:
    """Extract the last n frames from a video and return as list of PIL Images.

    Used for SVI continuation mode (input_video parameter).
    Each returned image is resized to 832×480 (Wan2.2 inference resolution).
    """
    from PIL import Image
    frames = []
    try:
        import cv2
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total == 0:
            print(f"ERROR: {video_path} has 0 frames")
            return frames
        start = max(0, total - n)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb).resize((832, 480), Image.LANCZOS)
            frames.append(pil_img)
        cap.release()
        print(f"  Extracted {len(frames)} overlap frames from {Path(video_path).name}")
    except Exception as e:
        print(f"ERROR extracting frames as PIL: {e}")
    return frames


def enhance_frame_sequence_contrast(
    frames: list,
    clahe_clip_limit: float = 2.0,
    clahe_tile_size: int = 8,
    target_mean: float = 130.0,
    target_saturation: float = 90.0,   # HSV S-channel mean target (0-255 scale)
    max_scale: float = 1.35,
    min_scale: float = 0.80,
    max_sat_scale: float = 2.0,        # clamp: never more than 2× saturation boost
) -> list:
    """CLAHE + saturation restoration + global mean normalization.

    FP8 DiT weights cause two cascading failure modes:
      1. Luminance collapse: pixels cluster toward mean → fog/haze (fixed by CLAHE)
      2. Saturation collapse: chroma drains toward grey across segments → sepia →
         greyscale. CLAHE operates on L channel only and does NOT restore saturation.
         Without explicit saturation correction the 5-frame overlap passed to the next
         segment carries the desaturation forward, compounding each segment.

    Three-stage fix:
      1. CLAHE in LAB L-channel: restores local contrast, preserves hue/saturation.
      2. HSV saturation restoration: rescales S channel so the sequence mean reaches
         target_saturation (~90), breaking the cross-segment desaturation cascade.
      3. Global mean normalization: anchors brightness to target_mean=130.

    Per-sequence scaling (stages 2 and 3) preserves intra-segment variation while
    resetting both exposure and saturation for the next segment's conditioning frames.
    """
    import cv2
    import numpy as np
    from PIL import Image as _Image

    if not frames:
        return frames

    # Stage 1: CLAHE in LAB color space — restores local contrast
    clahe = cv2.createCLAHE(
        clipLimit=clahe_clip_limit,
        tileGridSize=(clahe_tile_size, clahe_tile_size),
    )
    enhanced_arrs = []
    for frame in frames:
        arr = np.array(frame)
        lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
        l_ch, a_ch, b_ch = cv2.split(lab)
        l_enhanced = clahe.apply(l_ch)
        lab_enhanced = cv2.merge([l_enhanced, a_ch, b_ch])
        enhanced_arrs.append(cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB))

    # Stage 2: HSV saturation restoration — breaks the cross-segment desaturation cascade.
    # FP8 quantization drains chroma: segment tails are sepia/grey, poisoning next segment's
    # conditioning. Re-saturate before saving so conditioning frames look vivid.
    hsv_arrs = [cv2.cvtColor(a, cv2.COLOR_RGB2HSV).astype(np.float32) for a in enhanced_arrs]
    sat_mean = float(np.mean([a[:, :, 1].mean() for a in hsv_arrs]))
    if sat_mean > 5.0:  # skip if already near-greyscale source
        sat_scale = float(np.clip(target_saturation / max(sat_mean, 1.0), 1.0, max_sat_scale))
        if sat_scale > 1.02:
            sat_restored = []
            for hsv in hsv_arrs:
                hsv_c = hsv.copy()
                hsv_c[:, :, 1] = np.clip(hsv_c[:, :, 1] * sat_scale, 0, 255)
                sat_restored.append(cv2.cvtColor(hsv_c.astype(np.uint8), cv2.COLOR_HSV2RGB))
            print(f"  Saturation: sat_mean={sat_mean:.1f} → target={target_saturation:.0f} scale={sat_scale:.3f}")
        else:
            sat_restored = enhanced_arrs
            print(f"  Saturation: sat_mean={sat_mean:.1f} already adequate, no boost")
    else:
        sat_restored = enhanced_arrs
        print(f"  Saturation: sat_mean={sat_mean:.1f} near-grey source, skipping boost")

    # Stage 3: global mean normalization — prevents brightness drift across segments
    float_arrs = [a.astype(np.float32) for a in sat_restored]
    seq_mean = float(np.mean([a.mean() for a in float_arrs]))

    if seq_mean < 5.0:
        print(f"  Brightness: CLAHE applied, sequence too dark ({seq_mean:.1f}), skipping scale")
        return [_Image.fromarray(a) for a in sat_restored]

    scale = float(np.clip(target_mean / seq_mean, min_scale, max_scale))
    print(f"  Brightness: seq_mean={seq_mean:.1f} → target={target_mean:.0f} scale={scale:.3f}")

    if abs(scale - 1.0) < 0.02:
        return [_Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)) for a in float_arrs]

    return [
        _Image.fromarray(np.clip(a * scale, 0, 255).astype(np.uint8))
        for a in float_arrs
    ]


def load_fp8_state_dict(fp8_dir: str) -> dict:
    """Load Wan2.2 FP8 split-block model, reconstructing proper key prefixes."""
    from safetensors import safe_open

    state_dict = {}
    fp8_dir = Path(fp8_dir)

    print(f"  Loading 40 FP8 transformer blocks (keeping FP8 dtype)...")
    for n in range(40):
        block_file = fp8_dir / f"blocks.{n}.safetensors"
        with safe_open(str(block_file), framework="pt", device="cpu") as f:
            for key in f.keys():
                state_dict[f"blocks.{n}.{key}"] = f.get_tensor(key)

    print(f"  Loading component files...")
    for fname, prefix in [
        ("patch_embedding.safetensors", "patch_embedding."),
        ("text_embedding.safetensors", "text_embedding."),
        ("time_embedding.safetensors", "time_embedding."),
        ("time_projection.safetensors", "time_projection."),
        ("head.safetensors", "head."),
    ]:
        with safe_open(str(fp8_dir / fname), framework="pt", device="cpu") as f:
            for key in f.keys():
                state_dict[f"{prefix}{key}"] = f.get_tensor(key)

    return state_dict


def apply_vram_management(
    model: torch.nn.Module,
    device: str = "cuda",
    gpu_resident: bool = True,
) -> None:
    """Apply DiffSynth block-wise VRAM management to a WanModel with FP8 weights."""
    from diffsynth.core.vram.layers import (
        enable_vram_management,
        AutoWrappedModule,
        AutoWrappedNonRecurseModule,
    )
    from diffsynth.models.wan_video_dit import DiTBlock, Head, MLP, RMSNorm
    import torch.nn as nn

    try:
        from diffsynth.core.vram.layers import AutoWrappedLinear
    except ImportError:
        AutoWrappedLinear = AutoWrappedModule

    if gpu_resident:
        vram_config = {
            "offload_dtype": torch.float8_e4m3fn,
            "offload_device": device,
            "onload_dtype": torch.float8_e4m3fn,
            "onload_device": device,
            "preparing_dtype": torch.float8_e4m3fn,
            "preparing_device": device,
            "computation_dtype": torch.bfloat16,
            "computation_device": device,
        }
        print(f"  VRAM mode: GPU-resident FP8 (no PCIe per block, {device})")
    else:
        vram_config = {
            "offload_dtype": torch.float8_e4m3fn,
            "offload_device": "cpu",
            "onload_dtype": torch.float8_e4m3fn,
            "onload_device": "cpu",
            "preparing_dtype": torch.float8_e4m3fn,
            "preparing_device": "cpu",
            "computation_dtype": torch.bfloat16,
            "computation_device": device,
        }
        print(f"  VRAM mode: CPU-offload FP8 (960 PCIe round-trips at 12 steps)")

    module_map = {
        DiTBlock: AutoWrappedNonRecurseModule,
        Head: AutoWrappedModule,
        MLP: AutoWrappedModule,
        nn.Linear: AutoWrappedLinear,
        nn.Conv3d: AutoWrappedModule,
        nn.LayerNorm: AutoWrappedModule,
        RMSNorm: AutoWrappedModule,
        nn.Conv2d: AutoWrappedModule,
    }
    enable_vram_management(model, module_map, vram_config)
    model.vram_management_enabled = True


def load_wan_dit_fp8(
    fp8_dir: str,
    device: str = "cuda",
    gpu_resident: bool = True,
) -> "WanModel":
    """Load Wan2.2-I2V-A14B DiT model from FP8 split-block format with VRAM management."""
    from diffsynth.models.wan_video_dit import WanModel
    from diffsynth.core.vram.initialization import skip_model_initialization

    config = {
        "has_image_input": False,
        "patch_size": [1, 2, 2],
        "in_dim": 36,
        "dim": 5120,
        "ffn_dim": 13824,
        "freq_dim": 256,
        "text_dim": 4096,
        "out_dim": 16,
        "num_heads": 40,
        "num_layers": 40,
        "eps": 1e-06,
        "require_clip_embedding": False,
    }

    print(f"  Initializing WanModel ({Path(fp8_dir).name})...")
    with skip_model_initialization():
        model = WanModel(**config)

    print(f"  Loading FP8 state dict from {fp8_dir} (native FP8 dtype, no cast)...")
    state_dict = load_fp8_state_dict(fp8_dir)

    missing, unexpected = model.load_state_dict(state_dict, strict=False, assign=True)
    if missing:
        print(f"  WARNING: {len(missing)} missing keys (first 5): {missing[:5]}")
    if unexpected:
        print(f"  WARNING: {len(unexpected)} unexpected keys (first 5): {unexpected[:5]}")
    del state_dict

    mode_label = "GPU-resident FP8" if gpu_resident else "CPU-offload FP8→BF16"
    print(f"  Applying VRAM management ({mode_label})...")
    apply_vram_management(model, device=device, gpu_resident=gpu_resident)

    if gpu_resident:
        vram_gb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e9
        print(f"  Moving FP8 DiT to {device} (one-time {vram_gb:.1f}GB transfer)...")
        model = model.to(device)

    model = model.eval()
    return model


def build_pipeline(lora_path_high: str, lora_path_low: str, device: str = "cuda") -> "WanVideoSviPipeline":
    """Build WanVideoSviPipeline with local FP8 DiTs + BF16 T5/VAE."""
    from diffsynth.pipelines.wan_video_svi import WanVideoSviPipeline
    from diffsynth.core.loader.config import ModelConfig
    from diffsynth.models.wan_video_text_encoder import HuggingfaceTokenizer

    gpu_resident = os.environ.get("SVI_GPU_RESIDENT", "0") != "0"

    pipe = WanVideoSviPipeline(device=device, torch_dtype=torch.bfloat16)

    print("Loading T5 text encoder and VAE (offloads to CPU)...")
    t5_config = ModelConfig(path=f"{WAN22_DIR}/models_t5_umt5-xxl-enc-bf16.pth", offload_device="cpu")
    vae_config = ModelConfig(path=f"{WAN22_DIR}/Wan2.1_VAE.pth", offload_device="cpu")
    model_pool = pipe.download_and_load_models([t5_config, vae_config])
    pipe.text_encoder = model_pool.fetch_model("wan_video_text_encoder")
    pipe.vae = model_pool.fetch_model("wan_video_vae")

    print(f"Loading high-noise DiT (FP8, gpu_resident={gpu_resident})...")
    pipe.dit = load_wan_dit_fp8(
        f"{WAN22_DIR}/high_noise_model_fp8", device=device, gpu_resident=gpu_resident
    )

    print(f"Loading low-noise DiT (FP8, gpu_resident={gpu_resident})...")
    pipe.dit2 = load_wan_dit_fp8(
        f"{WAN22_DIR}/low_noise_model_fp8", device=device, gpu_resident=gpu_resident
    )

    print("Loading tokenizer...")
    pipe.tokenizer = HuggingfaceTokenizer(
        name=f"{WAN22_DIR}/google/umt5-xxl",
        seq_len=512,
        clean="whitespace",
    )

    if os.path.exists(lora_path_high):
        print(f"Applying high-noise LoRA: {Path(lora_path_high).name}")
        pipe.load_lora(pipe.dit, lora_path_high, alpha=1)
    else:
        print(f"WARNING: High-noise LoRA not found at {lora_path_high}, skipping")

    if os.path.exists(lora_path_low):
        print(f"Applying low-noise LoRA: {Path(lora_path_low).name}")
        pipe.load_lora(pipe.dit2, lora_path_low, alpha=1)
    else:
        print(f"WARNING: Low-noise LoRA not found at {lora_path_low}, skipping")

    pipe.vram_management_enabled = pipe.check_vram_management_state()
    print(f"Pipeline ready. VRAM management: {pipe.vram_management_enabled}")
    return pipe


def run_inference(pipe: "WanVideoSviPipeline", config: dict) -> dict:
    """Execute one SVI inference request against an already-loaded pipeline.

    Two modes
    ---------
    S-08 (init_image_path set):
        I2V from a single character image. The pipeline generates 81 frames
        that animate the character starting from the provided image.

    S-09+ (prev_segment_path set):
        TRUE SVI CONTINUATION. The last `num_overlap_frames` (default 4) frames
        of the previous segment are extracted and passed as `input_video`.
        The pipeline generates 81 frames where the first ~4 closely continue
        from the conditioning frames, creating a seamless join.

        The caller must trim the first `num_overlap_frames` when assembling
        the final video to avoid visual duplication at segment boundaries.
    """
    from PIL import Image

    output_path = Path(config["output_path"])
    prev_segment_path = config.get("prev_segment_path", "")
    init_image_path = config.get("init_image_path", "")
    prompt = config.get("prompt", "cinematic scene, photorealistic")
    cfg = float(config.get("cfg", 7.0))
    steps = int(config.get("steps", 12))
    camera_motion = config.get("camera_motion", "static")
    motion_vector = config.get("motion_vector", "stationary")
    tea_cache_thresh = 0.0  # Wan2.2 has no calibrated TeaCache rescale coefficients; Wan2.1 coefficients cause gray output
    num_overlap_frames = int(config.get("num_overlap_frames", DEFAULT_OVERLAP_FRAMES))
    # denoising_strength for continuation mode (S-09+):
    #   1.0 = full noise, ignores input_video entirely (WRONG — fresh generation each segment)
    #   0.75 = moderate noise, preserves structure but allows heavy face reconstruction → warping
    #   0.60 = lower noise, tighter adherence to overlap frames → sharper faces, less warping
    # S-08 I2V mode always uses 1.0 (we want full generation from the still image).
    denoising_strength_continuation = float(config.get("denoising_strength", 0.60))
    # ref_image_path: optional path to original reference image (S-07 refined portrait).
    # When provided, passed as `anchor` to WanVideoSviPipeline.__call__().
    # Keeps original character visible in all attention layers of every S-09 segment.
    # Prevents progressive scene drift by keeping the original character reference visible
    # to the model even in deep continuation segments.
    ref_image_path = config.get("ref_image_path", "")

    # Seed: -1 or absent → random (important for continuation segments to vary)
    seed_val = config.get("seed", -1)
    if seed_val is None or int(seed_val) < 0:
        seed_val = random.randint(0, 2**31 - 1)
        print(f"  Using random seed: {seed_val}")
    else:
        seed_val = int(seed_val)
        print(f"  Using fixed seed: {seed_val}")

    if not init_image_path and (not prev_segment_path or not Path(prev_segment_path).exists()):
        return {
            "status": "error",
            "error": f"Neither init_image_path nor valid prev_segment_path provided: {prev_segment_path}",
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Will be populated in S-09+ mode; used for hard frame replacement after inference.
    overlap_frames_pil: list | None = None

    # Cinematic prompt — purely visual/descriptive, no character names (names cause
    # DiffSynth to render floating text rather than animate the character).
    # "temporal consistency" removed from positive prompt: it encourages frozen/static
    # frames. Motion language added explicitly to drive character and scene dynamics.
    full_prompt = (
        f"{prompt}, {camera_motion}, motion: {motion_vector}, "
        "natural human movement, expressive gestures, dynamic body language, "
        "vivid colors, high contrast, sharp detail, cinematic quality, photorealistic"
    )
    # Added anti-static and anti-dim terms to reinforce motion and exposure quality.
    negative_prompt = (
        "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，"
        "最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，"
        "画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，"
        "杂乱的背景，三条腿，背景人很多，倒着走，"
        "frozen pose, no movement, static character, dim lighting, underexposed, "
        "dark frame, hazy, foggy, low visibility, washed out, faded, "
        "desaturated, sepia tone, sepia filter, grayscale, black and white, monochrome, "
        "color faded, dull colors, muted colors, color drained"
    )

    try:
        # === Base call kwargs (common to both modes) ===
        call_kwargs: dict = dict(
            prompt=full_prompt,
            negative_prompt=negative_prompt,
            height=480,
            width=832,
            num_frames=81,
            cfg_scale=cfg,
            num_inference_steps=steps,
            tiled=False,
            seed=seed_val,
            # sigma_shift=5.0: official recommended value for Wan2.2 + SVI.
            # Controls the noise schedule shift — lower values preserve motion/dynamics,
            # higher values favor static detail. 5.0 is the vita-epfl default for svi_wan22.
            sigma_shift=5.0,
        )
        # num_motion_latent is an __init__ param of SviPipeline wrapper, NOT a __call__ param.
        # WanVideoSviPipeline.__call__ does not accept it — it was removed to fix TypeError.

        if init_image_path and Path(init_image_path).exists():
            # ── S-08 MODE: Bootstrap via 5-frame PIL zoom-in seed → S-09 continuation ──
            # Still image (repeated 4×) provides no temporal motion signal → model
            # generates static content that fades to gray over 81 frames.
            # Solution: create a 5-frame zoom-in (0%→6%, PIL only, no model), save
            # as seed.mp4, then call run_inference() recursively in S-09 mode.
            # S-09 gets real motion dynamics from the zoom and produces vivid output.
            print(f"[S-08 BOOTSTRAP] Init image: {Path(init_image_path).name}")
            ref_image = Image.open(init_image_path).convert("RGB").resize((832, 480), Image.LANCZOS)

            # 0%→6% zoom gives the model a clear forward-motion signal strong enough
            # to propagate dynamic movement across all 81 generated frames.
            # 0→2% was too subtle — model treated it as near-static and generated stasis.
            print("[S-08] Generating 5-frame zoom-in seed (0%→6% zoom, PIL only)...")
            seed_frames_pil = []
            for _i in range(5):
                _zoom = 1.0 + _i * 0.015
                _nw, _nh = int(832 * _zoom), int(480 * _zoom)
                _enl = ref_image.resize((_nw, _nh), Image.LANCZOS)
                _left, _top = (_nw - 832) // 2, (_nh - 480) // 2
                seed_frames_pil.append(_enl.crop((_left, _top, _left + 832, _top + 480)))

            from diffsynth.utils.data import save_video as _sv_seed
            _seed_path = output_path.parent / f"_s08_seed_{output_path.stem}.mp4"
            _sv_seed(seed_frames_pil, str(_seed_path), fps=15, quality=5)
            print(f"[S-08] Seed saved: {_seed_path.name} — 5 vivid zoom frames → S-09 input")

            _s09_config = dict(config)
            _s09_config["prev_segment_path"] = str(_seed_path)
            _s09_config.pop("init_image_path", None)
            # Preserve the original reference image so subsequent S-09 segments can
            # inject it as random_ref_frame, anchoring cross-attention to the original
            # character throughout all continuation segments.
            if "ref_image_path" not in _s09_config or not _s09_config["ref_image_path"]:
                _s09_config["ref_image_path"] = init_image_path
            _result = run_inference(pipe, _s09_config)
            if _seed_path.exists():
                _seed_path.unlink()
            return _result

        else:
            # ── S-09+ MODE: TRUE SVI CONTINUATION ────────────────────────────
            # The DiT patch_embedding requires 36-channel input, assembled by
            # the pipeline from two sources:
            #   input_image (last frame)  → y tensor: 4ch mask + 16ch VAE = 20ch
            #   input_video (last N frames) → latents: 16ch video latents
            #   Together: 20 + 16 = 36ch ✓
            #
            # denoising_strength=0.75: the scheduler adds noise only at 75% of
            # its timestep budget, so input_video's structure (the last N frames
            # of the previous segment) is preserved coarsely. The model then
            # denoises all 81 output frames, continuing naturally from that
            # structure — this is the core SVI temporal continuation mechanism.
            print(f"[S-09 SVI CONTINUATION] Extracting last {num_overlap_frames} frames "
                  f"from {Path(prev_segment_path).name} for seam conditioning + hard replacement...")
            overlap_frames_pil = extract_last_n_frames_as_pil(
                prev_segment_path, n=num_overlap_frames
            )
            if not overlap_frames_pil:
                return {
                    "status": "error",
                    "error": f"Failed to extract overlap frames from {prev_segment_path}",
                }
            last_frame = overlap_frames_pil[-1]  # last frame → image conditioning (y)
            # The Wan2.2 VAE temporal stride is 4: N frames → (N-1)//4 + 1 latents.
            # input_video must map to exactly 1 latent (max 4 frames).
            # We extract num_overlap_frames (5) for hard replacement but only pass
            # the last _MAX_INPUT_VIDEO_FRAMES (4) to the pipeline for conditioning.
            conditioning_frames = overlap_frames_pil[-_MAX_INPUT_VIDEO_FRAMES:]

            # ── CAMERA ANGLE TRANSITION (multi-reference I2V) ──────────────────
            # transition_mode:
            #   "none"     — normal continuation (default)
            #   "hard_cut" — Strategy A: switch input_image to new angle ref, ds=0.90.
            #                Model gets maximum freedom to adopt the new viewpoint.
            #                Sharp editorial cut style. Best for action cuts.
            #   "blend"    — Strategy C: pixel-space cosine-ramp blend of the last 4
            #                conditioning frames toward new angle ref (α 0→0.25).
            #                Soft cross-dissolve over 10-20 frames. Best for motivated
            #                scene changes (character turns, reveals).
            # In both modes: input_image hard-switches to new angle reference;
            # anchor stays as original S-07 character ref (identity lock).
            transition_mode = config.get("transition_mode", "none")
            new_angle_ref_path = config.get("new_angle_ref_image", "")

            if transition_mode in ("hard_cut", "blend") and new_angle_ref_path and Path(new_angle_ref_path).exists():
                new_ref_img = Image.open(new_angle_ref_path).convert("RGB").resize((832, 480), Image.LANCZOS)

                if transition_mode == "blend":
                    # Strategy C: cosine-ramp pixel blend of conditioning frames.
                    # Cosine ramp stays low longer than linear — more old-angle motion
                    # signal in early frames, gentle rise toward new angle at frame 4.
                    # α_i = max_alpha * (1 - cos(π * i / n)) / 2
                    # At n=4: frame 0 → 0.0, frame 1 → 0.037, frame 2 → 0.125, frame 3 → 0.25
                    max_alpha = float(config.get("transition_blend_max_alpha", 0.25))
                    n_cond = len(conditioning_frames)
                    blended_conditioning = []
                    for i, old_f in enumerate(conditioning_frames):
                        alpha = max_alpha * (1.0 - math.cos(math.pi * i / n_cond)) / 2.0
                        blended_conditioning.append(Image.blend(old_f, new_ref_img, alpha))
                    conditioning_frames = blended_conditioning
                    denoising_strength_continuation = float(
                        config.get("transition_ds", 0.80)
                    )
                    print(
                        f"  [STRATEGY-C BLEND] Cosine-ramp blend conditioning frames "
                        f"(max_alpha={max_alpha:.2f}, n={n_cond}), "
                        f"ds={denoising_strength_continuation}"
                    )
                else:
                    # Strategy A: hard cut — only switch input_image; keep raw
                    # conditioning frames so the model has residual motion context,
                    # but elevated ds=0.90 largely overrides that context anyway.
                    denoising_strength_continuation = float(
                        config.get("transition_ds", 0.90)
                    )
                    print(
                        f"  [STRATEGY-A HARD CUT] New angle ref: {Path(new_angle_ref_path).name}, "
                        f"ds={denoising_strength_continuation}"
                    )

                # Both strategies: switch input_image to new angle reference.
                # anchor (ref_image_path) stays as original S-07 portrait for identity lock.
                last_frame = new_ref_img
                print(f"  input_image switched to new angle: {Path(new_angle_ref_path).name}")

            elif transition_mode != "none":
                print(
                    f"  WARNING: transition_mode={transition_mode!r} but new_angle_ref_image "
                    f"not found ({new_angle_ref_path!r}) — falling back to normal continuation"
                )

            call_kwargs["input_image"] = last_frame           # 20ch: mask + VAE encoding
            call_kwargs["input_video"] = conditioning_frames  # 16ch: video latents (max 4)
            call_kwargs["denoising_strength"] = denoising_strength_continuation
            print(f"  input_image: last frame (image conditioning, 20ch)")
            print(f"  input_video: {len(conditioning_frames)} conditioning frames (VAE constraint, 16ch)")
            print(f"  hard replace: first {len(overlap_frames_pil)} output frames ← exact source frames")
            print(f"  36ch total = 16 + 20 ✓  denoising_strength={denoising_strength_continuation}")

            # Inject original reference image as anchor into every S-09 segment.
            # WanVideoSviPipeline.__call__ uses `anchor` (not `random_ref_frame`).
            # The anchor is preprocessed into a padded latent sequence, keeping the
            # original character visible in all attention layers — prevents progressive
            # scene drift caused by each segment conditioning only on its degraded predecessor.
            if ref_image_path and Path(ref_image_path).exists():
                try:
                    _ref_img = Image.open(ref_image_path).convert("RGB").resize((832, 480), Image.LANCZOS)
                    call_kwargs["anchor"] = _ref_img
                    print(f"  anchor: {Path(ref_image_path).name} injected (character anchor for all attention layers)")
                except Exception as _e:
                    print(f"  anchor: failed to load {ref_image_path}: {_e} — skipping")

        print(f"Running SVI inference: cfg={cfg:.2f} steps={steps} seed={seed_val} → {output_path.name}")

        tea_model_id = "Wan2.1-I2V-14B-480P"
        if tea_cache_thresh > 0.0:
            try:
                video_frames = pipe(
                    tea_cache_l1_thresh=tea_cache_thresh,
                    tea_cache_model_id=tea_model_id,
                    **call_kwargs,
                )
                print(f"  TeaCache enabled (thresh={tea_cache_thresh}, id={tea_model_id}) — ~2-3x speedup")
            except (TypeError, ValueError):
                print("  TeaCache: not supported by this pipeline build, running standard inference")
                video_frames = pipe(**call_kwargs)
        else:
            video_frames = pipe(**call_kwargs)

        # ── HARD FRAME REPLACEMENT (S-09+ only) ──────────────────────────────
        # The model's first `num_overlap_frames` output frames condition on the
        # previous segment's tail but are NOT pixel-identical to it — the model
        # still denoises them, producing slight drift. Hard-replacing them with
        # the exact source frames guarantees a pixel-perfect seam regardless of
        # what the model generated. Assembly then trims these frames so they are
        # invisible in the final video; they exist only to provide continuity at
        # the join point.
        if overlap_frames_pil is not None and isinstance(video_frames, list):
            n = min(num_overlap_frames, len(overlap_frames_pil), len(video_frames))
            video_frames[:n] = overlap_frames_pil[:n]
            print(f"  Hard-replaced first {n} frames with exact source frames "
                  f"(pixel-identical seam, trimmed during assembly)")
        elif overlap_frames_pil is not None:
            print(f"  WARNING: video_frames type {type(video_frames)} — "
                  f"skipping hard replacement; seam relies on model conditioning only")

        # ── EXPOSURE NORMALIZATION (applied AFTER hard replacement) ───────────
        # Normalizes the full 81-frame sequence to a consistent mean brightness
        # BEFORE writing to disk. This means when the next segment reads this
        # file for conditioning, it extracts bright (normalized) overlap frames
        # rather than the raw dim tail — preventing compounding brightness drift
        # across the 4 autoregressive segments.
        # Applied after hard replacement so the overlap frames (0-4, trimmed by
        # assembly) are also normalized; their brightness is visible to the next
        # segment's conditioning extractor.
        if isinstance(video_frames, list):
            video_frames = enhance_frame_sequence_contrast(video_frames, target_mean=130.0)

        from diffsynth.utils.data import save_video
        save_video(video_frames, str(output_path), fps=15, quality=5)
        print(f"SUCCESS: Segment saved to {output_path}")
        return {
            "status": "ok",
            "output_path": str(output_path),
            "num_overlap_frames": num_overlap_frames,
            "seed": seed_val,
        }

    except Exception as e:
        import traceback
        msg = traceback.format_exc()
        print(f"ERROR: SVI inference failed: {e}\n{msg}")
        return {"status": "error", "error": str(e), "traceback": msg}


def main() -> int:
    parser = argparse.ArgumentParser(description="VGA SVI Inference Bridge")
    parser.add_argument("--config", required=True, help="Path to JSON config file")
    args = parser.parse_args()

    print("=== VGA SVI Inference Bridge ===")
    verify_and_configure_attention()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"ERROR: Config not found: {config_path}")
        return 1

    config = json.loads(config_path.read_text(encoding="utf-8"))

    lora_path_high = config.get("lora_path_high", "")
    lora_path_low = config.get("lora_path_low", "")

    init_image_path = config.get("init_image_path", "")
    prev_segment_path = config.get("prev_segment_path", "")
    if not init_image_path and not Path(prev_segment_path).exists():
        print(f"ERROR: Neither init_image_path nor prev_segment_path found: {prev_segment_path}")
        return 1

    try:
        print("Building SVI pipeline...")
        pipe = build_pipeline(lora_path_high, lora_path_low)
        result = run_inference(pipe, config)
        if result["status"] == "ok":
            return 0
        print(f"ERROR: {result.get('error', 'unknown')}")
        return 1
    except Exception as e:
        import traceback
        print(f"ERROR: Pipeline setup failed: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
