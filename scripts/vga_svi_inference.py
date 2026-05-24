#!/usr/bin/env python3
"""
VGA SVI Inference Bridge — runs inside svi_wan22 conda env.
Called by SVIWrapper via subprocess with a JSON config file.

Two operation modes
-------------------
  S-08 (Segment_1):  init_image_path provided → I2V from single character image.

  S-09+ (Segments 2..N): prev_segment_path provided → TRUE SVI CONTINUATION.
    Extracts the last `num_overlap_frames` (default 4) frames from the previous
    segment and passes them as `input_video` to WanVideoSviPipeline.
    This gives the model MULTI-FRAME temporal context so each new segment
    continues naturally from where the previous ended.

    WRONG (old): extract 1 frame → input_image  (= fresh I2V, no motion context)
    RIGHT:       extract N frames → input_video  (= SVI temporal continuation)

    The first `num_overlap_frames` of each continuation segment will closely
    match the last `num_overlap_frames` of the previous segment, providing
    the seamless join. The caller (test_pipeline.py / AssemblyAgent) must
    trim those overlap frames when concatenating segments into the final video.

Direct FP8 model loading: loads Wan2.2-I2V-A14B FP8 split-block format
by reconstructing proper key prefixes, bypassing ModelScope/HuggingFace.
"""
from __future__ import annotations

import argparse
import json
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

# Default overlap: 4 frames shared between consecutive segments.
# Assembly trims the first 4 frames of each continuation segment.
DEFAULT_OVERLAP_FRAMES = 4


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
    cfg = float(config.get("cfg", 5.5))
    steps = int(config.get("steps", 12))
    camera_motion = config.get("camera_motion", "static")
    motion_vector = config.get("motion_vector", "stationary")
    tea_cache_thresh = float(config.get("tea_cache_l1_thresh", 0.1))
    num_overlap_frames = int(config.get("num_overlap_frames", DEFAULT_OVERLAP_FRAMES))
    # denoising_strength for continuation mode (S-09+):
    #   1.0 = full noise, ignores input_video entirely (WRONG — fresh generation each segment)
    #   0.75 = moderate noise, preserves structure from overlap frames (CORRECT — true continuation)
    # S-08 I2V mode always uses 1.0 (we want full generation from the still image).
    denoising_strength_continuation = float(config.get("denoising_strength", 0.75))

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

    # Cinematic prompt — purely visual/descriptive, no character names (names cause
    # DiffSynth to render floating text rather than animate the character).
    full_prompt = (
        f"{prompt}, {camera_motion}, motion: {motion_vector}, "
        "cinematic quality, photorealistic, temporal consistency"
    )
    negative_prompt = (
        "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，"
        "最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，"
        "画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，"
        "杂乱的背景，三条腿，背景人很多，倒着走"
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
        )

        if init_image_path and Path(init_image_path).exists():
            # ── S-08 MODE: I2V from single character image ──────────────────
            # The init_image is the refined character portrait from S-07.
            # DiffSynth animates this image into 81 frames.
            print(f"[S-08 I2V] Using init image: {Path(init_image_path).name}")
            ref_image = Image.open(init_image_path).convert("RGB").resize((832, 480), Image.LANCZOS)
            call_kwargs["input_image"] = ref_image

        else:
            # ── S-09+ MODE: TRUE SVI CONTINUATION via input_video ───────────
            # Extract last N frames of previous segment as PIL Images.
            # These become the conditioning context for the new segment.
            # The first N frames of the output will naturally continue from
            # those conditioning frames → seamless temporal join.
            print(f"[S-09 CONTINUATION] Extracting {num_overlap_frames} overlap frames "
                  f"from {Path(prev_segment_path).name}...")
            overlap_frames_pil = extract_last_n_frames_as_pil(
                prev_segment_path, n=num_overlap_frames
            )
            if not overlap_frames_pil:
                return {
                    "status": "error",
                    "error": f"Failed to extract overlap frames from {prev_segment_path}",
                }
            # Pass as input_video — the SVI pipeline uses these for temporal continuity.
            # Do NOT set input_image for continuation (that would reset to I2V mode).
            # denoising_strength<1.0 is CRITICAL: at 1.0 (default) the pipeline adds full
            # noise and ignores input_video entirely. At 0.75, it preserves the coarse
            # latent structure from the overlap frames, producing true seamless continuation.
            call_kwargs["input_video"] = overlap_frames_pil
            call_kwargs["denoising_strength"] = denoising_strength_continuation
            print(f"  Conditioning on {len(overlap_frames_pil)} overlap frames (input_video, "
                  f"denoising_strength={denoising_strength_continuation})")

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
