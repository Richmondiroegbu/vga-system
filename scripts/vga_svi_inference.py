#!/usr/bin/env python3
"""
VGA SVI Inference Bridge — runs inside svi_wan22 conda env.
Called by SVIWrapper via subprocess with a JSON config file.

Direct FP8 model loading: loads Wan2.2-I2V-A14B FP8 split-block format
by reconstructing proper key prefixes, bypassing ModelScope/HuggingFace downloads.
Block-wise VRAM management enables both 28GB BF16 DiTs to run on 32GB VRAM.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import torch

SVI_REPO = "/workspace/Stable-Video-Infinity"
if SVI_REPO not in sys.path:
    sys.path.insert(0, SVI_REPO)

WAN22_DIR = "/workspace/models/wan22"


def extract_last_frame(video_path: str, output_image_path: str) -> bool:
    """Extract last frame from a video file and save as JPEG."""
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


def load_fp8_state_dict(fp8_dir: str) -> dict:
    """Load Wan2.2 FP8 split-block model, reconstructing proper key prefixes.

    Keeps tensors in their native FP8 dtype (float8_e4m3fn) to minimize RAM.
    Two 14GB FP8 DiTs = 28GB total, well within the 43GB cgroup limit.
    VRAM management casts FP8→BF16 during computation (per-block, on-demand).

    FP8 files layout:
      blocks.N.safetensors  (N=0..39): block tensors WITHOUT block prefix
      head.safetensors: head.bias, head.weight, modulation (WITH head. prefix)
      patch_embedding.safetensors, text_embedding.safetensors, etc.

    WanModel state_dict key mapping:
      blocks.N.<key>           ← blocks.N.safetensors/<key>
      head.head.bias/weight    ← head.safetensors/head.bias, head.weight
      head.modulation          ← head.safetensors/modulation
      patch_embedding.<key>    ← patch_embedding.safetensors/<key>
      text_embedding.<key>     ← text_embedding.safetensors/<key>
      time_embedding.<key>     ← time_embedding.safetensors/<key>
      time_projection.<key>    ← time_projection.safetensors/<key>
    """
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
        ("head.safetensors", "head."),  # head.bias→head.head.bias, modulation→head.modulation
    ]:
        with safe_open(str(fp8_dir / fname), framework="pt", device="cpu") as f:
            for key in f.keys():
                state_dict[f"{prefix}{key}"] = f.get_tensor(key)

    return state_dict


def apply_vram_management(model: torch.nn.Module, device: str = "cuda") -> None:
    """Apply DiffSynth block-wise VRAM management to a WanModel with FP8 weights.

    DiTs are kept in FP8 on CPU (14GB each). Each DiTBlock is wrapped in
    AutoWrappedNonRecurseModule which, during forward(), casts the block from
    FP8 CPU → BF16 CUDA for computation, then the copy is freed. This keeps
    VRAM usage to ~700MB per active block vs 28GB for the full BF16 model.

    Memory budget (43GB cgroup limit):
      - Two FP8 DiTs on CPU: 14 + 14 = 28GB
      - Peak VRAM per timestep: 1 block × 700MB BF16 ≈ 700MB
    """
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

    # Keep FP8 model on CPU for offload and onload.
    # computation_device="cuda" and computation_dtype=bfloat16 forces cast_to()
    # in AutoWrappedModule.computation(), which deep-copies each block to CUDA BF16.
    vram_config = {
        "offload_dtype": torch.float8_e4m3fn,
        "offload_device": "cpu",
        "onload_dtype": torch.float8_e4m3fn,
        "onload_device": "cpu",          # Stay on CPU — cast_to handles GPU transfer
        "preparing_dtype": torch.float8_e4m3fn,
        "preparing_device": "cpu",
        "computation_dtype": torch.bfloat16,
        "computation_device": device,    # GPU: cast_to creates BF16 copy here
    }
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


def load_wan_dit_fp8(fp8_dir: str, device: str = "cuda") -> "WanModel":
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
    del state_dict  # explicitly release dict (model now owns the tensors)

    print(f"  Applying block-wise VRAM management (FP8→BF16 on-demand)...")
    apply_vram_management(model, device=device)

    model = model.eval()
    return model


def build_pipeline(lora_path_high: str, lora_path_low: str, device: str = "cuda") -> "WanVideoSviPipeline":
    """Build WanVideoSviPipeline with local FP8 DiTs + BF16 T5/VAE."""
    from diffsynth.pipelines.wan_video_svi import WanVideoSviPipeline
    from diffsynth.core.loader.config import ModelConfig
    from diffsynth.models.wan_video_text_encoder import HuggingfaceTokenizer

    pipe = WanVideoSviPipeline(device=device, torch_dtype=torch.bfloat16)

    # === Load high-noise and low-noise DiTs (FP8 → BF16 with VRAM management) ===
    print("Loading high-noise DiT (FP8)...")
    pipe.dit = load_wan_dit_fp8(f"{WAN22_DIR}/high_noise_model_fp8", device=device)

    print("Loading low-noise DiT (FP8)...")
    pipe.dit2 = load_wan_dit_fp8(f"{WAN22_DIR}/low_noise_model_fp8", device=device)

    # === Load T5 text encoder and VAE via standard DiffSynth (single files) ===
    print("Loading T5 text encoder and VAE...")
    t5_config = ModelConfig(path=f"{WAN22_DIR}/models_t5_umt5-xxl-enc-bf16.pth", offload_device="cpu")
    vae_config = ModelConfig(path=f"{WAN22_DIR}/Wan2.1_VAE.pth", offload_device="cpu")
    model_pool = pipe.download_and_load_models([t5_config, vae_config])
    pipe.text_encoder = model_pool.fetch_model("wan_video_text_encoder")
    pipe.vae = model_pool.fetch_model("wan_video_vae")

    # === Load tokenizer from local path ===
    print("Loading tokenizer...")
    pipe.tokenizer = HuggingfaceTokenizer(
        name=f"{WAN22_DIR}/google/umt5-xxl",
        seq_len=512,
        clean="whitespace",
    )

    # === Apply SVI LoRAs for temporal consistency ===
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


def main() -> int:
    parser = argparse.ArgumentParser(description="VGA SVI Inference Bridge")
    parser.add_argument("--config", required=True, help="Path to JSON config file")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"ERROR: Config not found: {config_path}")
        return 1

    config = json.loads(config_path.read_text(encoding="utf-8"))

    output_path = Path(config["output_path"])
    prev_segment_path = config.get("prev_segment_path", "")
    prompt = config.get("prompt", "cinematic scene, photorealistic")
    lora_path_high = config["lora_path_high"]
    lora_path_low = config["lora_path_low"]
    cfg = float(config.get("cfg", 5.5))
    steps = int(config.get("steps", 12))
    camera_motion = config.get("camera_motion", "static")
    motion_vector = config.get("motion_vector", "stationary")

    if not prev_segment_path or not Path(prev_segment_path).exists():
        print(f"ERROR: Previous segment not found: {prev_segment_path}")
        return 1

    output_path.parent.mkdir(parents=True, exist_ok=True)

    ref_image_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            ref_image_path = f.name
        print(f"Extracting last frame from {Path(prev_segment_path).name}...")
        if not extract_last_frame(prev_segment_path, ref_image_path):
            return 1

        from PIL import Image
        ref_image = Image.open(ref_image_path).convert("RGB").resize((832, 480))

        print("Building SVI pipeline...")
        pipe = build_pipeline(lora_path_high, lora_path_low)

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

        print(f"Running SVI inference: cfg={cfg:.2f} steps={steps} → {output_path.name}")
        video_frames = pipe(
            prompt=full_prompt,
            negative_prompt=negative_prompt,
            input_image=ref_image,
            height=480,
            width=832,
            num_frames=81,
            cfg_scale=cfg,
            num_inference_steps=steps,
            tiled=False,
            seed=42,
        )

        from diffsynth.utils.data import save_video
        save_video(video_frames, str(output_path), fps=15, quality=5)
        print(f"SUCCESS: Segment saved to {output_path}")
        return 0

    except Exception as e:
        import traceback
        print(f"ERROR: SVI inference failed: {e}")
        traceback.print_exc()
        return 1

    finally:
        if ref_image_path and os.path.exists(ref_image_path):
            try:
                os.unlink(ref_image_path)
            except OSError:
                pass


if __name__ == "__main__":
    sys.exit(main())
