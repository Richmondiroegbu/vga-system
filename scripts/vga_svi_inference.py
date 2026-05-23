#!/usr/bin/env python3
"""
VGA SVI Inference Bridge — runs inside svi_wan22 conda env.
Called by SVIWrapper via subprocess with a JSON config file.

Direct FP8 model loading: loads Wan2.2-I2V-A14B FP8 split-block format
by reconstructing proper key prefixes, bypassing ModelScope/HuggingFace downloads.

GPU-resident DiT mode (SVI_GPU_RESIDENT=1, default ON):
  Both FP8 DiTs (14GB each = 28GB total) stay in GPU VRAM the entire inference.
  FP8→BF16 casts happen on-GPU with no PCIe transfers per block, eliminating
  the ~960 PCIe round-trips that were the primary inference bottleneck.
  RTX 5090 (32GB) fits both DiTs (28GB) + activations (~2GB).

CPU-offload mode (SVI_GPU_RESIDENT=0):
  Legacy mode: each DiTBlock is moved CPU→GPU per timestep (safe but slow).
  Use this if VRAM OOM occurs in GPU-resident mode.
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


def verify_and_configure_attention() -> bool:
    """Verify FlashAttention-2 availability and configure PyTorch SDPA backend.

    Returns True if flash_attn package is importable.
    """
    # Configure PyTorch SDPA backends. Flash Attention is fastest but requires
    # head_dim <= 256. The VAE uses head_dim=384, so it falls back to
    # mem_efficient_sdp. Leave all backends enabled; PyTorch picks the best
    # available kernel per operation. Disabling math_sdp prevents the slow
    # non-fused fallback on large sequences.
    if hasattr(torch.backends.cuda, "enable_flash_sdp"):
        torch.backends.cuda.enable_flash_sdp(True)           # DiT attention (head_dim ≤ 256)
        torch.backends.cuda.enable_mem_efficient_sdp(True)   # VAE attention (head_dim=384)
        torch.backends.cuda.enable_math_sdp(False)           # disable slow O(n²) fallback
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


def apply_vram_management(
    model: torch.nn.Module,
    device: str = "cuda",
    gpu_resident: bool = True,
) -> None:
    """Apply DiffSynth block-wise VRAM management to a WanModel with FP8 weights.

    gpu_resident=True (default, RTX 5090 recommended):
      Both FP8 DiTs stay on GPU. No PCIe transfers during inference.
      FP8→BF16 cast happens on-GPU per block (fast). Requires 28GB+ VRAM.
      Eliminates 960 PCIe round-trips at 12 steps (40 blocks × 2 DiTs × 12 steps).

    gpu_resident=False (CPU-offload fallback):
      Each DiTBlock migrates CPU→GPU→CPU per timestep via PCIe (~960 transfers).
      Safe for constrained VRAM; use if VRAM OOM occurs with gpu_resident=True.
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

    if gpu_resident:
        # GPU-resident: blocks stay on GPU in FP8 between timesteps.
        # computation_device=device triggers GPU-side FP8→BF16 cast (no PCIe).
        # After forward: BF16 copy freed, block stays on GPU as FP8.
        vram_config = {
            "offload_dtype": torch.float8_e4m3fn,
            "offload_device": device,       # stay on GPU between uses
            "onload_dtype": torch.float8_e4m3fn,
            "onload_device": device,        # already on GPU, no-op
            "preparing_dtype": torch.float8_e4m3fn,
            "preparing_device": device,     # GPU prepare — no-op
            "computation_dtype": torch.bfloat16,
            "computation_device": device,   # GPU BF16 cast, no PCIe transfer
        }
        print(f"  VRAM mode: GPU-resident FP8 (no PCIe per block, {device})")
    else:
        # CPU-offload: safe for VRAM-constrained environments, slower.
        # Each block: CPU FP8 → GPU BF16 → compute → GPU freed → back on CPU.
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
    del state_dict  # explicitly release dict (model now owns the tensors)

    mode_label = "GPU-resident FP8" if gpu_resident else "CPU-offload FP8→BF16"
    print(f"  Applying VRAM management ({mode_label})...")
    apply_vram_management(model, device=device, gpu_resident=gpu_resident)

    if gpu_resident:
        # One-time 14GB PCIe transfer: move FP8 weights to GPU now.
        # After this, all 40 blocks live on GPU — no further PCIe during inference.
        vram_gb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e9
        print(f"  Moving FP8 DiT to {device} (one-time {vram_gb:.1f}GB transfer)...")
        model = model.to(device)

    model = model.eval()
    return model


def build_pipeline(lora_path_high: str, lora_path_low: str, device: str = "cuda") -> "WanVideoSviPipeline":
    """Build WanVideoSviPipeline with local FP8 DiTs + BF16 T5/VAE.

    SVI_GPU_RESIDENT env var (default=1):
      1 → both FP8 DiTs stay on GPU (fast, requires 32GB VRAM)
      0 → CPU-offload per block (safe, slow — legacy behaviour)
    """
    from diffsynth.pipelines.wan_video_svi import WanVideoSviPipeline
    from diffsynth.core.loader.config import ModelConfig
    from diffsynth.models.wan_video_text_encoder import HuggingfaceTokenizer

    # GPU-resident mode: keeps FP8 DiTs on GPU to avoid PCIe round-trips per block.
    # Disable with SVI_GPU_RESIDENT=0 if VRAM OOM occurs (e.g. pod has <30GB VRAM).
    gpu_resident = os.environ.get("SVI_GPU_RESIDENT", "1") != "0"

    pipe = WanVideoSviPipeline(device=device, torch_dtype=torch.bfloat16)

    # === Load T5 text encoder and VAE FIRST (DiffSynth loads them to GPU initially,
    #     then offloads to CPU per offload_device="cpu"). Must happen BEFORE the
    #     GPU-resident DiTs are placed in VRAM or there is no room left.
    #     GPU budget: T5 loads (~10GB BF16) → offloads to CPU → VAE (~1GB) → offloads.
    #     After both offload: ~0GB GPU used, ready for the two 14GB FP8 DiTs (28GB). ===
    print("Loading T5 text encoder and VAE (offloads to CPU)...")
    t5_config = ModelConfig(path=f"{WAN22_DIR}/models_t5_umt5-xxl-enc-bf16.pth", offload_device="cpu")
    vae_config = ModelConfig(path=f"{WAN22_DIR}/Wan2.1_VAE.pth", offload_device="cpu")
    model_pool = pipe.download_and_load_models([t5_config, vae_config])
    pipe.text_encoder = model_pool.fetch_model("wan_video_text_encoder")
    pipe.vae = model_pool.fetch_model("wan_video_vae")

    # Force PyTorch to return the CUDA allocator pool back to the GPU after T5/VAE
    # offload. Without this, PyTorch's caching allocator holds onto the ~10GB T5
    # VRAM even though the tensors are on CPU — leaving no room for the two 14GB DiTs.
    if gpu_resident:
        import gc as _gc
        _gc.collect()
        torch.cuda.empty_cache()
        free_gb = (torch.cuda.mem_get_info()[0]) / 1e9
        print(f"  VRAM after T5/VAE offload + cache clear: {free_gb:.1f}GB free")

    # === Load high-noise and low-noise DiTs AFTER T5/VAE have offloaded to CPU ===
    # GPU is now free for the two FP8 DiTs (28GB total on RTX 5090 32GB).
    print(f"Loading high-noise DiT (FP8, gpu_resident={gpu_resident})...")
    pipe.dit = load_wan_dit_fp8(
        f"{WAN22_DIR}/high_noise_model_fp8", device=device, gpu_resident=gpu_resident
    )

    print(f"Loading low-noise DiT (FP8, gpu_resident={gpu_resident})...")
    pipe.dit2 = load_wan_dit_fp8(
        f"{WAN22_DIR}/low_noise_model_fp8", device=device, gpu_resident=gpu_resident
    )

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


def run_inference(pipe: "WanVideoSviPipeline", config: dict) -> dict:
    """Execute one SVI inference request against an already-loaded pipeline.

    Called both by main() (standalone subprocess mode) and by vga_svi_server.py
    (persistent server mode). Keeping inference logic here avoids duplication.

    Returns {"status": "ok", "output_path": str} or {"status": "error", "error": str}.
    """
    from PIL import Image

    output_path = Path(config["output_path"])
    prev_segment_path = config.get("prev_segment_path", "")
    prompt = config.get("prompt", "cinematic scene, photorealistic")
    cfg = float(config.get("cfg", 5.5))
    steps = int(config.get("steps", 12))
    camera_motion = config.get("camera_motion", "static")
    motion_vector = config.get("motion_vector", "stationary")
    # TeaCache: DiffSynth attention caching for 2-3x speedup. Threshold 0.1 is safe;
    # higher values skip more (faster but may reduce quality). 0.0 disables.
    tea_cache_thresh = float(config.get("tea_cache_l1_thresh", 0.1))

    if not prev_segment_path or not Path(prev_segment_path).exists():
        return {"status": "error", "error": f"Previous segment not found: {prev_segment_path}"}

    output_path.parent.mkdir(parents=True, exist_ok=True)

    ref_image_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            ref_image_path = f.name
        print(f"Extracting last frame from {Path(prev_segment_path).name}...")
        if not extract_last_frame(prev_segment_path, ref_image_path):
            return {"status": "error", "error": "Failed to extract reference frame"}

        ref_image = Image.open(ref_image_path).convert("RGB").resize((832, 480))

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

        call_kwargs: dict = dict(
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

        print(f"Running SVI inference: cfg={cfg:.2f} steps={steps} → {output_path.name}")
        if tea_cache_thresh > 0.0:
            # Wan2.2-I2V-A14B shares the Wan2.1-I2V-14B-480P architecture.
            # TeaCache uses this profile for its rel_l1 thresholds.
            tea_model_id = "Wan2.1-I2V-14B-480P"
            try:
                video_frames = pipe(
                    tea_cache_l1_thresh=tea_cache_thresh,
                    tea_cache_model_id=tea_model_id,
                    **call_kwargs,
                )
                print(f"  TeaCache enabled (thresh={tea_cache_thresh}, id={tea_model_id}) — ~2-3x speedup")
            except (TypeError, ValueError):
                # Pipeline version doesn't support TeaCache or model_id not recognised
                print("  TeaCache: not supported by this pipeline build, running standard inference")
                video_frames = pipe(**call_kwargs)
        else:
            video_frames = pipe(**call_kwargs)

        from diffsynth.utils.data import save_video
        save_video(video_frames, str(output_path), fps=15, quality=5)
        print(f"SUCCESS: Segment saved to {output_path}")
        return {"status": "ok", "output_path": str(output_path)}

    except Exception as e:
        import traceback
        msg = traceback.format_exc()
        print(f"ERROR: SVI inference failed: {e}\n{msg}")
        return {"status": "error", "error": str(e), "traceback": msg}

    finally:
        if ref_image_path and os.path.exists(ref_image_path):
            try:
                os.unlink(ref_image_path)
            except OSError:
                pass


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

    lora_path_high = config["lora_path_high"]
    lora_path_low = config["lora_path_low"]

    if not Path(config.get("prev_segment_path", "")).exists():
        print(f"ERROR: Previous segment not found: {config.get('prev_segment_path', '')}")
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
