# VGA Model Stack — Complete Download, Installation & Setup Guide

> **Target Environment:** RunPod RTX 4090 · Ubuntu 24 · Python 3.10+ · CUDA 12.x  
> **Purpose:** All base models, LoRAs, and auxiliary assets required by the VGA (Video Generation Automation) pipeline.  
> **Pipeline Architecture:** Python FastAPI pipeline (no ComfyUI) — all models loaded and inferred programmatically.  
> **SVI Environment:** The `svi_wan22` branch (Stable-Video-Infinity) requires a **separate conda env** with PyTorch 2.7.1 + cu128 — different from the main VGA cu124 stack. See Section 3.5.  
> **Last verified:** May 2026 (v7.2)

---

## Table of Contents

1. [Prerequisites & Environment Setup](#1-prerequisites--environment-setup)
2. [Core Python Dependencies](#2-core-python-dependencies)
3. [Base Models](#3-base-models)
   - 3.1 [Qwen2.5-14B-Instruct-unsloth-bnb-4bit](#31-qwen25-14b-instruct-unsloth-bnb-4bit)
   - 3.2 [FLUX.2-klein-4B](#32-flux2-klein-4b)
   - 3.3 [Z-Image-Turbo](#33-z-image-turbo)
   - 3.4 [Wan2.2-I2V-A14B-FP8 (nalexand fork)](#34-wan22-i2v-a14b-fp8-nalexand-fork)
   - 3.5 [SVI Model (vita-video-gen) — Stable Video Infinity](#35-svi-model-vita-video-gen--stable-video-infinity)
   - 3.6 [LatentSync-1.6](#36-latentsync-16)
   - 3.7 [Fun-CosyVoice3-0.5B-2512](#37-fun-cosyvoice3-05b-2512)
   - 3.8 [MusicGen Medium](#38-musicgen-medium)
   - 3.9 [MMAudio](#39-mmaudio)
   - 3.10 [CLIP ViT Large Patch14](#310-clip-vit-large-patch14)
4. [LoRAs](#4-loras)
   - 4.1 [Snapshot LoRAs (Identity & Style)](#41-snapshot-loras-identity--style)
   - 4.2 [Consistance_Edit_Lora](#42-consistance_edit_lora)
   - 4.3 [SVI LoRAs — Correct Filenames & Paths](#43-svi-loras--correct-filenames--paths)
5. [Auxiliary / Identity Assets](#5-auxiliary--identity-assets)
6. [Recommended Directory Layout](#6-recommended-directory-layout)
7. [Environment Variables Reference](#7-environment-variables-reference)
8. [Full Batch Download Script](#8-full-batch-download-script)
9. [Verification Checklist](#9-verification-checklist)

---

## 1. Prerequisites & Environment Setup

### 1.1 System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | RTX 3090 (24 GB VRAM) | RTX 4090 (24 GB VRAM) |
| RAM | 32 GB | 64 GB+ |
| Storage | 200 GB SSD | 500 GB NVMe |
| CUDA | 12.0 | 12.4+ |
| Python | 3.10 | 3.10 |
| OS | Ubuntu 22.04 | Ubuntu 24 |

### 1.2 Conda Environment (recommended)

```bash
conda create -n vga python=3.10 -y
conda activate vga

# Install ffmpeg (required by torchaudio AND MMAudio — must be < 7)
conda install -c conda-forge 'ffmpeg<7' -y

# Also install system-level sox (required by CosyVoice)
sudo apt-get install -y sox libsox-dev
```

### 1.3 HuggingFace Hub CLI & fast transfer

```bash
pip install -U huggingface_hub hf-transfer

# Enable high-performance XET transfer (significantly faster downloads)
export HF_XET_HIGH_PERFORMANCE=1
export HF_HUB_ENABLE_HF_TRANSFER=1

# Authenticate — required for gated models (vita-video-gen/svi-model, LatentSync-1.6, etc.)
huggingface-cli login
```

### 1.4 PyTorch — install BEFORE flash-attn and other compiled packages

> **CRITICAL:** All PyTorch packages (torch, torchvision, torchaudio, xformers) MUST come from the same CUDA index. The VGA bootstrap uses **cu124**. MMAudio requires PyTorch **2.5.1+**.

```bash
# CUDA 12.4 wheel — matches VGA bootstrap (cu124) and satisfies MMAudio >=2.5.1 requirement
pip install "torch>=2.5.1" torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu124

# Verify
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

---

## 2. Core Python Dependencies

Install in the following order to avoid build conflicts.

### 2.1 HuggingFace Ecosystem

```bash
pip install \
    "huggingface-hub>=0.21.0" \
    "hf-transfer>=0.1.6" \
    "transformers>=4.45.0" \
    "diffusers>=0.30.0" \
    "accelerate>=0.34.0"
```

> **Note for Z-Image-Turbo and FLUX.2-klein-4B:** These models require diffusers features (ZImagePipeline, Flux2KleinPipeline) that may not yet be in the stable release. Install from source if the pipeline class is not found:
> ```bash
> pip install git+https://github.com/huggingface/diffusers.git
> ```

### 2.2 LoRA / Adapter Support

```bash
pip install \
    "peft>=0.12.0" \
    "safetensors>=0.4.3"
```

### 2.3 Optimization Libraries

**Flash Attention 2** — requires CUDA 12.0+, Ampere/Ada GPU (RTX 3090/4090/A100), and ninja:

```bash
# Install ninja first (dramatically speeds up compilation)
pip install ninja

# Install flash-attn (builds from source — takes 3–10 min on GPU server)
# Requires GPU compute capability >= 8.0 (Ampere+)
pip install "flash-attn>=2.6.0" --no-build-isolation
```

> **SVI note:** vita-epfl/Stable-Video-Infinity (svi_wan22 branch) was tested with `flash_attn==2.8.0.post2`. Pin this version if you encounter compatibility issues:
> ```bash
> pip install "flash-attn==2.8.0.post2" --no-build-isolation
> ```

**xFormers:**

```bash
pip install xformers --index-url https://download.pytorch.org/whl/cu124
```

### 2.4 Vision / Motion / Audio

```bash
pip install \
    "open-clip-torch>=2.24.0" \
    "Pillow>=10.0.0" \
    "numpy>=1.24.0" \
    "opencv-python>=4.9.0" \
    "pydub>=0.25.0" \
    "ffmpeg-python>=0.2.0"
```

### 2.5 Utilities & FastAPI Runtime

```bash
pip install \
    "psutil>=5.9.0" \
    "requests>=2.31.0" \
    "fastapi>=0.115.0" \
    "uvicorn>=0.30.0" \
    "pydantic>=2.0.0" \
    scipy \
    omegaconf \
    einops \
    mediapipe \
    face-alignment \
    decord \
    soundfile \
    "python-dotenv>=1.0.0"
```

---

## 3. Base Models

All models default to storage under `/workspace/models/` on RunPod. Adjust paths as needed.

---

### 3.1 Qwen2.5-14B-Instruct-unsloth-bnb-4bit

| Field | Value |
|-------|-------|
| **HF Repo** | `unsloth/Qwen2.5-14B-Instruct-unsloth-bnb-4bit` |
| **HF URL** | https://huggingface.co/unsloth/Qwen2.5-14B-Instruct-unsloth-bnb-4bit |
| **Size** | ~8 GB (4-bit BNB quantised) |
| **VRAM** | ~10–12 GB |
| **License** | Apache 2.0 |
| **Architecture** | Qwen2.5 Transformer (14B parameters, bnb 4-bit) |

#### What it is

An Unsloth-packaged 4-bit BitsAndBytes quantised version of Alibaba's Qwen2.5-14B-Instruct. Used in VGA as the orchestrator LLM. Supports 128K context, multilingual, and strong instruction following.

#### Install dependencies

```bash
pip install unsloth bitsandbytes
# OR for latest bleeding edge:
pip install --upgrade --no-cache-dir "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
```

#### Download

```bash
huggingface-cli download unsloth/Qwen2.5-14B-Instruct-unsloth-bnb-4bit \
    --local-dir /workspace/models/qwen
```

#### Python Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "/workspace/models/qwen"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    load_in_4bit=True,      # BNB 4-bit already quantised
)
```

#### Verify

```bash
python -c "
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('/workspace/models/qwen')
print('Vocab size:', tok.vocab_size)
"
```

---

### 3.2 FLUX.2-klein-4B

| Field | Value |
|-------|-------|
| **HF Repo** | `black-forest-labs/FLUX.2-klein-4B` |
| **HF URL** | https://huggingface.co/black-forest-labs/FLUX.2-klein-4B |
| **GitHub** | https://github.com/black-forest-labs/flux2 |
| **Size** | ~23.7 GB (BF16 full weights + diffusers layout) |
| **VRAM** | ~8–10 GB FP16 (without CPU offload); ~13 GB BF16 full load; safely within 16–20 GB with standard inference; full 24 GB headroom on RTX 4090 |
| **License** | Apache 2.0 |
| **Architecture** | 4B-parameter rectified flow transformer |

> **VRAM Decision:** The official Black Forest Labs GitHub states "Klein 4B fits in ~8GB VRAM". At FP16/BF16 on an RTX 4090 (24 GB), FLUX.2-klein-4B runs comfortably within the 16–20 GB VRAM range. **This version is used in VGA** (not the 8 GB quantised variant). Confirmed well within the 16–20 GB target.

#### What it is

A 4-billion-parameter text-to-image and multi-reference image editing model from Black Forest Labs. Runs on consumer GPUs (RTX 3090/4090+). Fully open-source under Apache 2.0. Supports T2I, I2I, and multi-reference generation in a single unified model. Sub-second inference on modern hardware.

#### Repository layout (after download)

```
FLUX.2-klein-4B/
├── flux-2-klein-4b.safetensors   # single-file weights (~7.75 GB)
├── scheduler/
│   └── scheduler_config.json
├── text_encoder/
│   ├── config.json
│   ├── tokenizer.json
│   └── model.safetensors
├── tokenizer/
├── transformer/
│   ├── config.json
│   └── diffusion_pytorch_model.safetensors
├── vae/
│   ├── config.json
│   └── diffusion_pytorch_model.safetensors
└── model_index.json
```

#### Install diffusers from source (for Flux2KleinPipeline)

```bash
pip install git+https://github.com/huggingface/diffusers.git
```

#### Download

```bash
# Full BF16 (~23.7 GB)
huggingface-cli download black-forest-labs/FLUX.2-klein-4B \
    --local-dir /workspace/models/flux2
```

#### Environment variable (official GitHub inference script)

```bash
export KLEIN_4B_MODEL_PATH="/workspace/models/flux2"
```

#### Python Usage

```python
import torch
from diffusers import Flux2KleinPipeline

pipe = Flux2KleinPipeline.from_pretrained(
    "/workspace/models/flux2",
    torch_dtype=torch.bfloat16
)
pipe.enable_model_cpu_offload()  # save VRAM — recommended for VGA pipeline

image = pipe(
    prompt="A cinematic still of a hero standing on a mountaintop at sunrise",
    height=1024,
    width=1024,
    guidance_scale=1.0,
    num_inference_steps=4,
    generator=torch.Generator(device="cuda").manual_seed(42)
).images[0]
image.save("flux_klein_output.png")
```

---

### 3.3 Z-Image-Turbo

| Field | Value |
|-------|-------|
| **HF Repo** | `Tongyi-MAI/Z-Image-Turbo` |
| **HF URL** | https://huggingface.co/Tongyi-MAI/Z-Image-Turbo |
| **GitHub** | https://github.com/Tongyi-MAI/Z-Image |
| **Size** | ~16 GB |
| **VRAM** | ~16 GB (BF16); ~6 GB (FP8 mode) |
| **License** | Apache 2.0 |
| **Architecture** | Scalable Single-Stream DiT (S3-DiT) |

#### What it is

A distilled, high-speed image generation model from Alibaba's Tongyi-MAI team. Achieves sub-second inference latency on H800 GPUs. Supports photorealistic generation and bilingual (Chinese/English) text rendering. Uses only 8 NFEs (Number of Function Evaluations).

#### Install diffusers from source (required for ZImagePipeline)

```bash
pip install git+https://github.com/huggingface/diffusers.git
```

#### Download

```bash
# Fast download with XET transfer
export HF_XET_HIGH_PERFORMANCE=1
huggingface-cli download Tongyi-MAI/Z-Image-Turbo \
    --local-dir /workspace/models/zimage
```

#### Python Usage

```python
import torch
from diffusers import ZImagePipeline

pipe = ZImagePipeline.from_pretrained(
    "/workspace/models/zimage",
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=False,
)
pipe.to("cuda")

# Optional: enable Flash Attention backend
# pipe.transformer.set_attention_backend("flash")

# Optional: compile for faster repeated inference
# pipe.transformer.compile()

image = pipe(
    prompt="A hero standing triumphant after overcoming adversity, cinematic golden hour",
    height=1024,
    width=1024,
    num_inference_steps=9,   # 8 actual DiT forwards
    guidance_scale=0.0,      # Turbo models: 0.0 guidance
    generator=torch.Generator("cuda").manual_seed(42),
).images[0]
image.save("z_image_turbo_output.png")
```

---

### 3.4 Wan2.2-I2V-A14B-FP8 (nalexand fork)

| Field | Value |
|-------|-------|
| **HF Repo (FP8 fork)** | `nalexand/Wan2.2-I2V-A14B-FP8` |
| **HF URL** | https://huggingface.co/nalexand/Wan2.2-I2V-A14B-FP8 |
| **HF Repo (official)** | `Wan-AI/Wan2.2-I2V-A14B` |
| **HF URL (official)** | https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B |
| **GitHub (nalexand)** | https://github.com/nalexand/Wan2.2 |
| **GitHub (official)** | https://github.com/Wan-Video/Wan2.2 |
| **Diffusers variant** | `Wan-AI/Wan2.2-I2V-A14B-Diffusers` |
| **Size** | ~30 GB (FP8); ~60 GB (official FP16) |
| **VRAM** | ~8 GB (FP8 optimised); ~24 GB (FP16) |
| **License** | Apache 2.0 |
| **Architecture** | Mixture-of-Experts (MoE) I2V transformer, 14B active params (~27B total) |

#### What it is

Wan2.2 I2V-A14B is a state-of-the-art Image-to-Video model using a two-expert MoE architecture (high-noise + low-noise experts). The `nalexand` fork provides an 8 GB-optimised FP8 version with a custom `generate_local.py` script for single-card GPU inference and long video generation loops. This is the backbone for SVI (Stable Video Infinity) LoRA adaptation in VGA.

#### Download (FP8 — recommended for VGA/RTX 4090)

```bash
huggingface-cli download nalexand/Wan2.2-I2V-A14B-FP8 \
    --local-dir /workspace/models/wan22
```

> **Note:** The nalexand HF card shows download directory as `./Wan2.2-T2V-A14B` in its README example — this is a typo in their card. The correct target is your chosen local directory (e.g. `/workspace/models/wan22`). The model is always `I2V-A14B-FP8`.

#### Download (official FP16 — optional, for reference or manual FP8 conversion)

```bash
pip install "huggingface_hub[cli]"
huggingface-cli download Wan-AI/Wan2.2-I2V-A14B \
    --local-dir /workspace/models/wan22_fp16
```

#### Install inference requirements (official repo)

```bash
git clone https://github.com/Wan-Video/Wan2.2.git /workspace/Wan2.2
cd /workspace/Wan2.2
# Ensure torch >= 2.5.1 is already installed before this
pip install -r requirements.txt
```

#### Inference (nalexand FP8 — VGA primary usage)

```bash
# Single card inference with nalexand fork
cd /workspace/Wan2.2   # nalexand fork (cloned from github.com/nalexand/Wan2.2)
python generate_local.py \
    --task i2v-A14B \
    --size "1280*720" \
    --image ./input_frame.png \
    --ckpt_dir /workspace/models/wan22 \
    --prompt "A person overcoming adversity, rising from darkness into golden light, cinematic motion"
```

#### Inference (official, single GPU with offload)

```bash
cd /workspace/Wan2.2_official   # Wan-Video/Wan2.2
python generate.py \
    --task i2v-A14B \
    --size 1280*720 \
    --ckpt_dir /workspace/models/wan22_fp16 \
    --offload_model True \
    --convert_model_dtype \
    --image examples/i2v_input.JPG \
    --prompt "Your cinematic prompt here"
```

#### SVI LoRAs placement

See Section 4.3 for **correct SVI LoRA filenames**. The SVI inference script (vita-epfl/Stable-Video-Infinity svi_wan22 branch) expects:

```
/workspace/models/svi/version-2.0/
├── SVI_Wan2.2-I2V-A14B_high_noise_lora_v2.0_pro.safetensors
└── SVI_Wan2.2-I2V-A14B_low_noise_lora_v2.0_pro.safetensors
```

---

### 3.5 SVI Model (vita-video-gen) — Stable Video Infinity

| Field | Value |
|-------|-------|
| **HF Repo** | `vita-video-gen/svi-model` |
| **HF URL** | https://huggingface.co/vita-video-gen/svi-model |
| **GitHub** | https://github.com/vita-epfl/Stable-Video-Infinity |
| **GitHub (Wan2.2 branch)** | https://github.com/vita-epfl/Stable-Video-Infinity/tree/svi_wan22 |
| **Paper** | arXiv:2510.09212 (ICLR 2026 Oral) |
| **License** | MIT (model weights + code) |
| **Architecture** | LoRA adapters on Wan2.2-I2V-A14B backbone (Error Recycling Fine-Tuning) |

#### What it is

Stable Video Infinity (SVI) generates arbitrarily long videos with high temporal consistency and controllable scene transitions via per-segment autoregressive inference with error-recycling. Only LoRA adapters are tuned (lightweight). Uses Wan2.2-I2V-A14B as the backbone. The VGA pipeline uses the **v2.0 Pro** variant with dual-LoRA (high_noise + low_noise).

> **CRITICAL NOTE — Wan2.2 branch:** The main branch of vita-epfl/Stable-Video-Infinity targets Wan2.1. The **`svi_wan22` branch** is required for Wan2.2 support. Always checkout this branch for VGA.

#### Clone the inference repo (svi_wan22 branch — REQUIRED for Wan2.2)

```bash
# Option A: Clone directly to svi_wan22 branch (recommended)
git clone https://github.com/vita-epfl/Stable-Video-Infinity.git \
    -b svi_wan22 /workspace/Stable-Video-Infinity
cd /workspace/Stable-Video-Infinity

# Option B: Clone + checkout
# git clone https://github.com/vita-epfl/Stable-Video-Infinity.git /workspace/Stable-Video-Infinity
# cd /workspace/Stable-Video-Infinity && git checkout svi_wan22

# Install SVI environment — Official svi_wan22 branch requirements
# PyTorch 2.7.1 + cu128 is the officially tested stack for svi_wan22 branch
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
    --index-url https://download.pytorch.org/whl/cu128

pip install -e .
pip install "flash_attn==2.8.0.post2" --no-build-isolation
conda install -c conda-forge ffmpeg librosa libiconv -y
```

> **Note:** The `svi_wan22` branch is built on DiffSynth 2.0, requiring its own environment. The VGA bootstrap uses cu124/PyTorch 2.5.1+ for the main pipeline; the SVI subshell uses cu128/PyTorch 2.7.1 for inference calls. Run SVI inference in a separate conda env (`svi_wan22`) or via subprocess to avoid CUDA conflicts.

#### Download SVI LoRA weights (v2.0 Pro — Wan2.2)

```bash
# ── CRITICAL: LoRA files live in version-2.0/ subfolder of the repo ──
# High-noise LoRA
huggingface-cli download vita-video-gen/svi-model \
    --include "version-2.0/SVI_Wan2.2-I2V-A14B_high_noise_lora_v2.0_pro.safetensors" \
    --local-dir /workspace/models/svi

# Low-noise LoRA
huggingface-cli download vita-video-gen/svi-model \
    --include "version-2.0/SVI_Wan2.2-I2V-A14B_low_noise_lora_v2.0_pro.safetensors" \
    --local-dir /workspace/models/svi

# OR: download entire svi-model repo (includes all versions + auxiliary assets)
huggingface-cli download vita-video-gen/svi-model \
    --local-dir /workspace/models/svi
```

#### SVI Inference (svi_wan22 branch)

```bash
CUDA_VISIBLE_DEVICES=0 python inference_svi_2.0.py \
    --output_root /workspace/videos \
    --height 480 \
    --width 832 \
    --lora_path_high /workspace/models/svi/version-2.0/SVI_Wan2.2-I2V-A14B_high_noise_lora_v2.0_pro.safetensors \
    --lora_path_low  /workspace/models/svi/version-2.0/SVI_Wan2.2-I2V-A14B_low_noise_lora_v2.0_pro.safetensors \
    --fps 15 \
    --ref_image_path ./data/toy_test/demo2/frame.jpg \
    --prompt_path ./data/toy_test/demo2/prompt.txt \
    --num_clips 10
```

> **WARNING:** SVI LoRA CANNOT be used with the original Wan2.2 workflow directly. The padding settings are different. Always use the SVI inference script from the `svi_wan22` branch.

---

### 3.6 LatentSync-1.6

| Field | Value |
|-------|-------|
| **HF Repo** | `ByteDance/LatentSync-1.6` |
| **HF URL** | https://huggingface.co/ByteDance/LatentSync-1.6 |
| **GitHub** | https://github.com/bytedance/LatentSync |
| **Released** | 2025-06-11 |
| **Size** | ~7.5 GB total (UNet ~5 GB + SyncNet ~1.6 GB + auxiliary) |
| **VRAM** | 6.5 GB minimum (inference); 20 GB (stage2 training with optimisations) |
| **License** | Apache 2.0 |
| **Architecture** | Audio-conditioned latent diffusion UNet (Stable Diffusion based) |

#### What it is

An end-to-end lip-sync framework trained on 512×512 resolution videos (v1.6 upgrade over v1.5 to resolve blurriness in teeth/lips). Uses Whisper for audio embeddings integrated into UNet via cross-attention. v1.5 and v1.6 share identical code — only the checkpoint and config `resolution` value differ.

#### Clone the inference repo

```bash
git clone https://github.com/bytedance/LatentSync.git /workspace/LatentSync
cd /workspace/LatentSync
pip install -r requirements.txt
```

#### Checkpoint directory structure expected

```
/workspace/LatentSync/checkpoints/
├── latentsync_unet.pt          # ~5 GB   — main UNet (v1.6)
├── stable_syncnet.pt           # ~1.6 GB — SyncNet discriminator
├── whisper/
│   └── tiny.pt                 # ~75 MB  — Whisper encoder
├── vae/
│   ├── config.json
│   └── diffusion_pytorch_model.safetensors
└── auxiliary/
    ├── 2DFAN4-cd938726ad.zip
    ├── i3d_torchscript.pt
    ├── koniq_pretrained.pkl
    ├── s3fd-619a316812.pth
    ├── sfd_face.pth
    ├── syncnet_v2.model
    ├── vgg16-397923af.pth
    └── vit_g_hybrid_pt_1200e_ssv2_ft.pth
```

#### Download — all files recommended (includes auxiliary assets)

```bash
cd /workspace/LatentSync

# All files — recommended (includes auxiliary, VAE, whisper)
huggingface-cli download ByteDance/LatentSync-1.6 \
    --local-dir ./checkpoints

# OR individual key files:
huggingface-cli download ByteDance/LatentSync-1.6 latentsync_unet.pt --local-dir ./checkpoints
huggingface-cli download ByteDance/LatentSync-1.6 stable_syncnet.pt  --local-dir ./checkpoints
huggingface-cli download ByteDance/LatentSync-1.6 whisper/tiny.pt    --local-dir ./checkpoints
```

#### Or use the official setup script

```bash
cd /workspace/LatentSync
bash setup_env.sh
```

#### Inference

```bash
# Via script
cd /workspace/LatentSync
bash inference.sh

# Via Gradio app
python gradio_app.py
```

#### Config note (v1.5 → v1.6 switch)

To switch between versions, load the corresponding checkpoint and update `resolution` in the UNet config file at `configs/unet/`. No other code changes needed.

---

### 3.7 Fun-CosyVoice3-0.5B-2512

| Field | Value |
|-------|-------|
| **HF Repo** | `FunAudioLLM/Fun-CosyVoice3-0.5B-2512` |
| **HF URL** | https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512 |
| **GitHub** | https://github.com/FunAudioLLM/CosyVoice |
| **Released** | December 2025 |
| **Size** | ~9.75 GB |
| **VRAM** | ~4–6 GB for inference |
| **License** | Apache 2.0 |
| **Architecture** | 0.5B LLM-based TTS with Matcha-TTS acoustic backend |

#### What it is

Fun-CosyVoice 3.0 — an advanced multilingual TTS system from Alibaba's FunAudioLLM team. Supports 9 languages + 18+ Chinese dialects, zero-shot voice cloning, and bi-streaming (text-in / audio-out) with ~150ms latency. Latest release in the CosyVoice family (supersedes CosyVoice2-0.5B). Confirmed released Dec 2025.

#### Clone the repo (includes Matcha-TTS submodule)

```bash
git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git /workspace/CosyVoice
cd /workspace/CosyVoice

# If submodule clone fails due to network issues, retry:
git submodule update --init --recursive
```

#### Install dependencies

```bash
cd /workspace/CosyVoice

# Ubuntu system dependencies (sox required)
sudo apt-get install -y sox libsox-dev

# Python dependencies
pip install -r requirements.txt
```

#### Download models

```python
from huggingface_hub import snapshot_download
import os

os.chdir("/workspace/CosyVoice")

# CosyVoice 3 main model
snapshot_download(
    'FunAudioLLM/Fun-CosyVoice3-0.5B-2512',
    local_dir='pretrained_models/Fun-CosyVoice3-0.5B'
)

# Text normalisation resource (optional but recommended for production)
snapshot_download(
    'FunAudioLLM/CosyVoice-ttsfrd',
    local_dir='pretrained_models/CosyVoice-ttsfrd'
)
```

#### Optional: install ttsfrd for better text normalisation

```bash
cd /workspace/CosyVoice/pretrained_models/CosyVoice-ttsfrd/
unzip resource.zip -d .
pip install ttsfrd_dependency-0.1-py3-none-any.whl
pip install ttsfrd-0.4.2-cp310-cp310-linux_x86_64.whl
```

> If ttsfrd is not installed, the system falls back to WeTextProcessing automatically.

#### Python Usage (inference)

```python
import sys
sys.path.append('/workspace/CosyVoice/third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import AutoModel
import torchaudio

cosyvoice = AutoModel(model_dir='/workspace/CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B')

# Zero-shot voice cloning (English)
for i, j in enumerate(cosyvoice.inference_zero_shot(
    'Inspire audiences by telling stories of people who overcame adversity.',
    'You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。',
    '/workspace/CosyVoice/asset/zero_shot_prompt.wav',
    stream=False
)):
    torchaudio.save(f'zero_shot_{i}.wav', j['tts_speech'], cosyvoice.sample_rate)
```

#### Model file layout

```
pretrained_models/Fun-CosyVoice3-0.5B/
├── campplus.onnx
├── config.json
├── configuration.json
├── cosyvoice3.yaml
├── flow.decoder.estimator.fp32.onnx   # 1.33 GB
├── flow.pt
├── speech_tokenizer_v3.batch.onnx
└── asset/
```

---

### 3.8 MusicGen Medium

| Field | Value |
|-------|-------|
| **HF Repo** | `facebook/musicgen-medium` |
| **HF URL** | https://huggingface.co/facebook/musicgen-medium |
| **GitHub** | https://github.com/facebookresearch/audiocraft |
| **Size** | ~3.1 GB |
| **VRAM** | ~16 GB (required for medium 1.5B model) |
| **License** | CC-BY-NC 4.0 |
| **Architecture** | 1.5B autoregressive Transformer, EnCodec 32kHz tokenizer |

#### What it is

Meta's MusicGen Medium — a single-stage autoregressive Transformer for controllable text-to-music generation. Trained on 20K hours of licensed music. Part of the AudioCraft library.

#### Install AudioCraft

```bash
# PyTorch >= 2.5.1 must already be installed
pip install -U audiocraft

# OR bleeding edge from source:
pip install -U git+https://github.com/facebookresearch/audiocraft.git

# OR clone + local install (required for training):
git clone https://github.com/facebookresearch/audiocraft.git /workspace/audiocraft
cd /workspace/audiocraft
pip install -e .
```

#### Download (automatic on first use via AudioCraft)

```python
from audiocraft.models import MusicGen
# Model downloads automatically to HF cache on first call:
model = MusicGen.get_pretrained('facebook/musicgen-medium')
```

#### Download (explicit, to custom path)

```bash
huggingface-cli download facebook/musicgen-medium \
    --local-dir /workspace/models/musicgen

# Override AudioCraft's default cache location:
export AUDIOCRAFT_CACHE_DIR=/workspace/models
```

#### Python Usage

```python
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write

model = MusicGen.get_pretrained('facebook/musicgen-medium')
model.set_generation_params(duration=8)  # generate 8 seconds

descriptions = ['inspirational orchestral music, triumph, overcoming adversity']
wav = model.generate(descriptions)

for idx, one_wav in enumerate(wav):
    audio_write(
        f'musicgen_output_{idx}',
        one_wav.cpu(),
        model.sample_rate,
        strategy="loudness"
    )
```

#### Via 🤗 Transformers pipeline (alternative)

```python
from transformers import pipeline
import scipy

synthesiser = pipeline("text-to-audio", "facebook/musicgen-medium")
music = synthesiser(
    "inspirational cinematic orchestral score",
    forward_params={"do_sample": True}
)
scipy.io.wavfile.write(
    "musicgen_out.wav",
    rate=music["sampling_rate"],
    data=music["audio"]
)
```

---

### 3.9 MMAudio

| Field | Value |
|-------|-------|
| **HF Repo** | `hkchengrex/MMAudio` |
| **HF URL** | https://huggingface.co/hkchengrex/MMAudio |
| **HF Tree** | https://huggingface.co/hkchengrex/MMAudio/tree/main |
| **GitHub** | https://github.com/hkchengrex/MMAudio |
| **Size** | ~2.5 GB total (weights auto-downloaded) |
| **VRAM** | ~6 GB (16-bit mode) |
| **License** | CC BY-NC 4.0 (model weights) |
| **Architecture** | Multimodal joint training — video+text → audio synthesis |

#### What it is

CVPR 2025 — generates synchronized audio from video and/or text prompts. Uses a synchronisation module to align audio with video frames. Default model: `large_44k_v2`. Requires **PyTorch 2.5.1+** and **ffmpeg < 7**.

#### Expected directory layout (full)

```
/workspace/MMAudio/
├── ext_weights/
│   ├── best_netG.pt                # 44.1kHz vocoder (auto-downloaded)
│   ├── synchformer_state_dict.pth  # Synchformer visual extractor
│   ├── v1-16.pth                   # 16kHz VAE
│   └── v1-44.pth                   # 44.1kHz VAE
└── weights/
    ├── mmaudio_small_16k.pth       # 601 MB
    ├── mmaudio_small_44k.pth       # 601 MB
    ├── mmaudio_medium_44k.pth      # 2.49 GB
    ├── mmaudio_large_44k.pth       # 4.12 GB
    └── mmaudio_large_44k_v2.pth    # 4.12 GB  ← recommended (generalises better)
```

#### Installation

```bash
# 1. Prerequisites: PyTorch 2.5.1+ and ffmpeg < 7
#    (ffmpeg<7 must be installed BEFORE this step — see section 1.2)

# 2. Clone
git clone https://github.com/hkchengrex/MMAudio.git /workspace/MMAudio
cd /workspace/MMAudio

# 3. Install (upgrade pip first to avoid setup.py not found error)
pip install --upgrade pip
pip install -e .
```

#### Download weights (automatic on first run)

Models download automatically when you run the demo script:

```bash
cd /workspace/MMAudio
python demo.py --duration=8 --video=<path_to_video> --prompt "your prompt"
```

#### Manual download (to a specific directory)

```bash
# Download all weights from HuggingFace
huggingface-cli download hkchengrex/MMAudio \
    --local-dir /workspace/MMAudio
# This places weights/ and ext_weights/ as they appear in the repo tree.
```

MD5 checksums for verification are provided in `mmaudio/utils/download_utils.py`.

#### Python Usage (from VGA pipeline)

```bash
cd /workspace/MMAudio

# Video-to-audio (recommended)
python demo.py \
    --duration=8 \
    --video=./input_video.mp4 \
    --prompt "cinematic score, triumphant brass, motivational"

# Text-to-audio (omit --video)
python demo.py \
    --duration=8 \
    --prompt "inspirational orchestral swell, overcoming adversity"
```

Output saved to `./output/` as `.flac` (audio) and `.mp4` (video+audio).

---

### 3.10 CLIP ViT Large Patch14

| Field | Value |
|-------|-------|
| **HF Repo** | `openai/clip-vit-large-patch14` |
| **HF URL** | https://huggingface.co/openai/clip-vit-large-patch14 |
| **Size** | ~890 MB |
| **VRAM** | ~2 GB |
| **License** | MIT |
| **Architecture** | ViT-L/14 image encoder + masked self-attention text encoder |

#### What it is

OpenAI's CLIP model (ViT-L/14 variant). Used in VGA for identity verification, visual feature extraction, and image-text similarity scoring. Required by MMAudio (Synchformer uses CLIP features). Also auto-downloaded by dependent libraries (diffusers, open-clip-torch).

#### Install

```bash
pip install "open-clip-torch>=2.24.0"
# OR via transformers:
pip install "transformers>=4.45.0"
```

#### Download (explicit, to custom path)

```bash
huggingface-cli download openai/clip-vit-large-patch14 \
    --local-dir /workspace/auxiliary/clip
```

#### Python Usage

```python
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

model = CLIPModel.from_pretrained("/workspace/auxiliary/clip").to("cuda")
processor = CLIPProcessor.from_pretrained("/workspace/auxiliary/clip")

image = Image.open("./reference.png")
inputs = processor(
    text=["a photo of a hero", "a photo of a villain"],
    images=image,
    return_tensors="pt",
    padding=True
).to("cuda")

outputs = model(**inputs)
probs = outputs.logits_per_image.softmax(dim=1)
print("Similarity scores:", probs)
```

---

## 4. LoRAs

### 4.1 Snapshot LoRAs (Identity & Style)

These LoRAs are loaded dynamically at runtime from environment variables. They are user-specific and must be provided externally.

| LoRA | Environment Variable | Description |
|------|---------------------|-------------|
| Identity LoRA | `LORA_IDENTITY_REPO` | Character-specific identity preservation LoRA |
| Style LoRA | `LORA_STYLE_REPO` | Visual style/aesthetic LoRA |

#### Setting up

```bash
# In your .env or shell profile:
export LORA_IDENTITY_REPO="your-hf-username/your-identity-lora"
export LORA_STYLE_REPO="your-hf-username/your-style-lora"
```

#### Download at runtime

```python
import os
from huggingface_hub import snapshot_download

identity_repo = os.environ["LORA_IDENTITY_REPO"]
style_repo = os.environ["LORA_STYLE_REPO"]

snapshot_download(identity_repo, local_dir="/workspace/loras/identity/character_main")
snapshot_download(style_repo,    local_dir="/workspace/loras/style/cinematic")
```

---

### 4.2 Consistance_Edit_Lora

| Field | Value |
|-------|-------|
| **HF Repo** | `lrzjason/Consistance_Edit_Lora` |
| **HF URL** | https://huggingface.co/lrzjason/Consistance_Edit_Lora |
| **Format** | `.safetensors` |
| **License** | See repo card |

#### What it is

A LoRA adapter that improves structural consistency during image editing workflows (designed for Qwen-VL edit pipelines). Prevents random movement of image structure when using Qwen-VL as the image encoder. Follows a Kontext-style workflow: reference image is set but not encoded. Trained on FLUX.2-klein 9B-based pipeline but applicable to 4B as well.

#### Download

```bash
huggingface-cli download lrzjason/Consistance_Edit_Lora \
    --local-dir /workspace/loras/consistency
```

#### Usage guidance

- **Higher LoRA strength** → more structural consistency but may reduce editability
- **Lower LoRA strength** → if the base model produces results but adding the LoRA blocks them
- Suggested strength: `0.2–1.5` (experiment per image)
- Combine with identity/style LoRAs for best results in high-fidelity editing

#### Load with diffusers

```python
pipe.load_lora_weights(
    "/workspace/loras/consistency",
    adapter_name="consistance_edit"
)
pipe.set_adapters(["consistance_edit"], adapter_weights=[0.8])
```

---

### 4.3 SVI LoRAs — Correct Filenames & Paths

> **⚠️ CRITICAL — FILENAME CORRECTION:** The correct filenames for the SVI v2.0 Pro LoRAs (as verified against `vita-video-gen/svi-model` on HuggingFace and `vita-epfl/Stable-Video-Infinity` on GitHub) include the full model identifier in the name. Many references use shortened names — these will cause `FileNotFoundError` at runtime.

| File | Source Repo | Path inside Repo | Description |
|------|------------|-----------------|-------------|
| `SVI_Wan2.2-I2V-A14B_high_noise_lora_v2.0_pro.safetensors` | `vita-video-gen/svi-model` | `version-2.0/` subfolder | High-noise expert LoRA (early denoising stages) |
| `SVI_Wan2.2-I2V-A14B_low_noise_lora_v2.0_pro.safetensors` | `vita-video-gen/svi-model` | `version-2.0/` subfolder | Low-noise expert LoRA (late denoising / detail refinement) |

#### Verified HuggingFace paths

```
https://huggingface.co/vita-video-gen/svi-model/blob/main/version-2.0/SVI_Wan2.2-I2V-A14B_high_noise_lora_v2.0_pro.safetensors
https://huggingface.co/vita-video-gen/svi-model/blob/main/version-2.0/SVI_Wan2.2-I2V-A14B_low_noise_lora_v2.0_pro.safetensors
```

#### Download (specific files — recommended to avoid downloading the entire repo)

```bash
# High-noise LoRA
huggingface-cli download vita-video-gen/svi-model \
    --include "version-2.0/SVI_Wan2.2-I2V-A14B_high_noise_lora_v2.0_pro.safetensors" \
    --local-dir /workspace/models/svi

# Low-noise LoRA
huggingface-cli download vita-video-gen/svi-model \
    --include "version-2.0/SVI_Wan2.2-I2V-A14B_low_noise_lora_v2.0_pro.safetensors" \
    --local-dir /workspace/models/svi

# OR: full repo (includes all SVI versions — Wan2.1 and Wan2.2, base + pro)
huggingface-cli download vita-video-gen/svi-model \
    --local-dir /workspace/models/svi
```

#### Expected layout after download

```
/workspace/models/svi/
└── version-2.0/
    ├── SVI_Wan2.2-I2V-A14B_high_noise_lora_v2.0_pro.safetensors
    ├── SVI_Wan2.2-I2V-A14B_low_noise_lora_v2.0_pro.safetensors
    ├── SVI_Wan2.2-I2V-A14B_high_noise_lora_v2.0.safetensors      ← non-pro variant
    └── SVI_Wan2.2-I2V-A14B_low_noise_lora_v2.0.safetensors       ← non-pro variant
```

#### Convenience symlinks for loras/ directory

```bash
mkdir -p /workspace/loras/svi
ln -sf /workspace/models/svi/version-2.0/SVI_Wan2.2-I2V-A14B_high_noise_lora_v2.0_pro.safetensors \
       /workspace/loras/svi/SVI_Wan2.2-I2V-A14B_high_noise_lora_v2.0_pro.safetensors
ln -sf /workspace/models/svi/version-2.0/SVI_Wan2.2-I2V-A14B_low_noise_lora_v2.0_pro.safetensors \
       /workspace/loras/svi/SVI_Wan2.2-I2V-A14B_low_noise_lora_v2.0_pro.safetensors
```

---

## 5. Auxiliary / Identity Assets

These files are companion assets for the VGA identity system. Referenced by the `IdentityTracker` and `DriftController` components.

| File | Purpose |
|------|---------|
| `embedding.npy` | Precomputed CLIP/identity embedding vector for the target character |
| `reference.png` | Reference image used for identity anchoring |
| `metadata.json` | Character and session metadata |
| `identity_registry.json` | Registry mapping character IDs to their asset paths |

#### Typical location

```
/workspace/assets/identity/
├── embedding.npy
├── reference.png
├── metadata.json
└── identity_registry.json
```

Per-character layout:

```
/workspace/identity/
├── char_A/
│   ├── embedding.npy
│   ├── reference.png
│   └── metadata.json
└── char_B/
    ├── embedding.npy
    ├── reference.png
    └── metadata.json
```

#### Generate embedding.npy from a reference image

```python
import numpy as np
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

model = CLIPModel.from_pretrained("/workspace/auxiliary/clip").to("cuda")
processor = CLIPProcessor.from_pretrained("/workspace/auxiliary/clip")

image = Image.open("/workspace/assets/identity/reference.png")
inputs = processor(images=image, return_tensors="pt").to("cuda")

with torch.no_grad():
    embedding = model.get_image_features(**inputs)
    embedding = embedding / embedding.norm(dim=-1, keepdim=True)  # L2 normalise

np.save("/workspace/assets/identity/embedding.npy", embedding.cpu().numpy())
print("Saved embedding shape:", embedding.shape)
```

---

## 6. Recommended Directory Layout

```
/workspace/vga/                               ← project root
├── models/
│   ├── qwen/                                 ← unsloth/Qwen2.5-14B-Instruct-unsloth-bnb-4bit
│   ├── flux2/                                ← black-forest-labs/FLUX.2-klein-4B
│   ├── zimage/                               ← Tongyi-MAI/Z-Image-Turbo
│   ├── wan22/                                ← nalexand/Wan2.2-I2V-A14B-FP8 (VGA primary)
│   ├── wan22_fp16/                           ← Wan-AI/Wan2.2-I2V-A14B (optional FP16)
│   ├── svi/                                  ← vita-video-gen/svi-model
│   │   └── version-2.0/
│   │       ├── SVI_Wan2.2-I2V-A14B_high_noise_lora_v2.0_pro.safetensors
│   │       └── SVI_Wan2.2-I2V-A14B_low_noise_lora_v2.0_pro.safetensors
│   ├── latentsync/                           ← ByteDance/LatentSync-1.6 (snapshot)
│   ├── cosyvoice/                            ← (not used directly — CosyVoice uses its own dir)
│   ├── musicgen/                             ← facebook/musicgen-medium
│   └── mmaudio/                              ← hkchengrex/MMAudio (snapshot)
├── loras/
│   ├── identity/
│   │   ├── character_main/                   ← LORA_IDENTITY_REPO contents
│   │   ├── char_A/
│   │   └── char_B/
│   ├── style/
│   │   └── cinematic/                        ← LORA_STYLE_REPO contents
│   ├── consistency/                          ← lrzjason/Consistance_Edit_Lora
│   └── svi/                                  ← symlinks to models/svi/version-2.0/
│       ├── SVI_Wan2.2-I2V-A14B_high_noise_lora_v2.0_pro.safetensors
│       └── SVI_Wan2.2-I2V-A14B_low_noise_lora_v2.0_pro.safetensors
├── auxiliary/
│   └── clip/                                 ← openai/clip-vit-large-patch14
├── LatentSync/                               ← cloned github.com/bytedance/LatentSync
│   └── checkpoints/
│       ├── latentsync_unet.pt
│       ├── stable_syncnet.pt
│       ├── whisper/tiny.pt
│       ├── vae/
│       └── auxiliary/
├── CosyVoice/                                ← cloned github.com/FunAudioLLM/CosyVoice
│   └── pretrained_models/
│       ├── Fun-CosyVoice3-0.5B/
│       └── CosyVoice-ttsfrd/
├── MMAudio/                                  ← cloned github.com/hkchengrex/MMAudio
│   ├── ext_weights/
│   │   ├── best_netG.pt
│   │   ├── synchformer_state_dict.pth
│   │   ├── v1-16.pth
│   │   └── v1-44.pth
│   └── weights/
│       └── mmaudio_large_44k_v2.pth
├── Stable-Video-Infinity/                    ← cloned (svi_wan22 branch)
│   └── [inference scripts]
├── Wan2.2/                                   ← cloned github.com/nalexand/Wan2.2 or Wan-Video/Wan2.2
├── identity/
│   ├── char_A/
│   └── char_B/
├── assets/
│   └── identity/
│       ├── embedding.npy
│       ├── reference.png
│       ├── metadata.json
│       └── identity_registry.json
├── hrg/
│   ├── checkpoints/
│   └── approvals/
├── state/
│   └── run_manifest.json
├── logs/
└── .env_vga                                  ← auto-generated by bootstrap_pipeline.py
```

---

## 7. Environment Variables Reference

The `.env_vga` file is auto-generated by `bootstrap_pipeline.py`. For reference/manual setup:

```bash
# === HuggingFace ===
HF_HOME=/workspace/cache/huggingface
HUGGINGFACE_HUB_CACHE=/workspace/cache/huggingface
HF_HUB_ENABLE_HF_TRANSFER=1
HF_XET_HIGH_PERFORMANCE=1
HF_HUB_DOWNLOAD_TIMEOUT=300
HF_HUB_HTTP_TOTAL_TIMEOUT=600
HF_HUB_MAX_RETRIES=5
HUGGING_FACE_HUB_TOKEN=hf_your_token_here   # required for gated repos

# === Model Paths ===
VGA_MODELS_DIR=/workspace/models
VGA_LORAS_DIR=/workspace/loras
VGA_ASSETS_DIR=/workspace/assets

# === LoRA Repos (dynamic, user-defined) ===
LORA_IDENTITY_REPO=your-username/your-identity-lora
LORA_STYLE_REPO=your-username/your-style-lora

# === SVI LoRA Config (correct filenames — verified against vita-video-gen/svi-model) ===
SVI_LORA_REPO=vita-video-gen/svi-model
SVI_HIGH_NOISE_FILE=version-2.0/SVI_Wan2.2-I2V-A14B_high_noise_lora_v2.0_pro.safetensors
SVI_LOW_NOISE_FILE=version-2.0/SVI_Wan2.2-I2V-A14B_low_noise_lora_v2.0_pro.safetensors

# === AudioCraft ===
AUDIOCRAFT_CACHE_DIR=/workspace/models

# === PyTorch ===
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CUDA_VISIBLE_DEVICES=0

# === Framework ===
XFORMERS_ENABLE=1
FLASH_ATTN_ENABLE=1

# === VGA Quality Controls ===
QUALITY_CLIP_THRESHOLD=0.93
QUALITY_MAX_RETRIES=3
OPTICAL_FLOW_BACKEND=torchvision
SNR_MIN_DB=10
AUDIO_PEAK_MAX_DBFS=0
VRAM_ENFORCE_HARD_LIMIT=true

# === VGA Execution Contract ===
IMMUTABLE_CONTEXT_ENFORCE=true
IDENTITY_DRIFT_THRESHOLD=0.15
CROSS_MODAL_SYNC_THRESHOLD=0.9
HRG_REVIEW_ENABLED=true
HRG_APPROVAL_TIMEOUT_SECONDS=300
```

Load in Python:

```python
from dotenv import load_dotenv
load_dotenv("/workspace/.env_vga")
```

---

## 8. Full Batch Download Script

Save as `download_all_models.sh` and run once on the pod. Requires HF token for gated repos.

```bash
#!/usr/bin/env bash
set -euo pipefail

WORKSPACE="/workspace"
MODELS_DIR="$WORKSPACE/models"
LORAS_DIR="$WORKSPACE/loras"
AUX_DIR="$WORKSPACE/auxiliary"

echo "=== Enabling fast HF transfer ==="
pip install -U huggingface_hub hf-transfer
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_XET_HIGH_PERFORMANCE=1

# ── Authenticate (required for gated repos) ──────────────────────────────────
if [ -n "${HUGGING_FACE_HUB_TOKEN:-}" ]; then
    huggingface-cli login --token "$HUGGING_FACE_HUB_TOKEN"
else
    echo "WARNING: HUGGING_FACE_HUB_TOKEN not set — gated downloads will fail."
fi

# ── 1. CLIP (small — download first, dependency of many models) ──────────────
echo "=== 1. CLIP ViT-L/14 ==="
huggingface-cli download openai/clip-vit-large-patch14 \
    --local-dir "$AUX_DIR/clip"

# ── 2. Consistance LoRA ───────────────────────────────────────────────────────
echo "=== 2. Consistance_Edit_Lora ==="
huggingface-cli download lrzjason/Consistance_Edit_Lora \
    --local-dir "$LORAS_DIR/consistency"

# ── 3. SVI LoRAs (CORRECT filenames with version-2.0/ subfolder) ─────────────
echo "=== 3. SVI LoRAs (v2.0 Pro — Wan2.2) ==="
huggingface-cli download vita-video-gen/svi-model \
    --include "version-2.0/SVI_Wan2.2-I2V-A14B_high_noise_lora_v2.0_pro.safetensors" \
    --local-dir "$MODELS_DIR/svi"

huggingface-cli download vita-video-gen/svi-model \
    --include "version-2.0/SVI_Wan2.2-I2V-A14B_low_noise_lora_v2.0_pro.safetensors" \
    --local-dir "$MODELS_DIR/svi"

# Create symlinks in loras/svi/ for convenient access
mkdir -p "$LORAS_DIR/svi"
ln -sf "$MODELS_DIR/svi/version-2.0/SVI_Wan2.2-I2V-A14B_high_noise_lora_v2.0_pro.safetensors" \
       "$LORAS_DIR/svi/SVI_Wan2.2-I2V-A14B_high_noise_lora_v2.0_pro.safetensors"
ln -sf "$MODELS_DIR/svi/version-2.0/SVI_Wan2.2-I2V-A14B_low_noise_lora_v2.0_pro.safetensors" \
       "$LORAS_DIR/svi/SVI_Wan2.2-I2V-A14B_low_noise_lora_v2.0_pro.safetensors"

# ── 4. CosyVoice 3 ───────────────────────────────────────────────────────────
echo "=== 4. Fun-CosyVoice3-0.5B-2512 ==="
python -c "
from huggingface_hub import snapshot_download
snapshot_download('FunAudioLLM/Fun-CosyVoice3-0.5B-2512',
    local_dir='/workspace/CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B')
snapshot_download('FunAudioLLM/CosyVoice-ttsfrd',
    local_dir='/workspace/CosyVoice/pretrained_models/CosyVoice-ttsfrd')
"

# ── 5. MusicGen Medium ───────────────────────────────────────────────────────
echo "=== 5. MusicGen Medium ==="
huggingface-cli download facebook/musicgen-medium \
    --local-dir "$MODELS_DIR/musicgen"

# ── 6. MMAudio (weights auto-downloaded on first run) ────────────────────────
echo "=== 6. MMAudio (cloning repo) ==="
if [ ! -d "/workspace/MMAudio" ]; then
    git clone https://github.com/hkchengrex/MMAudio.git /workspace/MMAudio
    cd /workspace/MMAudio
    pip install --upgrade pip
    pip install -e .
    cd "$WORKSPACE"
fi
echo "Run 'python /workspace/MMAudio/demo.py --duration=8 --prompt test' to trigger weight download."
# OR: manual download from HuggingFace
huggingface-cli download hkchengrex/MMAudio \
    --local-dir /workspace/MMAudio

# ── 7. LatentSync-1.6 ───────────────────────────────────────────────────────
echo "=== 7. LatentSync-1.6 ==="
if [ ! -d "/workspace/LatentSync" ]; then
    git clone https://github.com/bytedance/LatentSync.git /workspace/LatentSync
    cd /workspace/LatentSync && pip install -r requirements.txt && cd "$WORKSPACE"
fi
huggingface-cli download ByteDance/LatentSync-1.6 \
    --local-dir /workspace/LatentSync/checkpoints

# ── 8. Qwen2.5-14B ──────────────────────────────────────────────────────────
echo "=== 8. Qwen2.5-14B-Instruct-unsloth-bnb-4bit ==="
huggingface-cli download unsloth/Qwen2.5-14B-Instruct-unsloth-bnb-4bit \
    --local-dir "$MODELS_DIR/qwen"

# ── 9. Z-Image-Turbo ────────────────────────────────────────────────────────
echo "=== 9. Z-Image-Turbo ==="
huggingface-cli download Tongyi-MAI/Z-Image-Turbo \
    --local-dir "$MODELS_DIR/zimage"

# ── 10. SVI Core Model (full repo — includes all LoRA versions) ──────────────
echo "=== 10. SVI Core (vita-video-gen/svi-model) ==="
huggingface-cli download vita-video-gen/svi-model \
    --local-dir "$MODELS_DIR/svi"

# ── 11. Wan2.2-I2V-A14B-FP8 (nalexand — largest download) ───────────────────
echo "=== 11. Wan2.2-I2V-A14B-FP8 (nalexand fork) ==="
huggingface-cli download nalexand/Wan2.2-I2V-A14B-FP8 \
    --local-dir "$MODELS_DIR/wan22"

# ── 12. Clone Wan2.2 inference repos ─────────────────────────────────────────
echo "=== 12. Wan2.2 inference repos ==="
if [ ! -d "/workspace/Wan2.2_nalexand" ]; then
    git clone https://github.com/nalexand/Wan2.2.git /workspace/Wan2.2_nalexand
fi
if [ ! -d "/workspace/Wan2.2_official" ]; then
    git clone https://github.com/Wan-Video/Wan2.2.git /workspace/Wan2.2_official
    cd /workspace/Wan2.2_official && pip install -r requirements.txt && cd "$WORKSPACE"
fi

# ── 13. SVI inference repo (svi_wan22 branch) ─────────────────────────────────
# svi_wan22 branch requires PyTorch 2.7.1 + cu128 (its own env — DiffSynth 2.0 base)
echo "=== 13. Stable-Video-Infinity (svi_wan22 branch) ==="
if [ ! -d "/workspace/Stable-Video-Infinity" ]; then
    git clone https://github.com/vita-epfl/Stable-Video-Infinity.git \
        -b svi_wan22 /workspace/Stable-Video-Infinity
fi
# Install into a dedicated conda env to isolate from main pipeline's cu124 stack:
# conda create -n svi_wan22 python=3.10 -y
# conda activate svi_wan22
# pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
# cd /workspace/Stable-Video-Infinity && pip install -e .
# pip install "flash_attn==2.8.0.post2" --no-build-isolation
# conda install -c conda-forge ffmpeg librosa libiconv -y
echo "SVI repo cloned at /workspace/Stable-Video-Infinity (branch: svi_wan22)"
echo "Run SVI setup in a dedicated conda env with PyTorch 2.7.1+cu128 — see Section 3.5"

# ── 14. FLUX.2-klein-4B (largest model — download last) ──────────────────────
echo "=== 14. FLUX.2-klein-4B ==="
huggingface-cli download black-forest-labs/FLUX.2-klein-4B \
    --local-dir "$MODELS_DIR/flux2"

echo ""
echo "=== ✅ All downloads complete! ==="
echo "SVI LoRA files in: $MODELS_DIR/svi/version-2.0/"
echo "Symlinks created in: $LORAS_DIR/svi/"
```

```bash
chmod +x download_all_models.sh
./download_all_models.sh
```

---

## 9. Verification Checklist

Run these checks after downloading to confirm each model is usable:

```bash
#!/usr/bin/env bash
echo "--- Verifying VGA model installations ---"

# 1. Qwen2.5
python -c "
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('/workspace/models/qwen')
print('[OK] Qwen2.5 tokenizer vocab size:', tok.vocab_size)
"

# 2. FLUX.2-klein-4B
python -c "
import os, json
p = '/workspace/models/flux2/model_index.json'
assert os.path.exists(p), 'model_index.json missing!'
print('[OK] FLUX.2-klein-4B model_index.json found')
"

# 3. Z-Image-Turbo
python -c "
from diffusers import ZImagePipeline
print('[OK] ZImagePipeline import successful')
" 2>/dev/null || echo "[WARN] ZImagePipeline not found — install diffusers from source"

# 4. Wan2.2-I2V-A14B-FP8
python -c "
import os
d = '/workspace/models/wan22'
files = os.listdir(d)
print('[OK] Wan2.2-FP8 files:', len(files), 'found')
"

# 5. SVI LoRAs — CORRECT FILENAMES
python -c "
import os
files = [
    '/workspace/models/svi/version-2.0/SVI_Wan2.2-I2V-A14B_high_noise_lora_v2.0_pro.safetensors',
    '/workspace/models/svi/version-2.0/SVI_Wan2.2-I2V-A14B_low_noise_lora_v2.0_pro.safetensors',
]
for f in files:
    assert os.path.exists(f), f'MISSING: {f}'
    print(f'[OK] {os.path.basename(f)} ({os.path.getsize(f) // 1024 // 1024} MB)')
"

# 6. LatentSync
python -c "
import os
assert os.path.exists('/workspace/LatentSync/checkpoints/latentsync_unet.pt'), 'UNet missing!'
assert os.path.exists('/workspace/LatentSync/checkpoints/whisper/tiny.pt'), 'Whisper missing!'
print('[OK] LatentSync checkpoints present')
"

# 7. CosyVoice3
python -c "
import os
assert os.path.exists('/workspace/CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B/cosyvoice3.yaml'), \
    'CosyVoice3 config missing!'
print('[OK] CosyVoice3 config found')
"

# 8. MusicGen
python -c "
from transformers import AutoProcessor
proc = AutoProcessor.from_pretrained('facebook/musicgen-medium')
print('[OK] MusicGen medium processor loaded')
"

# 9. CLIP
python -c "
from transformers import CLIPModel
m = CLIPModel.from_pretrained('/workspace/auxiliary/clip')
print('[OK] CLIP ViT-L/14 loaded, params:', sum(p.numel() for p in m.parameters()))
"

# 10. MMAudio
python -c "
import os
assert os.path.isdir('/workspace/MMAudio'), 'MMAudio repo not cloned!'
print('[OK] MMAudio repo present')
"

# 11. Consistance LoRA
python -c "
import os
d = '/workspace/loras/consistency'
assert os.path.isdir(d), 'Consistance_Edit_Lora directory missing!'
print('[OK] Consistance_Edit_Lora directory:', os.listdir(d))
"

# 12. PyTorch + CUDA check
python -c "
import torch
print(f'[OK] torch={torch.__version__} | cuda={torch.version.cuda} | gpu={torch.cuda.get_device_name(0)}')
assert torch.cuda.is_available(), 'CUDA not available!'
"

echo "--- Verification complete ---"
```

---

## Notes & Known Issues

| Issue | Resolution |
|-------|-----------|
| `ZImagePipeline not found` | Install diffusers from source: `pip install git+https://github.com/huggingface/diffusers.git` |
| `Flux2KleinPipeline not found` | Same — diffusers from source required |
| `flash-attn` build fails | Install ninja first (`pip install ninja`), ensure CUDA toolkit is on PATH, CUDA >= 12.0 required |
| MMAudio `ffmpeg<7` error | `conda install -c conda-forge 'ffmpeg<7'` — torchaudio requires this version cap |
| MMAudio PyTorch version error | MMAudio requires PyTorch **2.5.1+** — upgrade before cloning MMAudio |
| CosyVoice sox error | `sudo apt-get install sox libsox-dev` on Ubuntu (bootstrap includes this) |
| Wan2.2 OOM | Use nalexand FP8 fork (~8 GB VRAM); or `--offload_model True --convert_model_dtype` with official repo |
| `vita-video-gen/svi-model` access | Repo requires HF login: `huggingface-cli login` |
| SVI LoRA `FileNotFoundError` | LoRAs are inside `version-2.0/` subfolder — use `--include "version-2.0/SVI_Wan2.2-I2V-A14B_*"` |
| SVI LoRA wrong filenames | Correct: `SVI_Wan2.2-I2V-A14B_high_noise_lora_v2.0_pro.safetensors` (includes `-I2V-A14B_` and `lora_`) |
| SVI Wan2.2 wrong branch | Use `svi_wan22` branch, not `main` (main branch is for Wan2.1) |
| SVI env CUDA conflict | `svi_wan22` branch requires PyTorch 2.7.1 + cu128 (different from main VGA cu124 stack); run in separate conda env |
| SVI OOM on RTX 4090 | Use 480p resolution (480×832) — SVI training was at this resolution; higher resolutions cause OOM |
| LatentSync v1.5 → v1.6 | Only checkpoint file changes; update `resolution` in `configs/unet/*.yaml` |
| nalexand download dir | HF card example uses `./Wan2.2-T2V-A14B` — this is a typo; use your own target path |
| `ByteDance/LatentSync-1.6` gated | Requires HF login; use `huggingface-cli login` with a valid token |
| FLUX.2-klein-4B VRAM | Official HF card: "~13GB VRAM" (BF16); GitHub: "~8GB VRAM" (distilled sub-second mode). RTX 4090 (24 GB) handles both safely |

---

*Guide compiled from official HuggingFace model cards, GitHub repos, and community documentation — May 2026 (v7.2)*  
*SVI filenames verified against: https://huggingface.co/vita-video-gen/svi-model/tree/main/version-2.0*  
*SVI svi_wan22 install verified against: https://github.com/vita-epfl/Stable-Video-Infinity/tree/svi_wan22*  
*FLUX.2-klein-4B VRAM verified against: https://huggingface.co/black-forest-labs/FLUX.2-klein-4B and https://github.com/black-forest-labs/flux2*
