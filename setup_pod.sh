#!/bin/bash
# VGA pod setup — run once after pod creation.
# Automatically writes .env_vga, installs packages, clones DiffSynth,
# and starts model downloads. Zero manual intervention required.
# Usage: bash /workspace/vga_repo/setup_pod.sh
set -e

echo "=== VGA RTX PRO 6000 Blackwell Pod Setup ==="

# Workspace dirs
mkdir -p /workspace/{output,logs,hrg,models,loras,auxiliary,cache/huggingface,state,scripts}

# Write .env_vga fresh (overwrite, never append — avoids duplicate vars on re-run)
# HF_TOKEN must be set in environment before running this script:
#   export HF_TOKEN=hf_...your_token...
#   bash setup_pod.sh
if [ -z "${HF_TOKEN}" ]; then
    echo "ERROR: HF_TOKEN environment variable not set. Export it before running."
    echo "  export HF_TOKEN=hf_your_token_here"
    exit 1
fi

cat > /workspace/.env_vga << ENVEOF
HUGGING_FACE_HUB_TOKEN=${HF_TOKEN}
HF_TOKEN=${HF_TOKEN}
HF_HOME=/workspace/cache/huggingface
HUGGINGFACE_HUB_CACHE=/workspace/cache/huggingface
SVI_GPU_RESIDENT=1
WAN22_PRECISION=bf16
WAN22_BF16_DIR=/workspace/models/wan22_bf16
SVI_TEA_CACHE_THRESH=0.0
HF_HUB_ENABLE_HF_TRANSFER=1
ENVEOF
echo "✓ .env_vga written"

# Install Python packages
# torch 2.8.0+cu128 is pre-installed — skip it
# --break-system-packages required on Ubuntu 24.04 (PEP 668)
echo "--- Installing Python packages ---"
pip install --break-system-packages -q \
    git+https://github.com/huggingface/diffusers.git \
    "transformers>=5.9.0" accelerate==1.6.0 tokenizers sentencepiece
pip install --break-system-packages -q \
    "pydantic>=2.7,<3" pydantic-settings fastapi uvicorn python-dotenv bitsandbytes
pip install --break-system-packages -q \
    safetensors "huggingface-hub>=0.24" hf-transfer imageio imageio-ffmpeg
pip install --break-system-packages -q \
    opencv-python-headless Pillow numpy scipy open_clip_torch
echo "✓ Packages installed"

# Clone DiffSynth svi_wan22 (only if not already present)
if [ ! -d /workspace/Stable-Video-Infinity ]; then
    echo "--- Cloning DiffSynth svi_wan22 ---"
    git clone --branch svi_wan22 --depth 1 \
        https://github.com/vita-epfl/Stable-Video-Infinity.git \
        /workspace/Stable-Video-Infinity
fi
pip install --break-system-packages -q -e /workspace/Stable-Video-Infinity
echo "✓ DiffSynth installed"

echo "=== Verifying installs ==="
python3 -c "
import torch, diffsynth, pydantic, diffusers, transformers
print('torch:', torch.__version__)
print('cuda available:', torch.cuda.is_available())
print('pydantic:', pydantic.VERSION)
print('diffusers:', diffusers.__version__)
print('diffsynth: OK')
"

echo ""
echo "=== Starting model downloads (nohup, survives terminal close) ==="
# Export env vars explicitly so nohup subprocess picks them up (source alone is not enough)
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"
export HF_HOME=/workspace/cache/huggingface
export HF_HUB_ENABLE_HF_TRANSFER=1

nohup python3 /workspace/vga_repo/download_models.py \
    > /workspace/logs/download_models.log 2>&1 &
DOWNLOAD_PID=$!
echo "✓ Download started — PID: $DOWNLOAD_PID"
echo "  Monitor: tail -f /workspace/logs/download_models.log"
echo ""
echo "=== Setup complete — downloads running in background ==="
