#!/bin/bash
# VGA pod setup — run once after pod creation.
# Usage: source /workspace/.env_vga && bash /workspace/vga_repo/setup_pod.sh
set -e

echo "=== VGA RTX PRO 6000 Blackwell Pod Setup ==="

# Workspace dirs
mkdir -p /workspace/{output,logs,hrg,models,loras,auxiliary,cache/huggingface}

# Copy VGA source into workspace (if not already there)
[ -d /workspace/vga ] || cp -r /workspace/vga_repo/vga /workspace/
[ -d /workspace/scripts ] || cp -r /workspace/vga_repo/scripts /workspace/
[ -f /workspace/download_models.py ] || cp /workspace/vga_repo/download_models.py /workspace/

# Install Python packages
# torch 2.8.0+cu128 is pre-installed on this pod — skip it
# --break-system-packages required on Ubuntu 24.04 (PEP 668 externally-managed-environment)
pip install --break-system-packages -q diffusers==0.32.2 transformers==4.51.3 accelerate==1.6.0 \
    tokenizers sentencepiece
pip install --break-system-packages -q "pydantic>=2.7,<3" pydantic-settings fastapi uvicorn python-dotenv bitsandbytes
pip install --break-system-packages -q safetensors "huggingface-hub>=0.24" hf-transfer imageio imageio-ffmpeg
pip install --break-system-packages -q opencv-python-headless Pillow numpy scipy

# Clone DiffSynth svi_wan22 (only if not already present)
if [ ! -d /workspace/Stable-Video-Infinity ]; then
    echo "--- Cloning DiffSynth svi_wan22 ---"
    git clone --branch svi_wan22 --depth 1 \
        https://github.com/vita-epfl/Stable-Video-Infinity.git \
        /workspace/Stable-Video-Infinity
fi
pip install --break-system-packages -q -e /workspace/Stable-Video-Infinity

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
echo "=== Setup complete — starting model downloads in background ==="
nohup bash -c 'source /workspace/.env_vga && python3 /workspace/download_models.py \
    > /workspace/logs/download_models.log 2>&1' &
echo "Download PID: $! — tail /workspace/logs/download_models.log to monitor"
