#!/bin/bash
# VGA v17.2 — ComfyUI Pod Setup (fresh pod)
# Run once after pod creation:
#   bash /workspace/vga_repo/setup_comfyui_pod.sh
# OR one-liner (after cloning repo):
#   bash <(curl -fsSL https://raw.githubusercontent.com/Richmondiroegbu/vga-system/main/setup_comfyui_pod.sh)
set -e

HF_TOKEN="${HF_TOKEN:-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}"
WORKSPACE=/workspace
COMFYUI=$WORKSPACE/ComfyUI
MODELS=$WORKSPACE/models

echo "==========================================="
echo " VGA v17.2 ComfyUI Pod Setup"
echo "==========================================="

# ── Workspace directories ────────────────────────────────────────────────────
mkdir -p $WORKSPACE/{models,loras,auxiliary,cache/huggingface,logs,output}
mkdir -p $COMFYUI/models/{diffusion_models,vae,text_encoders,loras,ipadapter,clip,checkpoints}
mkdir -p $COMFYUI/custom_nodes

# ── Write .env ───────────────────────────────────────────────────────────────
cat > $WORKSPACE/.env_vga << ENVEOF
HUGGING_FACE_HUB_TOKEN=${HF_TOKEN}
HF_TOKEN=${HF_TOKEN}
HF_HOME=$WORKSPACE/cache/huggingface
HF_HUB_ENABLE_HF_TRANSFER=1
SVI_GPU_RESIDENT=1
WAN22_PRECISION=bf16
WAN22_BF16_DIR=$WORKSPACE/models/wan22_bf16
ENVEOF
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"
export HF_HOME=$WORKSPACE/cache/huggingface
export HF_HUB_ENABLE_HF_TRANSFER=1
echo "✓ Environment configured"

# ── Install Python packages ──────────────────────────────────────────────────
echo "--- Installing Python packages ---"
pip install --break-system-packages -q \
    "huggingface-hub>=0.24" hf-transfer safetensors \
    "transformers>=5.9.0" accelerate \
    "pydantic>=2.7,<3" python-dotenv \
    opencv-python-headless Pillow numpy scipy
echo "✓ Packages installed"

# ── Install ComfyUI ──────────────────────────────────────────────────────────
if [ ! -d "$COMFYUI/.git" ]; then
    echo "--- Installing ComfyUI ---"
    git clone https://github.com/comfyanonymous/ComfyUI.git $COMFYUI
    pip install --break-system-packages -q -r $COMFYUI/requirements.txt
    echo "✓ ComfyUI installed"
else
    echo "✓ ComfyUI already present"
fi

# ── Install custom nodes ─────────────────────────────────────────────────────
echo "--- Installing custom nodes ---"
cd $COMFYUI/custom_nodes

declare -A NODES=(
    ["ComfyUI-IPAdapter-Flux"]="https://github.com/Shakker-Labs/ComfyUI-IPAdapter-Flux.git"
    ["ComfyUI-Wan-SVI2Pro-FLF"]="https://github.com/Well-Made/ComfyUI-Wan-SVI2Pro-FLF.git"
    ["ComfyUI-LatentSyncWrapper"]="https://github.com/ShmuelRonen/ComfyUI-LatentSyncWrapper.git"
    ["ComfyUI-pause"]="https://github.com/wywywywy/ComfyUI-pause.git"
    ["ComfyUI-MMAudio"]="https://github.com/kijai/ComfyUI-MMAudio.git"
    ["ComfyUI-MusicGen-HF"]="https://github.com/ebrinz/ComfyUI-MusicGen-HF.git"
    ["ComfyUI-Manager"]="https://github.com/ltdrdata/ComfyUI-Manager.git"
)

for name in "${!NODES[@]}"; do
    url="${NODES[$name]}"
    if [ ! -d "$name" ]; then
        git clone --depth 1 "$url" "$name" 2>/dev/null && echo "  ✓ $name"
        [ -f "$name/requirements.txt" ] && pip install --break-system-packages -q -r "$name/requirements.txt" 2>/dev/null
    else
        echo "  ✓ $name (already present)"
    fi
done
echo "✓ Custom nodes installed"

# ── Download all models simultaneously ───────────────────────────────────────
echo "--- Downloading models (all simultaneously, ~30 min on US pod) ---"

python3 << 'PYEOF'
import os, sys, time, shutil, logging
from concurrent.futures import ThreadPoolExecutor, as_completed

os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'
if not os.environ.get('HUGGING_FACE_HUB_TOKEN'):
    raise RuntimeError('HUGGING_FACE_HUB_TOKEN not set')

os.environ['HF_HOME'] = '/workspace/cache/huggingface'
os.makedirs('/workspace/logs', exist_ok=True)

logging.basicConfig(level=logging.INFO,
    format='[%(asctime)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout),
              logging.FileHandler('/workspace/logs/download_models.log', 'a')])
log = logging.getLogger('download')

from huggingface_hub import snapshot_download, hf_hub_download

DOWNLOADS = [
    # Wan2.2 BF16 base model (largest — starts first)
    ('wan22_bf16', 'snapshot', 'Wan-AI/Wan2.2-I2V-A14B', '/workspace/models/wan22_bf16', {}),
    # SVI LoRAs
    ('svi_high', 'file', ('vita-video-gen/svi-model',
     'version-2.0/SVI_Wan2.2-I2V-A14B_high_noise_lora_v2.0_pro.safetensors',
     '/workspace/loras/svi'), None, {}),
    ('svi_low', 'file', ('vita-video-gen/svi-model',
     'version-2.0/SVI_Wan2.2-I2V-A14B_low_noise_lora_v2.0_pro.safetensors',
     '/workspace/loras/svi'), None, {}),
    # Consistency LoRA (FLUX)
    ('lora_consist', 'file', ('lrzjason/Consistance_Edit_Lora',
     'f2k_4B_consist_20260314.safetensors', '/workspace/loras/consistency'), None, {}),
    # CLIP
    ('clip', 'snapshot', 'openai/clip-vit-large-patch14', '/workspace/auxiliary/clip',
     {'ignore_patterns': ['flax_model.msgpack', 'tf_model.h5', 'pytorch_model.bin']}),
    # LLM (Qwen3)
    ('qwen3', 'snapshot', 'Qwen/Qwen3-14B', '/workspace/models/qwen', {}),
    # FLUX.2-klein-4B
    ('flux2', 'snapshot', 'black-forest-labs/FLUX.2-klein-4B', '/workspace/models/flux2', {}),
    # Z-Image-Turbo
    ('zimage', 'snapshot', 'Tongyi-MAI/Z-Image-Turbo', '/workspace/models/zimage', {}),
    # MMAudio
    ('mmaudio', 'snapshot', 'hkchengrex/MMAudio', '/workspace/models/mmaudio',
     {'allow_patterns': ['weights/mmaudio_medium_44k.pth', 'ext_weights/*']}),
    # MusicGen
    ('musicgen', 'snapshot', 'facebook/musicgen-medium', '/workspace/models/musicgen',
     {'ignore_patterns': ['state_dict.bin', 'compression_state_dict.bin']}),
    # CosyVoice3
    ('cosyvoice', 'snapshot', 'FunAudioLLM/Fun-CosyVoice3-0.5B-2512',
     '/workspace/CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B', {}),
    # LatentSync
    ('latentsync', 'snapshot', 'ByteDance/LatentSync-1.6',
     '/workspace/models/latentsync', {}),
    # IP-Adapter FLUX (for ComfyUI)
    ('ipadapter_flux', 'file', ('Shakker-Labs/FLUX.1-dev-IP-Adapter',
     'ip-adapter.bin', '/workspace/models/ipadapter'), None, {}),
]

def download_one(entry):
    name, typ = entry[0], entry[1]
    t0 = time.monotonic()
    try:
        if typ == 'snapshot':
            repo_id, local_dir, kwargs = entry[2], entry[3], entry[4]
            os.makedirs(local_dir, exist_ok=True)
            log.info('%s: starting %s', name, repo_id)
            snapshot_download(repo_id=repo_id, local_dir=local_dir, **kwargs)
        else:
            repo_id, filename, local_dir = entry[2]
            os.makedirs(local_dir, exist_ok=True)
            dest = os.path.join(local_dir, os.path.basename(filename))
            if os.path.exists(dest):
                log.info('%s: already present — skipping', name)
                return name, True
            log.info('%s: downloading %s', name, filename)
            hf_hub_download(repo_id=repo_id, filename=filename, local_dir=local_dir)
        elapsed = time.monotonic() - t0
        log.info('%s: DONE in %.1f min', name, elapsed/60)
        return name, True
    except Exception as exc:
        log.error('%s: FAILED: %s', name, exc)
        return name, False

log.info('=== Starting %d downloads simultaneously ===', len(DOWNLOADS))
t_start = time.monotonic()
failed = []
with ThreadPoolExecutor(max_workers=len(DOWNLOADS)) as pool:
    futures = {pool.submit(download_one, e): e[0] for e in DOWNLOADS}
    for fut in as_completed(futures):
        name, ok = fut.result()
        if not ok:
            failed.append(name)
elapsed = time.monotonic() - t_start
log.info('=== All downloads complete in %.1f min ===', elapsed/60)
if failed:
    log.error('FAILED: %s', failed)

# Fix SVI LoRA path nesting
svi_dir = '/workspace/loras/svi'
nested = os.path.join(svi_dir, 'version-2.0')
if os.path.isdir(nested):
    for f in os.listdir(nested):
        src, dst = os.path.join(nested, f), os.path.join(svi_dir, f)
        if not os.path.exists(dst):
            shutil.move(src, dst)
    try: os.rmdir(nested)
    except: pass
PYEOF

echo "✓ All models downloaded"

# ── Merge Wan2.2 BF16 shards → single files for ComfyUI ────────────────────
echo "--- Merging Wan2.2 BF16 shards for ComfyUI (10-15 min) ---"
python3 << 'PYEOF'
import glob, os, time
from safetensors.torch import load_file, save_file

COMFYUI_DIFFUSION = '/workspace/ComfyUI/models/diffusion_models'
os.makedirs(COMFYUI_DIFFUSION, exist_ok=True)

for noise_level in ['high', 'low']:
    out = f'{COMFYUI_DIFFUSION}/wan2.2_i2v_{noise_level}_noise_14B_bf16.safetensors'
    if os.path.exists(out):
        print(f'  {noise_level}_noise: already merged')
        continue
    shard_dir = f'/workspace/models/wan22_bf16/{noise_level}_noise_model'
    shards = sorted(glob.glob(f'{shard_dir}/diffusion_pytorch_model*.safetensors'))
    if not shards:
        print(f'  ERROR: no shards found in {shard_dir}')
        continue
    print(f'  Merging {noise_level}_noise ({len(shards)} shards)...')
    t0 = time.monotonic()
    sd = {}
    for s in shards:
        sd.update(load_file(s, device='cpu'))
    save_file(sd, out)
    del sd
    size = os.path.getsize(out)/1e9
    print(f'  ✓ {noise_level}_noise: {size:.1f}GB in {(time.monotonic()-t0)/60:.1f}min')

print('Merge complete')
PYEOF
echo "✓ Wan2.2 BF16 merged"

# ── Symlink remaining models into ComfyUI directories ───────────────────────
echo "--- Setting up ComfyUI model symlinks ---"
COMFYUI=/workspace/ComfyUI

# Wan VAE
ln -sf /workspace/models/wan22_bf16/Wan2.1_VAE.pth \
    $COMFYUI/models/vae/wan_2.1_vae.pth 2>/dev/null || true

# UMT5-XXL (Wan text encoder)
ln -sf /workspace/models/wan22_bf16/models_t5_umt5-xxl-enc-bf16.pth \
    $COMFYUI/models/text_encoders/umt5_xxl_bf16.pth 2>/dev/null || true

# FLUX VAE
find /workspace/models/flux2 -name "ae.safetensors" -exec \
    ln -sf {} $COMFYUI/models/vae/ae.safetensors \; 2>/dev/null || true

# FLUX text encoders
find /workspace/models/flux2 -name "t5xxl_fp16.safetensors" -exec \
    ln -sf {} $COMFYUI/models/text_encoders/ \; 2>/dev/null || true
find /workspace/models/flux2 -name "clip_l.safetensors" -exec \
    ln -sf {} $COMFYUI/models/text_encoders/ \; 2>/dev/null || true

# FLUX UNet
find /workspace/models/flux2 -name "flux1-*.safetensors" -exec \
    ln -sf {} $COMFYUI/models/diffusion_models/ \; 2>/dev/null || true

# SVI LoRAs
ln -sf /workspace/loras/svi/SVI_Wan2.2-I2V-A14B_high_noise_lora_v2.0_pro.safetensors \
    $COMFYUI/models/loras/ 2>/dev/null || true
ln -sf /workspace/loras/svi/SVI_Wan2.2-I2V-A14B_low_noise_lora_v2.0_pro.safetensors \
    $COMFYUI/models/loras/ 2>/dev/null || true

# Consistency LoRA
ln -sf /workspace/loras/consistency/f2k_4B_consist_20260314.safetensors \
    $COMFYUI/models/loras/ 2>/dev/null || true

# IP-Adapter
find /workspace/models/ipadapter -name "*.bin" -o -name "*.safetensors" | \
    xargs -I{} ln -sf {} $COMFYUI/models/ipadapter/ 2>/dev/null || true

# CLIP
find /workspace/auxiliary/clip -name "*.safetensors" -exec \
    ln -sf {} $COMFYUI/models/clip/ \; 2>/dev/null || true

echo "✓ Symlinks created"

# ── Copy VGA workflow into ComfyUI ───────────────────────────────────────────
if [ -f /workspace/vga_repo/comfyui_vga_pipeline.json ]; then
    cp /workspace/vga_repo/comfyui_vga_pipeline.json \
       $COMFYUI/user/default/workflows/vga_pipeline.json 2>/dev/null || \
    mkdir -p $COMFYUI/user/default/workflows && \
    cp /workspace/vga_repo/comfyui_vga_pipeline.json \
       $COMFYUI/user/default/workflows/vga_pipeline.json
    echo "✓ VGA workflow installed in ComfyUI"
fi

# ── Start ComfyUI ────────────────────────────────────────────────────────────
echo ""
echo "=== Starting ComfyUI on port 8188 ==="
cd $COMFYUI
nohup python main.py --listen 0.0.0.0 --port 8188 \
    > /workspace/logs/comfyui.log 2>&1 &
echo "ComfyUI PID: $!"
sleep 5
if curl -s http://localhost:8188 > /dev/null 2>&1; then
    echo "✓ ComfyUI is running at http://localhost:8188"
else
    echo "  ComfyUI starting... check: tail -f /workspace/logs/comfyui.log"
fi

echo ""
echo "==========================================="
echo " SETUP COMPLETE"
echo "==========================================="
echo ""
echo "Access ComfyUI via SSH tunnel from your PC:"
echo "  ssh -L 8188:localhost:8188 root@POD_IP -p POD_PORT -i ~/.ssh/id_ed25519 -N"
echo "Then open: http://localhost:8188"
echo ""
echo "The VGA workflow is pre-loaded in:"
echo "  ComfyUI menu → Workflows → vga_pipeline"
