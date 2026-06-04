import os, sys, time, logging
from concurrent.futures import ThreadPoolExecutor, as_completed

os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'
# Token is loaded from /workspace/.env_vga (see bootstrap) — never hardcode here
if not os.environ.get('HUGGING_FACE_HUB_TOKEN'):
    raise RuntimeError('HUGGING_FACE_HUB_TOKEN not set — source /workspace/.env_vga first')
os.environ['HF_HOME'] = '/workspace/cache/huggingface'
os.environ['HUGGINGFACE_HUB_CACHE'] = '/workspace/cache/huggingface'
os.makedirs('/workspace/logs', exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('/workspace/logs/download_models.log', 'a'),
    ],
)
log = logging.getLogger('download')

from huggingface_hub import snapshot_download, hf_hub_download

# (name, type, repo_id, local_dir, kwargs)
# type='snapshot' or type='file' (for single files)
# Groups run concurrently within each group; groups run sequentially between each other.
# wan22 runs alone first (largest, 39 GB) to saturate bandwidth before splitting.
DOWNLOAD_GROUPS = [
    # Group 1: wan22 BF16 base model (~40 GB — fills the pipe; runs first)
    # Upgraded from FP8 (nalexand/Wan2.2-I2V-A14B-FP8) to official BF16 base.
    # RTX PRO 6000 96GB: both DiTs GPU-resident simultaneously, no offloading.
    [
        ('wan22_bf16', 'snapshot', 'Wan-AI/Wan2.2-I2V-A14B',
         '/workspace/models/wan22_bf16', {}),
    ],
    # Group 2: small fast models in parallel
    [
        ('clip', 'snapshot', 'openai/clip-vit-large-patch14', '/workspace/auxiliary/clip',
         {'ignore_patterns': ['flax_model.msgpack', 'tf_model.h5', 'pytorch_model.bin']}),
        ('svi_high_noise', 'file',
         ('vita-video-gen/svi-model',
          'version-2.0/SVI_Wan2.2-I2V-A14B_high_noise_lora_v2.0_pro.safetensors',
          '/workspace/loras/svi'),
         None, {}),
        ('svi_low_noise', 'file',
         ('vita-video-gen/svi-model',
          'version-2.0/SVI_Wan2.2-I2V-A14B_low_noise_lora_v2.0_pro.safetensors',
          '/workspace/loras/svi'),
         None, {}),
        ('lora_consistency', 'file',
         ('lrzjason/Consistance_Edit_Lora', 'f2k_4B_consist_20260314.safetensors',
          '/workspace/loras/consistency'),
         None, {}),
    ],
    # Group 3: medium models in parallel (4–8 GB each)
    [
        ('qwen3', 'snapshot', 'Qwen/Qwen3-14B',
         '/workspace/models/qwen', {}),
        ('flux2', 'snapshot', 'black-forest-labs/FLUX.2-klein-4B',
         '/workspace/models/flux2', {}),
        ('zimage', 'snapshot', 'Tongyi-MAI/Z-Image-Turbo',
         '/workspace/models/zimage', {}),
        ('mmaudio', 'snapshot', 'hkchengrex/MMAudio', '/workspace/models/mmaudio',
         {'allow_patterns': ['weights/mmaudio_medium_44k.pth', 'ext_weights/*']}),
    ],
    # Group 4: remaining models in parallel
    [
        ('musicgen', 'snapshot', 'facebook/musicgen-medium', '/workspace/models/musicgen',
         {'ignore_patterns': ['state_dict.bin', 'compression_state_dict.bin']}),
        ('latentsync', 'snapshot', 'ByteDance/LatentSync-1.6',
         '/workspace/models/latentsync', {}),
        ('cosyvoice', 'snapshot', 'FunAudioLLM/Fun-CosyVoice3-0.5B-2512',
         '/workspace/CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B', {}),
    ],
]


def download_one(entry):
    name, typ = entry[0], entry[1]
    t0 = time.monotonic()
    try:
        if typ == 'snapshot':
            repo_id = entry[2]
            local_dir = entry[3]
            kwargs = entry[4]
            os.makedirs(local_dir, exist_ok=True)
            existing = [f for f in os.listdir(local_dir) if not f.startswith('.')]
            if existing:
                log.info('%s: already present (%d files) -- skipping', name, len(existing))
                return name, True
            log.info('%s: downloading from %s -> %s', name, repo_id, local_dir)
            snapshot_download(repo_id=repo_id, local_dir=local_dir, **kwargs)
        else:
            repo_id, filename, local_dir = entry[2]
            os.makedirs(local_dir, exist_ok=True)
            dest = os.path.join(local_dir, os.path.basename(filename))
            if os.path.exists(dest):
                log.info('%s: already present -- skipping', name)
                return name, True
            log.info('%s: downloading %s from %s', name, filename, repo_id)
            hf_hub_download(repo_id=repo_id, filename=filename, local_dir=local_dir)
        elapsed = time.monotonic() - t0
        log.info('%s: done in %.0fs', name, elapsed)
        return name, True
    except Exception as exc:
        log.error('%s: FAILED: %s', name, exc)
        return name, False


for g_idx, group in enumerate(DOWNLOAD_GROUPS):
    log.info('--- Starting download group %d (%d items) ---', g_idx + 1, len(group))
    with ThreadPoolExecutor(max_workers=len(group)) as pool:
        futures = {pool.submit(download_one, entry): entry[0] for entry in group}
        for fut in as_completed(futures):
            name, ok = fut.result()
            if not ok:
                log.error('Group %d: %s failed', g_idx + 1, name)

log.info('All downloads complete.')

# Fix LoRA path: hf_hub_download nests in version-2.0/ subfolder; move up one level.
import shutil
svi_dir = '/workspace/loras/svi'
nested = os.path.join(svi_dir, 'version-2.0')
if os.path.isdir(nested):
    for f in os.listdir(nested):
        src = os.path.join(nested, f)
        dst = os.path.join(svi_dir, f)
        if not os.path.exists(dst):
            shutil.move(src, dst)
            log.info('Moved LoRA %s -> %s', src, dst)
    try:
        os.rmdir(nested)
    except OSError:
        pass
