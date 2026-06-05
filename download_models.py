"""
VGA v17.2 Model Downloader — all models download simultaneously.

Every model runs in its own thread at the same time. Total time ≈ time
for the slowest single model (Wan2.2 BF16, ~118 GB) rather than the sum
of all download times. On a US RunPod pod with 250–500 MB/s bandwidth
the full stack completes in ~15–20 minutes.

Resume-safe: skips any model whose local_dir already has files.
"""
import os, sys, time, shutil, logging
from concurrent.futures import ThreadPoolExecutor, as_completed

# ── Environment ──────────────────────────────────────────────────────────────
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'

if not os.environ.get('HUGGING_FACE_HUB_TOKEN'):
    raise RuntimeError(
        'HUGGING_FACE_HUB_TOKEN not set.\n'
        'Run:  export HUGGING_FACE_HUB_TOKEN=hf_...  then retry.'
    )

os.environ['HF_HOME'] = '/workspace/cache/huggingface'
os.environ['HUGGINGFACE_HUB_CACHE'] = '/workspace/cache/huggingface'
os.makedirs('/workspace/logs', exist_ok=True)

# ── Logging ──────────────────────────────────────────────────────────────────
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

# ── Download Manifest ────────────────────────────────────────────────────────
# ALL entries run simultaneously in parallel threads.
# Entry format — snapshot:  (name, 'snapshot', repo_id, local_dir, kwargs)
# Entry format — file:      (name, 'file', (repo_id, filename, local_dir), None, {})
ALL_DOWNLOADS = [

    # ── Video generation (largest — starts first, finishes last) ─────────────
    ('wan22_bf16', 'snapshot',
     'Wan-AI/Wan2.2-I2V-A14B',
     '/workspace/models/wan22_bf16', {}),

    # ── SVI LoRAs (tiny — finish in seconds) ─────────────────────────────────
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
     ('lrzjason/Consistance_Edit_Lora',
      'f2k_4B_consist_20260314.safetensors',
      '/workspace/loras/consistency'),
     None, {}),

    # ── CLIP identity validator ───────────────────────────────────────────────
    ('clip', 'snapshot',
     'openai/clip-vit-large-patch14',
     '/workspace/auxiliary/clip',
     {'ignore_patterns': ['flax_model.msgpack', 'tf_model.h5', 'pytorch_model.bin']}),

    # ── LLM ──────────────────────────────────────────────────────────────────
    ('qwen3', 'snapshot',
     'Qwen/Qwen3-14B',
     '/workspace/models/qwen', {}),

    # ── Image generation ──────────────────────────────────────────────────────
    ('flux2', 'snapshot',
     'black-forest-labs/FLUX.2-klein-4B',
     '/workspace/models/flux2', {}),

    # ── Image refinement ─────────────────────────────────────────────────────
    ('zimage', 'snapshot',
     'Tongyi-MAI/Z-Image-Turbo',
     '/workspace/models/zimage', {}),

    # ── Audio ─────────────────────────────────────────────────────────────────
    ('mmaudio', 'snapshot',
     'hkchengrex/MMAudio',
     '/workspace/models/mmaudio',
     {'allow_patterns': ['weights/mmaudio_medium_44k.pth', 'ext_weights/*']}),

    ('musicgen', 'snapshot',
     'facebook/musicgen-medium',
     '/workspace/models/musicgen',
     {'ignore_patterns': ['state_dict.bin', 'compression_state_dict.bin']}),

    ('cosyvoice', 'snapshot',
     'FunAudioLLM/Fun-CosyVoice3-0.5B-2512',
     '/workspace/CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B', {}),

    # ── Lip sync ─────────────────────────────────────────────────────────────
    ('latentsync', 'snapshot',
     'ByteDance/LatentSync-1.6',
     '/workspace/models/latentsync', {}),
]


# ── Worker ───────────────────────────────────────────────────────────────────

def download_one(entry):
    name, typ = entry[0], entry[1]
    t0 = time.monotonic()
    try:
        if typ == 'snapshot':
            repo_id, local_dir, kwargs = entry[2], entry[3], entry[4]
            os.makedirs(local_dir, exist_ok=True)
            # Always call snapshot_download — it skips already-complete files
            # internally and resumes partial downloads. Never skip here.
            log.info('%s: starting/resuming  %s → %s', name, repo_id, local_dir)
            snapshot_download(repo_id=repo_id, local_dir=local_dir, **kwargs)
        else:
            repo_id, filename, local_dir = entry[2]
            os.makedirs(local_dir, exist_ok=True)
            dest = os.path.join(local_dir, os.path.basename(filename))
            if os.path.exists(dest):
                log.info('%s: already present — skipping', name)
                return name, True
            log.info('%s: starting  %s', name, filename)
            hf_hub_download(repo_id=repo_id, filename=filename, local_dir=local_dir)

        elapsed = time.monotonic() - t0
        log.info('%s: DONE in %.0fs (%.1f min)', name, elapsed, elapsed / 60)
        return name, True

    except Exception as exc:
        log.error('%s: FAILED: %s', name, exc)
        return name, False


# ── Run all downloads simultaneously ─────────────────────────────────────────

log.info('=== Starting all %d downloads simultaneously ===', len(ALL_DOWNLOADS))
t_start = time.monotonic()

failed = []
with ThreadPoolExecutor(max_workers=len(ALL_DOWNLOADS)) as pool:
    futures = {pool.submit(download_one, entry): entry[0] for entry in ALL_DOWNLOADS}
    for fut in as_completed(futures):
        name, ok = fut.result()
        if not ok:
            failed.append(name)
            log.error('FAILED: %s', name)

elapsed_total = time.monotonic() - t_start
log.info('=== All downloads complete in %.0fs (%.1f min) ===', elapsed_total, elapsed_total / 60)

if failed:
    log.error('FAILED models (re-run to retry): %s', failed)

# ── Fix SVI LoRA path ────────────────────────────────────────────────────────
# hf_hub_download nests files inside version-2.0/ subfolder; flatten it.
svi_dir = '/workspace/loras/svi'
nested = os.path.join(svi_dir, 'version-2.0')
if os.path.isdir(nested):
    for f in os.listdir(nested):
        src = os.path.join(nested, f)
        dst = os.path.join(svi_dir, f)
        if not os.path.exists(dst):
            shutil.move(src, dst)
            log.info('Moved LoRA %s → %s', f, svi_dir)
    try:
        os.rmdir(nested)
    except OSError:
        pass
