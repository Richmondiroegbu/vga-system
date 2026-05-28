"""
Downloads Wan2.2 I2V A14B high/low noise DiTs one at a time,
converts each from BF16 sharded safetensors to FP8 split-block format
(blocks.N.safetensors + component files), then deletes BF16 source.
"""
import json
import os
import shutil
import gc
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

TOKEN = os.environ.get("HUGGING_FACE_HUB_TOKEN", "")
if not TOKEN:
    raise RuntimeError("HUGGING_FACE_HUB_TOKEN not set — source /workspace/.env_vga first")
WAN22_DIR = Path("/workspace/models/wan22")
REPO_ID = "Wan-AI/Wan2.2-I2V-A14B"

COMPONENT_PREFIXES = [
    "patch_embedding", "text_embedding",
    "time_embedding", "time_projection", "head",
]


def convert_dit(bf16_dir: Path, fp8_dir: Path):
    fp8_dir.mkdir(exist_ok=True)
    index_file = bf16_dir / "diffusion_pytorch_model.safetensors.index.json"
    with open(index_file) as f:
        index = json.load(f)
    weight_map = index["weight_map"]  # key -> shard filename

    # Group keys by block number or component prefix
    block_keys = {}
    comp_keys = {}
    for key in weight_map:
        if key.startswith("blocks."):
            parts = key.split(".")
            n = int(parts[1])
            block_keys.setdefault(n, []).append(key)
        else:
            matched = False
            for pfx in COMPONENT_PREFIXES:
                if key.startswith(pfx + ".") or key == pfx:
                    comp_keys.setdefault(pfx, []).append(key)
                    matched = True
                    break
            if not matched:
                comp_keys.setdefault("misc", []).append(key)

    num_blocks = max(block_keys.keys()) + 1 if block_keys else 40
    print(f"  Found {num_blocks} blocks, {len(comp_keys)} component groups")

    # Open shard files with mmap — no full load into RAM
    shard_handles = {}
    for shard in set(weight_map.values()):
        shard_handles[shard] = safe_open(str(bf16_dir / shard), framework="pt", device="cpu")

    def load_fp8(keys_list):
        tensors = {}
        for key in keys_list:
            shard = weight_map[key]
            t = shard_handles[shard].get_tensor(key)
            tensors[key] = t.to(torch.float8_e4m3fn)
        return tensors

    # Convert each transformer block
    for n in range(num_blocks):
        if n not in block_keys:
            print(f"  WARNING: block {n} has no keys, skipping")
            continue
        tensors_fp8 = load_fp8(block_keys[n])
        prefix = f"blocks.{n}."
        block_tensors = {k[len(prefix):]: v for k, v in tensors_fp8.items()}
        out_file = fp8_dir / f"blocks.{n}.safetensors"
        save_file(block_tensors, str(out_file))
        if n % 8 == 0:
            print(f"  blocks.{n} saved ({len(block_tensors)} tensors)")
        del tensors_fp8, block_tensors
        gc.collect()

    # Convert component files
    for pfx, keys in comp_keys.items():
        if pfx == "misc":
            continue
        tensors_fp8 = load_fp8(keys)
        comp_tensors = {
            (k[len(pfx) + 1:] if k.startswith(pfx + ".") else k): v
            for k, v in tensors_fp8.items()
        }
        out_file = fp8_dir / f"{pfx}.safetensors"
        save_file(comp_tensors, str(out_file))
        print(f"  {pfx}.safetensors saved ({len(comp_tensors)} tensors)")
        del tensors_fp8, comp_tensors
        gc.collect()

    for h in shard_handles.values():
        del h
    gc.collect()
    print(f"  Conversion complete -> {fp8_dir}")


def download_and_convert(model_subdir: str):
    from huggingface_hub import snapshot_download

    bf16_dir = WAN22_DIR / model_subdir
    fp8_dir = WAN22_DIR / (model_subdir + "_fp8")

    if fp8_dir.exists() and list(fp8_dir.glob("blocks.*.safetensors")):
        print(f"{fp8_dir.name} already converted, skipping")
        return

    print(f"\n=== Downloading {model_subdir} (BF16 sharded) ===")
    snapshot_download(
        repo_id=REPO_ID,
        local_dir=str(WAN22_DIR),
        allow_patterns=[f"{model_subdir}/*"],
        token=TOKEN,
    )
    print(f"Download complete. Converting to FP8 split-block...")
    convert_dit(bf16_dir, fp8_dir)

    print(f"Deleting BF16 source ({model_subdir})...")
    shutil.rmtree(str(bf16_dir))
    print(f"Done. Recovered ~40GB.")


if __name__ == "__main__":
    download_and_convert("high_noise_model")
    download_and_convert("low_noise_model")
    print("\nALL FP8 DiTs ready.")
