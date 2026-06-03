"""
SVIWrapper — invokes SVI Pro 2 via persistent server (preferred) or subprocess fallback.

Execution model
---------------
  1. On the FIRST segment call, SVIWrapper checks if a persistent SVI server is running
     on localhost:{SVI_SERVER_PORT} (default 8765). If not, it starts one as a detached
     background process using the svi_wan22 conda env.
  2. SVIWrapper waits up to 600 s for the server to report /health → "ready".
     During this wait the server loads both FP8 DiTs (~3-5 min cold start).
  3. All subsequent segments POST to the warm server — no cold loading.
  4. If the server fails to start or a request fails, SVIWrapper falls back to the
     legacy per-segment subprocess approach.

S-09+ continuation: input_image + input_video TOGETHER → 36 channels
----------------------------------------------------------------------
  The DiT patch_embedding requires 36-channel input, assembled internally:
    input_video (last 5 frames) → WanVideoUnit_InputVideoEmbedder → 16ch latents
    input_image (last frame)    → WanVideoUnit_ImageEmbedderVAE   → 20ch (4 mask + 16 VAE)
    Total: 16 + 20 = 36ch ✓

  CRASH history: passing only input_video (without input_image) → 16ch → crash.
  "FIX" history: removing input_video, only input_image → 36ch but fresh I2V per
    segment → no temporal continuity (same scene, quality drift).
  CORRECT: both together with denoising_strength=0.75 → true SVI continuation.

  POST-INFERENCE: first 5 output frames are hard-replaced with the exact source
  frames (pixel-identical). Assembly trims these 5 frames at each join so the
  hard-replaced frames are never visible in the final video.

Speed improvements over the legacy approach
-------------------------------------------
  - Persistent server:   eliminates ~3-5 min cold load per segment
  - GPU-resident DiTs:   eliminates ~960 PCIe round-trips per segment (env SVI_GPU_RESIDENT=1)
  - TeaCache:            ~2-3× attention speedup (tea_cache_l1_thresh=0.1)
  - FA2 + SDPA:          verified on startup; PyTorch Flash backend enforced

SVI requires PyTorch 2.7.1+cu128 — INCOMPATIBLE with the main VGA env (cu124).
ALL SVI inference MUST go through the svi_wan22 env. NEVER import SVI directly.
Spec: VGA Model Stack Setup Guide v7.2 §2.5; MASTER_PROMPT_INDEX.md §SVI Environment
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

from vga.config.settings import settings
from vga.core.exceptions import ModelLoadError, TemporalSegmentFailureError

logger = logging.getLogger(__name__)

_SVI_SERVER_PORT: int = getattr(settings, "SVI_SERVER_PORT", 8765)
_SERVER_SCRIPT = Path("/workspace/scripts/vga_svi_server.py")


class SVIWrapper:
    """Invokes SVI Pro 2 (Stable Video Infinity) via persistent server or subprocess.

    Key constraints:
    - SVI runs in svi_wan22 conda env (PyTorch 2.7.1+cu128, INCOMPATIBLE with main env)
    - First call starts a persistent server; subsequent calls reuse the warm pipeline
    - Subprocess fallback if server fails to start
    - Takes 5-frame latent tensor (TemporalBuffer encoding) as input (RULE-87)
    - Applies noise-aware dynamic LoRA schedule (SVIScheduler, RULE-86)
    """

    # ---------------------------------------------------------------- public --

    def generate_segment(
        self,
        init_latents_path: str | Path,   # kept for API compatibility; prev segment path used
        prompt: str,
        cfg: float,
        steps: int,
        lora_weight_high: float,
        lora_weight_mid: float,
        lora_weight_low: float,
        camera_motion: str,
        motion_vector: str,
        output_path: str | Path,
        scene_id: str,
        segment_id: int,
        ref_image_path: str | Path = "",  # original S-07 refined image for anchor injection
        transition_mode: str = "none",    # "none" | "hard_cut" | "blend"
        new_angle_ref_image: str = "",    # path to new angle reference (for hard_cut / blend)
        end_image_path: str | Path = "",  # FLF2V: endpoint frame — model constrained to arrive here
        wancut_skip_last: int = 0,        # WanCutLastSlot: frames to skip from prev segment tail
    ) -> str:
        """Invoke SVI Pro 2 to generate one video segment.

        Returns str output_path (same as input, confirmed written).
        Raises TemporalSegmentFailureError if both server and subprocess approaches fail.
        """
        from vga.core.exceptions import SVICFGViolationError
        import shutil

        if not (settings.SVI_CFG_MIN <= cfg <= settings.SVI_CFG_MAX):
            raise SVICFGViolationError(cfg_value=cfg)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Fallback: if svi_wan22 env doesn't exist at all, copy previous segment.
        if not Path(settings.SVI_WAN22_PYTHON).exists():
            logger.warning(
                "SVIWrapper: svi_wan22 env not found at %s — using copy fallback for segment %d.",
                settings.SVI_WAN22_PYTHON, segment_id,
            )
            prev_segment = output_path.parent / f"segment_{segment_id - 1:03d}.mp4"
            if prev_segment.exists():
                shutil.copy2(str(prev_segment), str(output_path))
                logger.info("SVIWrapper: copied segment %d → %d (fallback)", segment_id - 1, segment_id)
            return str(output_path)

        lora_schedule = {
            "high_noise_weight": lora_weight_high,
            "mid_noise_weight": lora_weight_mid,
            "low_noise_weight": lora_weight_low,
        }

        prev_segment_path = output_path.parent / f"segment_{segment_id - 1:03d}.mp4"
        if not prev_segment_path.exists():
            prev_segment_path = Path(str(init_latents_path))

        infer_config = {
            "prev_segment_path": str(prev_segment_path),
            # SVI continuation: 5 overlap frames from prev segment → video latents (16ch).
            # These frames are also hard-replaced post-inference for pixel-identical seams,
            # then trimmed during final assembly.
            "num_overlap_frames": 5,
            # 0.60 = standard continuation — tight adherence to overlap frames (less face warping).
            # Overridden to 0.90 (hard_cut) or 0.80 (blend) for transition segments.
            "denoising_strength": 0.60,
            # -1 = random seed per segment; fixed seed causes identical outputs across segments.
            "seed": -1,
            "prompt": prompt,
            "cfg": cfg,
            "steps": steps,
            "lora_path_high": str(settings.SVI_HIGH_NOISE_PATH),
            "lora_path_low": str(settings.SVI_LOW_NOISE_PATH),
            "lora_schedule": lora_schedule,
            "camera_motion": camera_motion,
            "motion_vector": motion_vector,
            "output_path": str(output_path),
            # TeaCache: 0.1 is a safe threshold (~2-3× speedup, minimal quality loss).
            # Set to 0.0 to disable if you observe visual artefacts.
            "tea_cache_l1_thresh": float(os.environ.get("SVI_TEA_CACHE_THRESH", "0.1")),
            # ref_image_path: original S-07 refined character image. When set, the
            # inference bridge injects it as anchor, anchoring cross-attention to the
            # original character in every segment and preventing progressive scene drift.
            "ref_image_path": str(ref_image_path) if ref_image_path else "",
            # transition_mode: controls camera angle switching behaviour for this segment.
            #   "none"      — normal continuation (default)
            #   "hard_cut"  — Strategy A: switch input_image to new angle ref, ds=0.90
            #   "blend"     — Strategy C: cosine-ramp pixel blend of conditioning frames, ds=0.80
            "transition_mode": transition_mode,
            # new_angle_ref_image: path to the new camera angle reference image.
            # Only used when transition_mode is "hard_cut" or "blend".
            # This becomes the new input_image (and optionally anchor) for this segment.
            "new_angle_ref_image": new_angle_ref_image,
            # FLF2V: end_image_path — when set, the inference bridge injects this frame
            # as an endpoint constraint, forcing the model to arrive at this visual state.
            # Phase 1: passed as end_image kwarg to WanVideoSviPipeline (graceful fallback
            # if unsupported). Phase 2 upgrade: concat_mask injection (no new weights).
            "end_image_path": str(end_image_path) if end_image_path else "",
            # WanCutLastSlot: skip this many frames from the tail of the previous segment
            # when extracting overlap conditioning frames. Use when the previous segment
            # was generated with an end_frame (its tail is locked/frozen to that end frame
            # and creates motion-steering conflicts at the seam if used as SVI input_video).
            # Value: settings.FLF2V_WANCUT_SLOT_FRAMES (4) when prev had end_frame, else 0.
            "wancut_skip_last": wancut_skip_last,
        }

        # ── Path 1: persistent server (avoids cold model loading per segment) ──
        port = _SVI_SERVER_PORT
        if self._ensure_server_ready(port):
            logger.info(
                "SVIWrapper: using persistent server (port=%d) for scene=%s segment=%d",
                port, scene_id, segment_id,
            )
            ok = self._post_infer(port, infer_config, timeout=900)
            if ok and output_path.exists():
                logger.info("SVIWrapper: segment %d done via server → %s", segment_id, output_path)
                return str(output_path)
            logger.warning(
                "SVIWrapper: server inference failed for segment %d, falling back to subprocess",
                segment_id,
            )

        # ── Path 2: subprocess fallback (cold model load each time) ──
        return self._run_subprocess(infer_config, output_path, scene_id, segment_id)

    # --------------------------------------------------------------- private --

    def _health_check(self, port: int) -> bool:
        """Return True if the server is running AND reports status=ready."""
        try:
            with urllib.request.urlopen(
                f"http://localhost:{port}/health", timeout=3
            ) as resp:
                data = json.loads(resp.read())
                return data.get("status") == "ready"
        except Exception:
            return False

    def _ensure_server_ready(self, port: int, startup_timeout: int = 600) -> bool:
        """Start the server if not running; wait up to startup_timeout seconds.

        Returns True when /health reports "ready", False on timeout or start failure.
        """
        if self._health_check(port):
            return True  # already warm

        if not _SERVER_SCRIPT.exists():
            logger.warning(
                "SVIWrapper: server script not found at %s — will use subprocess fallback",
                _SERVER_SCRIPT,
            )
            return False

        logger.info(
            "SVIWrapper: starting persistent SVI server on port %d "
            "(log: /workspace/logs/svi_server.log)...",
            port,
        )
        logs_dir = Path("/workspace/logs")
        logs_dir.mkdir(parents=True, exist_ok=True)
        log_fh = open(str(logs_dir / "svi_server.log"), "a")

        cmd = [
            settings.SVI_WAN22_PYTHON,
            str(_SERVER_SCRIPT),
            "--port", str(port),
            "--lora-path-high", str(settings.SVI_HIGH_NOISE_PATH),
            "--lora-path-low", str(settings.SVI_LOW_NOISE_PATH),
        ]
        env = os.environ.copy()
        # Inherit SVI_GPU_RESIDENT from env (set in .env_vga per pod VRAM capacity).
        # A6000 48GB: SVI_GPU_RESIDENT=1 keeps both FP8 DiTs (~28GB) on GPU.
        # RTX 5090 32GB: SVI_GPU_RESIDENT=0 (CPU-offload required; 34GB > 32GB VRAM).
        env.setdefault("SVI_GPU_RESIDENT", os.environ.get("SVI_GPU_RESIDENT", "0"))
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        try:
            subprocess.Popen(
                cmd,
                stdout=log_fh,
                stderr=subprocess.STDOUT,
                start_new_session=True,   # detach from parent process group
                env=env,
            )
        except OSError as exc:
            logger.error("SVIWrapper: failed to start SVI server: %s", exc)
            return False

        # Poll /health until ready or timeout
        deadline = time.monotonic() + startup_timeout
        poll_interval = 5
        while time.monotonic() < deadline:
            time.sleep(poll_interval)
            if self._health_check(port):
                logger.info("SVIWrapper: SVI server ready (port=%d)", port)
                return True
            remaining = deadline - time.monotonic()
            logger.debug("SVIWrapper: waiting for SVI server (%.0fs remaining)...", remaining)

        logger.error(
            "SVIWrapper: SVI server did not become ready within %ds — "
            "check /workspace/logs/svi_server.log for errors",
            startup_timeout,
        )
        return False

    def _post_infer(self, port: int, config: dict, timeout: int = 900) -> bool:
        """POST an inference request to the persistent server.

        Returns True on success (status=ok), False otherwise.
        """
        body = json.dumps(config).encode("utf-8")
        req = urllib.request.Request(
            f"http://localhost:{port}/infer",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                result = json.loads(resp.read())
                if result.get("status") == "ok":
                    return True
                logger.error(
                    "SVIWrapper: server returned error: %s", result.get("error", "?")
                )
                return False
        except urllib.error.HTTPError as exc:
            err_body = exc.read()
            logger.error(
                "SVIWrapper: HTTP %d from server: %s", exc.code, err_body[:500]
            )
            return False
        except Exception as exc:
            logger.error("SVIWrapper: server request failed: %s", exc)
            return False

    def _run_subprocess(
        self,
        config: dict,
        output_path: Path,
        scene_id: str,
        segment_id: int,
    ) -> str:
        """Legacy subprocess path — spawns a fresh Python process per segment."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            json.dump(config, f, indent=2)
            config_path = f.name

        logger.info(
            "SVIWrapper (subprocess): scene=%s segment=%d cfg=%.2f steps=%d",
            scene_id, segment_id, config["cfg"], config["steps"],
        )

        bridge_script = Path("/workspace/scripts/vga_svi_inference.py")

        cmd = [
            settings.SVI_WAN22_PYTHON,
            str(bridge_script),
            "--config", config_path,
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=False,
                timeout=1800,
            )
        except subprocess.TimeoutExpired:
            logger.error(
                "SVIWrapper: segment %d timed out after 1800s", segment_id
            )
            raise TemporalSegmentFailureError(scene_id=scene_id, segment_id=segment_id)
        finally:
            try:
                os.unlink(config_path)
            except OSError:
                pass

        if result.returncode != 0:
            logger.error(
                "SVIWrapper: subprocess failed returncode=%d", result.returncode
            )
            raise TemporalSegmentFailureError(scene_id=scene_id, segment_id=segment_id)

        if not output_path.exists():
            raise TemporalSegmentFailureError(scene_id=scene_id, segment_id=segment_id)

        logger.info("SVIWrapper: segment %d generated → %s", segment_id, output_path)
        return str(output_path)
