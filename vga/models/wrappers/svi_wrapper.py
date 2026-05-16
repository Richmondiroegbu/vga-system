"""
SVIWrapper — invokes SVI Pro 2 via subprocess in the svi_wan22 conda environment.
SVI requires PyTorch 2.7.1+cu128, which is INCOMPATIBLE with the main VGA env (cu124).
ALL SVI inference MUST go through subprocess invocation. NEVER import SVI directly.
Spec: VGA Model Stack Setup Guide v7.2 §2.5; MASTER_PROMPT_INDEX.md §SVI Environment
"""
from __future__ import annotations

import json
import logging
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np

from vga.config.settings import settings
from vga.core.exceptions import ModelLoadError, TemporalSegmentFailureError

logger = logging.getLogger(__name__)


class SVIWrapper:
    """Invokes SVI Pro 2 (Stable Video Infinity) via subprocess.

    Key constraints:
    - SVI runs in svi_wan22 conda env (PyTorch 2.7.1+cu128, INCOMPATIBLE with main env)
    - Invoked via settings.SVI_WAN22_PYTHON (never imported directly)
    - Takes 5-frame latent tensor (TemporalBuffer encoding) as input (RULE-87)
    - Applies noise-aware dynamic LoRA schedule (SVIScheduler, RULE-86)
    """

    def generate_segment(
        self,
        init_latents_path: str | Path,
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
    ) -> str:
        """Invoke SVI Pro 2 to generate one video segment.

        Args:
            init_latents_path: path to .npy file with 5-frame encoded latents (RULE-87)
            prompt:            scene description
            cfg:               classifier-free guidance in [5.0, 6.0]
            steps:             diffusion steps (30 minimum)
            lora_weight_high:  LoRA weight for high-noise phase (0.6)
            lora_weight_mid:   LoRA weight for mid-noise phase (0.5)
            lora_weight_low:   LoRA weight for low-noise phase (0.4)
            camera_motion:     from CompositionPlan
            motion_vector:     from CompositionPlan
            output_path:       where to write the output video segment
            scene_id:          for error reporting
            segment_id:        for error reporting

        Returns:
            str output_path (same as input, confirmed written)

        Raises:
            TemporalSegmentFailureError if subprocess fails
        """
        from vga.core.exceptions import SVICFGViolationError
        if not (settings.SVI_CFG_MIN <= cfg <= settings.SVI_CFG_MAX):
            raise SVICFGViolationError(cfg_value=cfg)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        lora_schedule = {
            "high_noise_weight": lora_weight_high,
            "mid_noise_weight": lora_weight_mid,
            "low_noise_weight": lora_weight_low,
        }

        # Write inference config to a temp JSON so subprocess can read it
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            config = {
                "init_latents_path": str(init_latents_path),
                "prompt": prompt,
                "cfg": cfg,
                "steps": steps,
                "lora_path_high": str(settings.SVI_HIGH_NOISE_PATH),
                "lora_path_low": str(settings.SVI_LOW_NOISE_PATH),
                "lora_schedule": lora_schedule,
                "camera_motion": camera_motion,
                "motion_vector": motion_vector,
                "output_path": str(output_path),
                "repo_path": str(settings.SVI_REPO_PATH),
            }
            json.dump(config, f, indent=2)
            config_path = f.name

        logger.info(
            "SVIWrapper: invoking subprocess scene=%s segment=%d cfg=%.2f steps=%d",
            scene_id, segment_id, cfg, steps,
        )

        cmd = [
            settings.SVI_WAN22_PYTHON,
            str(settings.SVI_REPO_PATH / "vga_svi_inference.py"),
            "--config", config_path,
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,   # 10 minute max per segment
            )
        except subprocess.TimeoutExpired:
            raise TemporalSegmentFailureError(
                scene_id=scene_id,
                segment_id=segment_id,
            )

        if result.returncode != 0:
            logger.error(
                "SVIWrapper subprocess failed: returncode=%d stderr=%s",
                result.returncode, result.stderr[:500],
            )
            raise TemporalSegmentFailureError(
                scene_id=scene_id,
                segment_id=segment_id,
            )

        if not output_path.exists():
            raise TemporalSegmentFailureError(
                scene_id=scene_id,
                segment_id=segment_id,
            )

        logger.info(
            "SVIWrapper: segment %d generated → %s", segment_id, output_path
        )
        return str(output_path)
