"""
ZImageWrapper — wraps Z-Image-Turbo for identity-preserving image refinement.
Used at S-07 (ImageRefinementAgent). Drift ≤ 0.02 per step (RULE-93).
Spec: VGA Model Stack Setup Guide v7.2 §2.3; RULE-93
"""
from __future__ import annotations

import logging

from vga.config.settings import settings
from vga.core.exceptions import ModelLoadError

logger = logging.getLogger(__name__)


class ZImageWrapper:
    """Wraps Z-Image-Turbo for image-to-image refinement with identity preservation."""

    MODEL_KEY = "z-image-turbo"

    def __init__(self) -> None:
        self._pipe = None
        logger.info("ZImageWrapper initialized (lazy load)")

    def refine(
        self,
        image: "PIL.Image.Image",
        prompt: str,
        strength: float | None = None,
        seed: int = 42,
    ) -> "PIL.Image.Image":
        """Refine an image with Z-Image-Turbo.

        Args:
            image:    input image (usually best base image from S-05)
            prompt:   refinement prompt (identity description)
            strength: denoising strength in [ZIMAGE_DENOISE_MIN, ZIMAGE_DENOISE_MAX]
            seed:     random seed

        Returns:
            Refined PIL Image
        """
        strength = strength or settings.ZIMAGE_DENOISE_MIN

        if not (settings.ZIMAGE_DENOISE_MIN <= strength <= settings.ZIMAGE_DENOISE_MAX):
            logger.warning(
                "ZImageWrapper: strength %.3f out of range [%.3f, %.3f] — clamping",
                strength, settings.ZIMAGE_DENOISE_MIN, settings.ZIMAGE_DENOISE_MAX,
            )
            strength = max(
                settings.ZIMAGE_DENOISE_MIN,
                min(strength, settings.ZIMAGE_DENOISE_MAX),
            )

        self._ensure_loaded()

        try:
            import torch
            generator = torch.Generator().manual_seed(seed)
            result = self._pipe(
                prompt=prompt,
                image=image,
                strength=strength,
                guidance_scale=settings.ZIMAGE_CFG,
                num_inference_steps=4,
                generator=generator,
            )
            return result.images[0]
        except Exception as exc:
            raise ModelLoadError(f"ZImageWrapper inference failed: {exc}") from exc

    def _ensure_loaded(self) -> None:
        """Lazy-load Z-Image-Turbo pipeline."""
        if self._pipe is not None:
            return
        try:
            from diffusers import AutoPipelineForImage2Image
            import torch
            path = str(settings.ZIMAGE_MODEL_PATH)
            logger.info("ZImageWrapper: loading from %s", path)
            self._pipe = AutoPipelineForImage2Image.from_pretrained(
                path,
                torch_dtype=torch.float16,
                variant="fp16",
            )
            self._pipe.enable_model_cpu_offload()
            logger.info("ZImageWrapper: Z-Image-Turbo loaded")
        except Exception as exc:
            raise ModelLoadError(f"ZImageWrapper failed to load: {exc}") from exc
