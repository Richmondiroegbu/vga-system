"""
FluxWrapper — wraps FLUX.2-klein-4B for image generation.
S-05: base images (NO LoRA, RULE-91). S-06: with Consistance_Edit_LoRA.
Spec: VGA Model Stack Setup Guide v7.2 §2.2; RULE-88, RULE-91
"""
from __future__ import annotations

import logging
from typing import List, Optional

from vga.config.settings import settings
from vga.core.exceptions import CLIPValidationError, LoRALoadError, ModelLoadError
from vga.models.schemas import CompositionPlanSchema

logger = logging.getLogger(__name__)


class FluxWrapper:
    """Wraps FLUX.2-klein-4B for base image and identity reinforcement generation.

    RULE-91: generate_base_images() MUST NOT have any LoRA loaded.
    RULE-88: CompositionPlan is required for all generation calls.
    """

    MODEL_KEY = "flux2-klein"

    def __init__(self) -> None:
        self._pipe = None
        self._lora_loaded: Optional[str] = None
        logger.info("FluxWrapper initialized (lazy load)")

    def generate_base_images(
        self,
        prompt: str,
        composition_plan: CompositionPlanSchema,
        count: int = 6,
        seed: int = 42,
    ) -> List["PIL.Image.Image"]:
        """Generate base images. RULE-91: NO LoRA allowed. RULE-88: plan required.

        Args:
            prompt:           character identity prompt
            composition_plan: required CompositionPlan (RULE-88)
            count:            number of images to generate (default 6)
            seed:             base random seed (incremented per image)

        Returns:
            List of PIL Images
        """
        # RULE-91: assert no LoRA is loaded
        if self._lora_loaded is not None:
            raise LoRALoadError(
                f"LoRA '{self._lora_loaded}' is loaded but S-05 (base images) "
                f"MUST run without any LoRA. RULE-91."
            )

        self._ensure_loaded()
        full_prompt = self._build_composition_prompt(prompt, composition_plan)
        images = []

        for i in range(count):
            img = self._infer(full_prompt, seed=seed + i)
            images.append(img)
            logger.info("FluxWrapper: generated base image %d/%d", i + 1, count)

        return images

    def generate_with_consistency_lora(
        self,
        reference_image: "PIL.Image.Image",
        prompt: str,
        composition_plan: CompositionPlanSchema,
        lora_weight: float = 0.6,
    ) -> "PIL.Image.Image":
        """S-06: generate with Consistance_Edit_LoRA. RULE-88: plan required.

        Args:
            reference_image: input image for identity conditioning
            prompt:           prompt for generation
            composition_plan: required CompositionPlan (RULE-88)
            lora_weight:      LoRA weight in [0.4, 0.7]
        """
        if not (
            settings.FLUX_IDENTITY_LORA_WEIGHT_MIN
            <= lora_weight
            <= settings.FLUX_IDENTITY_LORA_WEIGHT_MAX
        ):
            raise LoRALoadError(
                f"LoRA weight {lora_weight} outside range "
                f"[{settings.FLUX_IDENTITY_LORA_WEIGHT_MIN}, "
                f"{settings.FLUX_IDENTITY_LORA_WEIGHT_MAX}]"
            )

        self._ensure_loaded()
        self._load_consistency_lora(lora_weight)
        full_prompt = self._build_composition_prompt(prompt, composition_plan)
        return self._infer_with_image(reference_image, full_prompt)

    def unload_lora(self) -> None:
        """Remove any loaded LoRA adapter from the pipeline."""
        if self._pipe is not None and self._lora_loaded is not None:
            try:
                self._pipe.unload_lora_weights()
            except Exception:
                pass
        self._lora_loaded = None

    def _ensure_loaded(self) -> None:
        """Lazy-load the FLUX pipeline."""
        if self._pipe is not None:
            return
        try:
            from diffusers import DiffusionPipeline
            import torch
            path = str(settings.FLUX2_MODEL_PATH)
            logger.info("FluxWrapper: loading FLUX.2-klein pipeline from %s", path)
            self._pipe = DiffusionPipeline.from_pretrained(
                path,
                torch_dtype=torch.bfloat16,
            )
            self._pipe.enable_model_cpu_offload()
            logger.info("FluxWrapper: FLUX.2-klein loaded with CPU offload")
        except Exception as exc:
            raise ModelLoadError(f"FluxWrapper failed to load: {exc}") from exc

    def _load_consistency_lora(self, weight: float) -> None:
        """Load the Consistance_Edit_LoRA for identity reinforcement."""
        if self._lora_loaded == "consistency":
            return   # already loaded
        try:
            self._pipe.load_lora_weights(
                str(settings.CONSISTENCY_LORA_PATH),
                weight_name=settings.CONSISTENCY_LORA_FILENAME,
            )
            self._pipe.fuse_lora(lora_scale=weight)
            self._lora_loaded = "consistency"
            logger.info(
                "FluxWrapper: Consistance LoRA loaded — %s (weight=%.2f)",
                settings.CONSISTENCY_LORA_FILENAME, weight,
            )
        except Exception as exc:
            raise LoRALoadError(f"FluxWrapper Consistance LoRA load failed: {exc}") from exc

    def _infer(self, prompt: str, seed: int = 42) -> "PIL.Image.Image":
        """Run FLUX inference with the given prompt."""
        import torch
        from PIL import Image
        generator = torch.Generator().manual_seed(seed)
        result = self._pipe(
            prompt=prompt,
            guidance_scale=settings.FLUX_CFG,
            num_inference_steps=settings.FLUX_STEPS,
            generator=generator,
        )
        return result.images[0]

    def _infer_with_image(
        self, image: "PIL.Image.Image", prompt: str
    ) -> "PIL.Image.Image":
        """Run FLUX inference with LoRA — Flux2KleinPipeline is T2I only.
        The Consistance LoRA handles reference-image consistency; no strength param."""
        import torch
        result = self._pipe(
            prompt=prompt,
            guidance_scale=settings.FLUX_CFG,
            num_inference_steps=settings.FLUX_STEPS,
            generator=torch.Generator().manual_seed(42),
        )
        return result.images[0]

    @staticmethod
    def _build_composition_prompt(
        base_prompt: str,
        plan: CompositionPlanSchema,
    ) -> str:
        """Build full generation prompt incorporating CompositionPlan fields."""
        return (
            f"{base_prompt}, "
            f"{plan.camera_angle} shot, "
            f"{plan.lighting_style} lighting, "
            f"character in {plan.character_positions[0].get('position', 'center') if plan.character_positions else 'center'}, "
            f"focus on {plan.focus_subject}, "
            f"cinematic quality, photorealistic"
        )
