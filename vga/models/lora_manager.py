"""
LoRAManager — manages LoRA adapter loading/unloading on diffusion pipelines.
Enforces RULE-91: NO LoRA during S-05 base image generation.
Spec: VGA Model Wrappers Spec v17.2 §LoRAManager
"""
from __future__ import annotations

import logging
from typing import Optional

from vga.core.exceptions import LoRALoadError

logger = logging.getLogger(__name__)


class LoRAManager:
    """Tracks which LoRA adapters are loaded on a given diffusion pipeline.

    RULE-91: S-05 (BaseImageAgent) MUST call assert_no_lora_loaded() before generation.
    """

    def __init__(self) -> None:
        self._loaded_lora: Optional[str] = None

    def load(self, pipe: object, lora_path: str, adapter_name: str, scale: float = 0.7) -> None:
        """Load a LoRA adapter onto the pipeline."""
        try:
            pipe.load_lora_weights(lora_path, adapter_name=adapter_name)  # type: ignore
            pipe.set_adapters([adapter_name], adapter_weights=[scale])  # type: ignore
            self._loaded_lora = adapter_name
            logger.info("LoRAManager: loaded '%s' (scale=%.2f)", adapter_name, scale)
        except Exception as exc:
            raise LoRALoadError(f"Failed to load LoRA '{adapter_name}': {exc}") from exc

    def unload(self, pipe: object) -> None:
        """Unload all LoRA adapters from the pipeline."""
        if self._loaded_lora is not None:
            try:
                pipe.unload_lora_weights()  # type: ignore
            except Exception:
                pass
            logger.info("LoRAManager: unloaded '%s'", self._loaded_lora)
            self._loaded_lora = None

    def assert_no_lora_loaded(self) -> None:
        """Assert no LoRA is active. Called by BaseImageAgent (RULE-91)."""
        if self._loaded_lora is not None:
            raise LoRALoadError(
                f"LoRA '{self._loaded_lora}' is loaded but S-05 base image generation "
                f"MUST run without any LoRA. Call unload() first. RULE-91."
            )

    def any_loaded(self) -> bool:
        return self._loaded_lora is not None

    @property
    def current(self) -> Optional[str]:
        return self._loaded_lora
