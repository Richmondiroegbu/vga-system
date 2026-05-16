"""
ModelManager — singleton controlling ALL model lifecycle operations.
Enforces sequential VRAM usage: ONE heavy model loaded at a time.
Spec: VGA System Architecture v17.2 §6 (Model Lifecycle); FR-200–FR-215
"""
from __future__ import annotations

import gc
import logging
import time
from typing import Any, Optional

from vga.config.settings import settings
from vga.core.exceptions import ModelLoadError, ModelUnloadError, VRAMViolationError

logger = logging.getLogger(__name__)

# Sentinel for "no model loaded"
_NO_MODEL = "__none__"


class ModelManager:
    """Singleton that controls model loading and VRAM lifecycle.

    Rules enforced:
    - Only ONE heavy model may be in VRAM at a time (VRAMViolationError otherwise)
    - Smart reuse: if the requested model is already loaded, skip reload
    - torch.load() always uses map_location="cpu" then .to(device)
    - Unload sequence: del → gc.collect() → torch.cuda.empty_cache() → sleep(2)
    """

    _instance: Optional["ModelManager"] = None

    def __new__(cls) -> "ModelManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self._current_model_name: str = _NO_MODEL
        self._current_model: Any = None
        self._initialized = True
        logger.info("ModelManager initialized — VRAM sequential contract active")

    # ─── Public API ───────────────────────────────────────────────────────────

    def load(self, model_name: str, loader_fn: Any, *args: Any, **kwargs: Any) -> Any:
        """Load a model using loader_fn.

        If model_name is already loaded, returns cached model (smart reuse).
        If a DIFFERENT model is loaded, raises VRAMViolationError — caller must
        call unload_all() first.

        loader_fn signature: loader_fn(*args, **kwargs) → model
        All internal torch.load() calls must use map_location="cpu".
        """
        if self._current_model_name == model_name:
            logger.info("ModelManager: smart reuse — %s already in VRAM", model_name)
            return self._current_model

        if self._current_model_name != _NO_MODEL:
            raise VRAMViolationError(
                f"Cannot load '{model_name}' while '{self._current_model_name}' is in VRAM. "
                f"Call unload_all() first. VRAM sequential contract violation."
            )

        self._assert_vram_free()

        logger.info("ModelManager: loading model '%s'", model_name)
        try:
            model = loader_fn(*args, **kwargs)
        except Exception as exc:
            raise ModelLoadError(
                f"Failed to load model '{model_name}': {exc}"
            ) from exc

        self._current_model_name = model_name
        self._current_model = model
        logger.info("ModelManager: '%s' loaded successfully", model_name)
        return model

    def unload_all(self) -> None:
        """Unload current model and release VRAM.

        Sequence: del → gc.collect() → torch.cuda.empty_cache() → sleep(2)
        """
        if self._current_model_name == _NO_MODEL:
            logger.debug("ModelManager: unload_all called but no model loaded")
            return

        model_name = self._current_model_name
        logger.info("ModelManager: unloading '%s'", model_name)

        try:
            del self._current_model
            self._current_model = None
            self._current_model_name = _NO_MODEL
            gc.collect()
            self._cuda_empty_cache()
            time.sleep(2)
        except Exception as exc:
            raise ModelUnloadError(
                f"Failed to unload model '{model_name}': {exc}"
            ) from exc

        logger.info("ModelManager: VRAM cleared after unloading '%s'", model_name)

    @property
    def current_model_name(self) -> str:
        """Name of the currently loaded model, or '__none__' if empty."""
        return self._current_model_name

    @property
    def is_empty(self) -> bool:
        """True when no model is currently loaded."""
        return self._current_model_name == _NO_MODEL

    def get_free_vram_ratio(self) -> float:
        """Return fraction of free VRAM (0.0–1.0). Returns 1.0 if CUDA unavailable."""
        try:
            import torch
            if not torch.cuda.is_available():
                return 1.0
            free, total = torch.cuda.mem_get_info()
            return free / total if total > 0 else 1.0
        except Exception:
            return 1.0

    # ─── Private Helpers ──────────────────────────────────────────────────────

    def _assert_vram_free(self) -> None:
        """Raise if free VRAM ratio is below minimum threshold."""
        ratio = self.get_free_vram_ratio()
        min_ratio = settings.VRAM_FREE_RATIO_MIN
        if ratio < min_ratio:
            raise VRAMViolationError(
                f"Insufficient free VRAM: {ratio:.1%} available, "
                f"minimum {min_ratio:.1%} required. Call unload_all() first."
            )

    def _cuda_empty_cache(self) -> None:
        """Call torch.cuda.empty_cache() if CUDA is available."""
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
