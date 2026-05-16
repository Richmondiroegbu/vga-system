"""
VRAMGuard — VRAM headroom assertions before model loads.
Spec: VGA Model Manager Spec v17.2 §4; VRAM Sequential Contract
"""
from __future__ import annotations

import logging

from vga.config.settings import settings
from vga.core.exceptions import VRAMViolationError

logger = logging.getLogger(__name__)


class VRAMGuard:
    """Asserts VRAM headroom before model loading."""

    @staticmethod
    def assert_headroom(required_ratio: float | None = None) -> None:
        """Raise VRAMViolationError if free VRAM ratio is below threshold.

        Args:
            required_ratio: minimum free ratio (defaults to settings.VRAM_FREE_RATIO_MIN)
        """
        threshold = required_ratio or settings.VRAM_FREE_RATIO_MIN
        free_ratio = VRAMGuard.get_free_ratio()

        if free_ratio < threshold:
            raise VRAMViolationError(
                f"Insufficient VRAM: {free_ratio:.1%} free, need {threshold:.1%}. "
                f"Call ModelManager.unload_all() before loading next model."
            )
        logger.debug("VRAMGuard: %.1f%% free — headroom OK", free_ratio * 100)

    @staticmethod
    def get_free_ratio() -> float:
        """Return fraction of free VRAM (0.0–1.0). Returns 1.0 if CUDA unavailable."""
        try:
            import torch
            if not torch.cuda.is_available():
                return 1.0
            free, total = torch.cuda.mem_get_info()
            return free / total if total > 0 else 1.0
        except Exception:
            return 1.0

    @staticmethod
    def get_free_gb() -> float:
        """Return free VRAM in gigabytes."""
        try:
            import torch
            if not torch.cuda.is_available():
                return 24.0   # default: assume RTX 4090
            free, _ = torch.cuda.mem_get_info()
            return free / (1024 ** 3)
        except Exception:
            return 24.0
