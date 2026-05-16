"""
MemorySanitizer — cleans up GPU/CPU memory between pipeline stages.
Spec: VGA Runtime Spec v17.2 §resilience/memory_sanitizer.py
"""
from __future__ import annotations

import gc
import logging

logger = logging.getLogger(__name__)


class MemorySanitizer:
    """Performs targeted memory cleanup between pipeline stages."""

    def sanitize_gpu(self) -> None:
        """Release all unreferenced GPU tensors."""
        try:
            import torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            logger.debug("MemorySanitizer: GPU memory sanitized")
        except Exception as exc:
            logger.warning("MemorySanitizer.sanitize_gpu failed: %s", exc)

    def sanitize_cpu(self) -> None:
        """Run Python garbage collection."""
        gc.collect()
        logger.debug("MemorySanitizer: CPU memory sanitized")

    def full_sanitize(self) -> None:
        """Full sanitization: CPU + GPU."""
        self.sanitize_cpu()
        self.sanitize_gpu()
