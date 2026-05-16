"""
Shutdown — graceful shutdown coordinator for VGA pipeline.
Ensures all in-flight stages complete and VRAM is released before exit.
Spec: VGA Codebase Structure Design v17.2 §core/shutdown.py
"""
from __future__ import annotations

import gc
import logging
import signal
import sys
from typing import Callable

logger = logging.getLogger(__name__)

_shutdown_handlers: list[Callable] = []


def register_shutdown_handler(fn: Callable) -> None:
    """Register a function to be called on graceful shutdown."""
    _shutdown_handlers.append(fn)


def _graceful_shutdown(signum: int, frame: object) -> None:
    """Signal handler — runs all registered shutdown hooks then exits cleanly."""
    logger.info("VGA shutdown signal received (signal %d) — starting graceful shutdown", signum)
    for handler in reversed(_shutdown_handlers):
        try:
            handler()
        except Exception as exc:
            logger.error("Shutdown handler failed: %s", exc)
    _release_gpu()
    logger.info("VGA shutdown complete")
    sys.exit(0)


def _release_gpu() -> None:
    """Release all GPU resources."""
    try:
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        logger.info("GPU resources released")
    except Exception as exc:
        logger.warning("GPU release failed during shutdown: %s", exc)


def install_signal_handlers() -> None:
    """Install SIGTERM and SIGINT handlers for graceful shutdown."""
    signal.signal(signal.SIGTERM, _graceful_shutdown)
    signal.signal(signal.SIGINT, _graceful_shutdown)
    logger.info("VGA shutdown signal handlers installed")
