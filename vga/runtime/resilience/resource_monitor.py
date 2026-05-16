"""
ResourceMonitor — monitors VRAM, RAM, and disk usage during pipeline execution.
Spec: VGA Runtime Spec v17.2 §resilience/resource_monitor.py
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ResourceSnapshot:
    vram_free_gb: float
    vram_total_gb: float
    ram_used_gb: float
    ram_total_gb: float
    disk_free_gb: float
    vram_free_ratio: float


class ResourceMonitor:
    """Monitors system resource usage and provides health signals."""

    def snapshot(self) -> ResourceSnapshot:
        """Take a current snapshot of system resources."""
        vram_free_gb = vram_total_gb = 24.0
        vram_free_ratio = 1.0

        try:
            import torch
            if torch.cuda.is_available():
                free, total = torch.cuda.mem_get_info()
                vram_free_gb = free / (1024 ** 3)
                vram_total_gb = total / (1024 ** 3)
                vram_free_ratio = free / total if total > 0 else 1.0
        except Exception:
            pass

        ram_used_gb = ram_total_gb = 0.0
        try:
            import psutil
            mem = psutil.virtual_memory()
            ram_used_gb = mem.used / (1024 ** 3)
            ram_total_gb = mem.total / (1024 ** 3)
        except Exception:
            pass

        disk_free_gb = 0.0
        try:
            import psutil
            disk = psutil.disk_usage("/workspace")
            disk_free_gb = disk.free / (1024 ** 3)
        except Exception:
            pass

        return ResourceSnapshot(
            vram_free_gb=vram_free_gb,
            vram_total_gb=vram_total_gb,
            ram_used_gb=ram_used_gb,
            ram_total_gb=ram_total_gb,
            disk_free_gb=disk_free_gb,
            vram_free_ratio=vram_free_ratio,
        )

    def is_healthy(self) -> bool:
        """Return True if resources are within safe operating limits."""
        snap = self.snapshot()
        return (
            snap.vram_free_ratio >= 0.10   # at least 10% VRAM free
            and snap.disk_free_gb >= 10.0  # at least 10 GB disk free
        )
