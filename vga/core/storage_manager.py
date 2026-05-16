"""
StorageManager — path resolution for all workspace outputs.
Single source of truth for output directory layout.
Spec: VGA File Responsibility Spec v17.2 §Storage
"""
from __future__ import annotations

import logging
from pathlib import Path

from vga.config.settings import settings

logger = logging.getLogger(__name__)


class StorageManager:
    """Resolves workspace paths for jobs, scenes, and assets."""

    @staticmethod
    def job_dir(job_id: str) -> Path:
        path = settings.OUTPUT_DIR / job_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    @staticmethod
    def scene_dir(job_id: str, scene_id: str) -> Path:
        path = settings.OUTPUT_DIR / job_id / scene_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    @staticmethod
    def segment_path(job_id: str, scene_id: str, segment_number: int) -> Path:
        return StorageManager.scene_dir(job_id, scene_id) / f"segment_{segment_number:03d}.mp4"

    @staticmethod
    def hrg_dir(job_id: str) -> Path:
        path = settings.HRG_DIR / job_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    @staticmethod
    def log_dir() -> Path:
        settings.LOGS_DIR.mkdir(parents=True, exist_ok=True)
        return settings.LOGS_DIR
