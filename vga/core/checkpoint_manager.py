"""
CheckpointManager — saves and restores pipeline stage checkpoints.
Enables crash recovery by persisting completed stage outputs to disk.
Spec: VGA Codebase Structure Design v17.2 §core/checkpoint_manager.py
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

from vga.config.settings import settings

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Persists pipeline stage outputs for crash recovery and replay."""

    def __init__(self) -> None:
        self._checkpoint_dir = settings.STATE_DIR / "checkpoints"
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save(self, job_id: str, scene_id: str, stage_id: str, data: Any) -> None:
        """Save a stage checkpoint."""
        path = self._checkpoint_path(job_id, scene_id, stage_id)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            if isinstance(data, dict):
                path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
            else:
                path.write_text(str(data), encoding="utf-8")
            logger.debug("Checkpoint saved: %s/%s/%s", job_id, scene_id, stage_id)
        except Exception as exc:
            logger.warning("CheckpointManager.save failed: %s", exc)

    def load(self, job_id: str, scene_id: str, stage_id: str) -> Optional[dict]:
        """Load a stage checkpoint. Returns None if not found."""
        path = self._checkpoint_path(job_id, scene_id, stage_id)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("CheckpointManager.load failed: %s", exc)
            return None

    def exists(self, job_id: str, scene_id: str, stage_id: str) -> bool:
        """Check if a checkpoint exists for a stage."""
        return self._checkpoint_path(job_id, scene_id, stage_id).exists()

    def clear_job(self, job_id: str) -> None:
        """Remove all checkpoints for a job after successful completion."""
        job_dir = self._checkpoint_dir / job_id
        if job_dir.exists():
            import shutil
            shutil.rmtree(str(job_dir))
            logger.info("CheckpointManager: cleared checkpoints for job %s", job_id)

    def _checkpoint_path(self, job_id: str, scene_id: str, stage_id: str) -> Path:
        return self._checkpoint_dir / job_id / scene_id / f"{stage_id}.json"
