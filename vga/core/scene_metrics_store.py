"""
SceneMetricsStore — persists per-scene quality metrics for observability and adaptation.
Spec: VGA Codebase Structure Design v17.2 §core/scene_metrics_store.py
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from vga.config.settings import settings

logger = logging.getLogger(__name__)


class SceneMetricsStore:
    """Stores and retrieves per-scene quality metrics (CLIP scores, SNR, continuity)."""

    def __init__(self) -> None:
        self._metrics_dir = settings.STATE_DIR / "metrics"
        self._metrics_dir.mkdir(parents=True, exist_ok=True)
        self._cache: Dict[str, list] = {}

    def record(
        self,
        job_id: str,
        scene_id: str,
        stage_id: str,
        metrics: Dict[str, Any],
    ) -> None:
        """Record metrics for a stage within a scene."""
        key = f"{job_id}:{scene_id}"
        entry = {"stage_id": stage_id, **metrics}
        self._cache.setdefault(key, []).append(entry)

        metrics_path = self._metrics_dir / job_id / f"{scene_id}.jsonl"
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with metrics_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, default=str) + "\n")
        except Exception as exc:
            logger.warning("SceneMetricsStore.record failed: %s", exc)

    def get_scene_metrics(self, job_id: str, scene_id: str) -> List[dict]:
        """Retrieve all metrics for a scene."""
        key = f"{job_id}:{scene_id}"
        if key in self._cache:
            return self._cache[key]

        metrics_path = self._metrics_dir / job_id / f"{scene_id}.jsonl"
        if not metrics_path.exists():
            return []
        try:
            lines = metrics_path.read_text(encoding="utf-8").splitlines()
            return [json.loads(l) for l in lines if l.strip()]
        except Exception:
            return []

    def get_clip_series(self, job_id: str, scene_id: str) -> List[float]:
        """Extract CLIP score series across stages for a scene."""
        return [
            m["clip_score"]
            for m in self.get_scene_metrics(job_id, scene_id)
            if "clip_score" in m
        ]
