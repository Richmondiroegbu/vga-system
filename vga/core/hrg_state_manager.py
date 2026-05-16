"""
HRGStateManager — persists HRG checkpoint decisions across pod restarts.
Enables the pipeline to resume from the correct HRG gate after interruption.
Spec: VGA Codebase Structure Design v17.2 §core/hrg_state_manager.py (v17.0: 11 checkpoints)
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Optional

from vga.config.settings import settings
from vga.models.enums import HRGCheckpoint

logger = logging.getLogger(__name__)

VALID_CHECKPOINTS = {f"HRG-{i}" for i in range(1, 12)}   # HRG-1 through HRG-11


class HRGStateManager:
    """Persists and retrieves HRG checkpoint decisions for crash recovery."""

    def __init__(self) -> None:
        self._state_dir = settings.HRG_DIR / "state"
        self._state_dir.mkdir(parents=True, exist_ok=True)

    def record_decision(
        self,
        job_id: str,
        scene_id: str,
        checkpoint: str,
        decision: str,
        reason: str = "",
    ) -> None:
        """Persist an HRG decision (approved/rejected) to disk.

        Args:
            job_id:      pipeline job ID
            scene_id:    scene being processed
            checkpoint:  HRG-1 through HRG-11
            decision:    "approved" or "rejected"
            reason:      optional rejection reason
        """
        if checkpoint not in VALID_CHECKPOINTS:
            raise ValueError(
                f"Invalid HRG checkpoint: {checkpoint!r}. Must be one of HRG-1..HRG-11"
            )

        state_file = self._state_path(job_id, scene_id, checkpoint)
        state_file.parent.mkdir(parents=True, exist_ok=True)
        state_file.write_text(
            json.dumps({
                "job_id": job_id,
                "scene_id": scene_id,
                "checkpoint": checkpoint,
                "decision": decision,
                "reason": reason,
            }, indent=2),
            encoding="utf-8",
        )
        logger.info(
            "HRGStateManager: recorded %s → %s for job=%s scene=%s",
            checkpoint, decision, job_id, scene_id,
        )

    def get_decision(
        self, job_id: str, scene_id: str, checkpoint: str
    ) -> Optional[str]:
        """Return the recorded decision for a checkpoint, or None if not yet decided."""
        state_file = self._state_path(job_id, scene_id, checkpoint)
        if not state_file.exists():
            return None
        try:
            data = json.loads(state_file.read_text(encoding="utf-8"))
            return data.get("decision")
        except Exception:
            return None

    def is_approved(self, job_id: str, scene_id: str, checkpoint: str) -> bool:
        """True if checkpoint has been approved."""
        return self.get_decision(job_id, scene_id, checkpoint) == "approved"

    def get_all_decisions(self, job_id: str, scene_id: str) -> Dict[str, str]:
        """Return all recorded HRG decisions for a scene."""
        decisions = {}
        for i in range(1, 12):
            cp = f"HRG-{i}"
            decision = self.get_decision(job_id, scene_id, cp)
            if decision is not None:
                decisions[cp] = decision
        return decisions

    def clear_scene(self, job_id: str, scene_id: str) -> None:
        """Remove all HRG state for a scene (after completion or reset)."""
        scene_dir = self._state_dir / job_id / scene_id
        if scene_dir.exists():
            import shutil
            shutil.rmtree(str(scene_dir))

    def _state_path(self, job_id: str, scene_id: str, checkpoint: str) -> Path:
        return self._state_dir / job_id / scene_id / f"{checkpoint}.json"
