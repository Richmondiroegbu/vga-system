"""
StateSyncService — synchronises ImmutableContext state to disk for observability.
Allows the Streamlit UI and API to read current pipeline state without coupling.
Spec: VGA Codebase Structure Design v17.2 §core/state_sync_service.py
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from vga.config.settings import settings
from vga.state.immutable_context import ImmutableContext

logger = logging.getLogger(__name__)


class StateSyncService:
    """Writes ImmutableContext snapshots to disk so the API/UI can poll state."""

    def __init__(self) -> None:
        self._state_dir = settings.STATE_DIR / "context"
        self._state_dir.mkdir(parents=True, exist_ok=True)

    def sync(self, context: ImmutableContext) -> None:
        """Write a JSON snapshot of the current context state."""
        job_id = context.job_id
        scene_id = context.scene_id
        state_file = self._state_dir / job_id / f"{scene_id}.json"
        state_file.parent.mkdir(parents=True, exist_ok=True)

        snapshot = {
            "job_id": job_id,
            "scene_id": scene_id,
            "current_stage": context.current_stage,
            "completed_stages": list(context.completed_stages),
            "identity_state": {
                "is_frozen": context.identity_state.is_frozen,
                "drift_score": context.identity_state.drift_score,
                "cumulative_drift": context.identity_state.cumulative_drift,
            },
            "temporal_state": {
                "segment_index": context.temporal_state.segment_index,
                "total_segments": context.temporal_state.total_segments,
                "buffer_initialized": context.temporal_state.buffer_initialized,
            },
            "composition_plan": (
                context.composition_plan.model_dump()
                if context.composition_plan else None
            ),
            "schema_version": settings.SCHEMA_VERSION,
        }

        try:
            state_file.write_text(
                json.dumps(snapshot, indent=2, default=str), encoding="utf-8"
            )
        except Exception as exc:
            logger.warning("StateSyncService.sync failed: %s", exc)

    def read(self, job_id: str, scene_id: str) -> dict | None:
        """Read the last synced context snapshot."""
        state_file = self._state_dir / job_id / f"{scene_id}.json"
        if not state_file.exists():
            return None
        try:
            return json.loads(state_file.read_text(encoding="utf-8"))
        except Exception:
            return None
