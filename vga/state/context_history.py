"""
ContextHistory — append-only log of ImmutableContext snapshots per stage.
Used for audit trails and drift series analysis.
Spec: VGA Data Contracts v17.2 §3.3; FR-953
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import List, Optional

from vga.state.immutable_context import ImmutableContext


@dataclass
class ContextSnapshot:
    """A single point-in-time capture of an ImmutableContext at a stage boundary."""

    stage_id: str
    scene_id: str
    job_id: str
    drift_score: float
    cumulative_drift: float
    is_identity_frozen: bool
    segment_index: int
    current_stage: Optional[str]


class ContextHistory:
    """Append-only log of context snapshots across pipeline stages.

    One ContextHistory per scene. Records state after each stage completes.
    Used for audit, replay, and identity drift series extraction.
    """

    def __init__(self, job_id: str, scene_id: str) -> None:
        self.job_id = job_id
        self.scene_id = scene_id
        self._snapshots: List[ContextSnapshot] = []

    def append(self, stage_id: str, context: ImmutableContext) -> None:
        """Capture the current context state after stage_id completes."""
        snap = ContextSnapshot(
            stage_id=stage_id,
            scene_id=context.scene_id,
            job_id=context.job_id,
            drift_score=context.identity_state.drift_score,
            cumulative_drift=context.identity_state.cumulative_drift,
            is_identity_frozen=context.identity_state.is_frozen,
            segment_index=context.temporal_state.segment_index,
            current_stage=context.current_stage,
        )
        self._snapshots.append(snap)

    def get_at_stage(self, stage_id: str) -> Optional[ContextSnapshot]:
        """Return the snapshot recorded after the given stage, or None."""
        for snap in self._snapshots:
            if snap.stage_id == stage_id:
                return snap
        return None

    def get_identity_drift_series(self) -> List[float]:
        """Return the per-stage drift_score series in chronological order."""
        return [snap.drift_score for snap in self._snapshots]

    def get_cumulative_drift_series(self) -> List[float]:
        """Return the cumulative_drift series in chronological order."""
        return [snap.cumulative_drift for snap in self._snapshots]

    def to_json(self) -> str:
        """Serialize the full history as a JSON string."""
        data = {
            "job_id": self.job_id,
            "scene_id": self.scene_id,
            "snapshots": [asdict(snap) for snap in self._snapshots],
        }
        return json.dumps(data, indent=2, default=str)

    def __len__(self) -> int:
        return len(self._snapshots)
