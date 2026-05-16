"""Identity state API routes (NEW v17.0)."""
from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from vga.config.settings import settings

router = APIRouter()


class IdentityStateResponse(BaseModel):
    scene_id: str
    is_frozen: bool
    drift_score: float
    cumulative_drift: float
    drift_history: List[float]
    threshold: float = settings.IDENTITY_CUMULATIVE_DRIFT_THRESHOLD
    schema_version: str = settings.SCHEMA_VERSION


_identity_state: dict = {}


@router.get("/{scene_id}", response_model=IdentityStateResponse)
def get_identity_state(scene_id: str) -> IdentityStateResponse:
    """Get current identity state for a scene."""
    state = _identity_state.get(scene_id)
    if not state:
        raise HTTPException(status_code=404, detail=f"No identity state for scene {scene_id}")
    return IdentityStateResponse(
        scene_id=scene_id,
        is_frozen=state.get("is_frozen", False),
        drift_score=state.get("drift_score", 0.0),
        cumulative_drift=state.get("cumulative_drift", 0.0),
        drift_history=state.get("drift_history", []),
    )


@router.post("/{scene_id}/update")
def update_identity_state(scene_id: str, state: dict) -> dict:
    """Internal: update identity state (called by pipeline)."""
    _identity_state[scene_id] = state
    return {"updated": True}
