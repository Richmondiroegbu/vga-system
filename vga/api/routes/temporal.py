"""Temporal buffer status API routes (NEW v17.0)."""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel

from vga.config.settings import settings

router = APIRouter()


class TemporalStatusResponse(BaseModel):
    scene_id: str
    segment_index: int
    total_segments: int
    buffer_initialized: bool
    buffer_size: int = settings.TEMPORAL_BUFFER_SIZE
    schema_version: str = settings.SCHEMA_VERSION


# In-memory temporal state store (updated by pipeline)
_temporal_state: dict = {}


@router.get("/{scene_id}", response_model=TemporalStatusResponse)
def get_temporal_status(scene_id: str) -> TemporalStatusResponse:
    """Get current TemporalEngine status for a scene."""
    state = _temporal_state.get(scene_id, {})
    return TemporalStatusResponse(
        scene_id=scene_id,
        segment_index=state.get("segment_index", 0),
        total_segments=state.get("total_segments", 0),
        buffer_initialized=state.get("buffer_initialized", False),
    )


@router.post("/{scene_id}/update")
def update_temporal_status(scene_id: str, status: dict) -> dict:
    """Internal: update temporal status (called by pipeline)."""
    _temporal_state[scene_id] = status
    return {"updated": True}
