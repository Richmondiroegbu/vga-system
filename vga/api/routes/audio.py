"""Audio quality API routes (NEW v17.0)."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from vga.models.schemas import AudioQualityRecord

router = APIRouter()

_audio_records: dict = {}


@router.get("/{scene_id}", response_model=AudioQualityRecord)
def get_audio_quality(scene_id: str) -> AudioQualityRecord:
    """Get audio quality record for a scene."""
    record = _audio_records.get(scene_id)
    if not record:
        raise HTTPException(status_code=404, detail=f"No audio record for scene {scene_id}")
    return AudioQualityRecord(**record)


@router.post("/{scene_id}")
def store_audio_quality(scene_id: str, record: AudioQualityRecord) -> dict:
    """Store audio quality record (called by pipeline)."""
    _audio_records[scene_id] = record.model_dump()
    return {"stored": True}
