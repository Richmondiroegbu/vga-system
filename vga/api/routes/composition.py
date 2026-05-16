"""CompositionPlan CRUD API routes (NEW v17.0)."""
from __future__ import annotations

import json
from typing import Optional

from fastapi import APIRouter, HTTPException

from vga.config.settings import settings
from vga.models.schemas import CompositionPlanSchema, CompositionPlanUpdateRequest

router = APIRouter()


@router.get("/{scene_id}", response_model=CompositionPlanSchema)
def get_composition_plan(scene_id: str, job_id: str = "default") -> CompositionPlanSchema:
    """Get CompositionPlan for a scene."""
    plan_file = settings.HRG_DIR / job_id / f"composition_plan_{scene_id}.json"
    if not plan_file.exists():
        raise HTTPException(status_code=404, detail=f"No CompositionPlan for scene {scene_id}")
    try:
        data = json.loads(plan_file.read_text(encoding="utf-8"))
        return CompositionPlanSchema(**data)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.patch("/{scene_id}")
def update_composition_plan(scene_id: str, update: CompositionPlanUpdateRequest) -> dict:
    """Update CompositionPlan fields (called during HRG-4 human edit)."""
    job_id = update.scene_id or "default"
    plan_file = settings.HRG_DIR / job_id / f"composition_plan_{scene_id}.json"

    if not plan_file.exists():
        raise HTTPException(status_code=404, detail=f"No CompositionPlan for scene {scene_id}")

    try:
        data = json.loads(plan_file.read_text(encoding="utf-8"))
        for field in ["camera_angle", "camera_motion", "character_positions",
                      "focus_subject", "lighting_style", "motion_vector"]:
            value = getattr(update, field, None)
            if value is not None:
                data[field] = value
        plan_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return {"updated": True, "scene_id": scene_id}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
