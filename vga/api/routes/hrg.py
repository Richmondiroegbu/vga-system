"""HRG checkpoint API routes — all 11 HRG endpoints."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from vga.config.settings import settings

router = APIRouter()


class HRGDecision(BaseModel):
    checkpoint: str
    scene_id: str
    job_id: str
    decision: str          # "approved" or "rejected"
    reason: Optional[str] = None


class HRGStatus(BaseModel):
    checkpoint: str
    scene_id: str
    status: str            # "pending", "approved", "rejected"
    display_data: Optional[dict] = None


@router.get("/{checkpoint}/{scene_id}", response_model=HRGStatus)
def get_hrg_status(checkpoint: str, scene_id: str) -> HRGStatus:
    """Get current HRG checkpoint status and display data."""
    data_file = settings.HRG_DIR / f"{checkpoint}_{scene_id}.json"
    if not data_file.exists():
        raise HTTPException(status_code=404, detail=f"No HRG data for {checkpoint}/{scene_id}")
    try:
        envelope = json.loads(data_file.read_text(encoding="utf-8"))
        return HRGStatus(
            checkpoint=checkpoint,
            scene_id=scene_id,
            status=envelope.get("status", "pending"),
            display_data=envelope.get("display_data"),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/decision")
def submit_hrg_decision(decision: HRGDecision) -> dict:
    """Submit human review decision for an HRG checkpoint."""
    response_file = (
        settings.HRG_DIR / f"{decision.checkpoint}_{decision.scene_id}_response.json"
    )
    settings.HRG_DIR.mkdir(parents=True, exist_ok=True)
    response_file.write_text(
        json.dumps({
            "checkpoint": decision.checkpoint,
            "scene_id": decision.scene_id,
            "decision": decision.decision,
            "reason": decision.reason or "",
        }, indent=2),
        encoding="utf-8",
    )
    return {"accepted": True, "checkpoint": decision.checkpoint, "decision": decision.decision}


@router.get("/pending")
def list_pending_hrgs() -> dict:
    """List all HRG checkpoints currently awaiting review."""
    pending = []
    if settings.HRG_DIR.exists():
        for f in settings.HRG_DIR.glob("HRG-*_*.json"):
            if "_response" not in f.name:
                try:
                    data = json.loads(f.read_text(encoding="utf-8"))
                    if data.get("status") == "pending":
                        pending.append(data)
                except Exception:
                    pass
    return {"pending": pending, "count": len(pending)}
