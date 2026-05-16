"""
Report API routes — pipeline report download and client feedback endpoints.
Spec: VGA FastAPI Layer Spec v17.2 §routes/report.py
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from vga.config.settings import settings

router = APIRouter()
_reports: Dict[str, dict] = {}


class ReportStoreRequest(BaseModel):
    job_id: str
    scene_id: str
    report: dict


@router.get("/{job_id}/{scene_id}")
def get_report(job_id: str, scene_id: str) -> dict:
    """Get the pipeline report for a completed job/scene."""
    key = f"{job_id}:{scene_id}"
    if key in _reports:
        return _reports[key]

    # Try disk
    report_path = settings.OUTPUT_DIR / job_id / scene_id / "pipeline_report.json"
    if report_path.exists():
        try:
            return json.loads(report_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    raise HTTPException(status_code=404, detail=f"No report for {job_id}/{scene_id}")


@router.post("/store")
def store_report(request: ReportStoreRequest) -> dict:
    """Store a pipeline report (called internally by QualityAgent)."""
    key = f"{request.job_id}:{request.scene_id}"
    _reports[key] = request.report
    return {"stored": True, "job_id": request.job_id}
