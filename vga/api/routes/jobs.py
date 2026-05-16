"""Job lifecycle API routes — includes all AVON client watcher endpoints."""
from __future__ import annotations

import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from vga.config.settings import settings

router = APIRouter()

# In-memory job store (production uses Redis/DB)
_jobs: Dict[str, dict] = {}
# Per-job client feedback reports
_client_reports: Dict[str, list] = {}


# ─── Request / Response Models ────────────────────────────────────────────────

class JobCreateRequest(BaseModel):
    topic: str
    protagonist_description: str = ""
    theme: str = "hope and resilience"
    duration_s: float = 60.0


class JobStatusResponse(BaseModel):
    """Extended status response consumed by the AVON client watcher."""
    job_id: str
    status: str                          # queued | running | completed | degraded | failed | cancelled
    topic: str
    current_stage: Optional[str] = None
    progress_percent: float = 0.0
    health: str = "good"                 # good | degraded | critical
    warnings: list = []
    stage_outputs_available: list = []
    stage_summary: dict = {}
    identity_drift: float = 0.0
    temporal_health: float = 1.0
    system_version: str = settings.SYSTEM_VERSION
    schema_version: str = settings.SCHEMA_VERSION


class JobMetadataResponse(BaseModel):
    """Expected duration and size range for file verification."""
    job_id: str
    expected_duration_s: float
    min_size_bytes: int
    max_size_bytes: int
    expected_resolution: str = "1280x720"
    expected_codec: str = "h264"
    schema_version: str = settings.SCHEMA_VERSION


class JobChecksumResponse(BaseModel):
    """Server-side SHA-256 checksum for client integrity verification."""
    job_id: str
    sha256: str
    file_path: str
    schema_version: str = settings.SCHEMA_VERSION


class ClientReportRequest(BaseModel):
    """Structured feedback POSTed by AVON after every run."""
    job_id: str
    quality_score: float
    confidence: float
    cleanup_triggered: bool
    validation_results: dict
    recommended_adjustments: dict = {}
    run_metadata: dict = {}
    schema_version: str = settings.SCHEMA_VERSION


class ClientReportResponse(BaseModel):
    accepted: bool
    job_id: str
    message: str


# ─── CRUD Endpoints ───────────────────────────────────────────────────────────

@router.post("/", response_model=JobStatusResponse)
def create_job(request: JobCreateRequest) -> JobStatusResponse:
    """Create a new VGA pipeline job."""
    job_id = f"job_{uuid.uuid4().hex[:8]}"
    _jobs[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "topic": request.topic,
        "request": request.model_dump(),
        "current_stage": None,
        "progress_percent": 0.0,
        "health": "good",
        "warnings": [],
        "stage_outputs_available": [],
        "stage_summary": {"completed_stages": 0, "total_stages": 16, "retry_count": 0},
        "identity_drift": 0.0,
        "temporal_health": 1.0,
    }
    _client_reports[job_id] = []
    return _build_status_response(job_id)


@router.get("/", response_model=list)
def list_jobs():
    """List all jobs."""
    return list(_jobs.values())


@router.get("/{job_id}", response_model=JobStatusResponse)
def get_job(job_id: str) -> JobStatusResponse:
    """Get full job status — polled by AVON watcher loop."""
    _require_job(job_id)
    return _build_status_response(job_id)


@router.delete("/{job_id}")
def delete_job(job_id: str) -> dict:
    """Trigger server-side workspace cleanup. AVON calls this ONLY after all validations pass.
    Idempotent — 404 treated as success by the client (RULE 3)."""
    if job_id not in _jobs:
        return {"deleted": True, "job_id": job_id, "note": "already_removed"}

    job = _jobs[job_id]
    output_dir = settings.OUTPUT_DIR / job_id
    deleted_files = []

    if output_dir.exists():
        for f in output_dir.rglob("*"):
            if f.is_file():
                deleted_files.append(str(f))
                try:
                    f.unlink()
                except OSError:
                    pass
        try:
            output_dir.rmdir()
        except OSError:
            pass

    _jobs.pop(job_id, None)
    return {
        "deleted": True,
        "job_id": job_id,
        "files_removed": len(deleted_files),
    }


# ─── AVON Artifact Download Endpoints ────────────────────────────────────────

@router.get("/{job_id}/output")
def download_output(job_id: str):
    """Stream-download the final video MP4."""
    _require_job(job_id)
    video_path = _find_artifact(job_id, "final_video.mp4")
    if not video_path:
        raise HTTPException(status_code=404, detail="Final video not yet available")
    return FileResponse(str(video_path), media_type="video/mp4", filename=f"{job_id}_final.mp4")


@router.get("/{job_id}/metadata", response_model=JobMetadataResponse)
def get_job_metadata(job_id: str) -> JobMetadataResponse:
    """Return expected duration + size range for client file verification."""
    _require_job(job_id)
    job = _jobs[job_id]
    duration_s = job["request"].get("duration_s", 60.0)
    # Estimate: ~1 MB/s at typical 720p H.264 encoding
    min_bytes = int(duration_s * 500_000)    # 500 KB/s floor
    max_bytes = int(duration_s * 5_000_000)  # 5 MB/s ceiling
    return JobMetadataResponse(
        job_id=job_id,
        expected_duration_s=duration_s,
        min_size_bytes=min_bytes,
        max_size_bytes=max_bytes,
    )


@router.get("/{job_id}/checksum", response_model=JobChecksumResponse)
def get_job_checksum(job_id: str) -> JobChecksumResponse:
    """Return SHA-256 checksum of the final video for client integrity check."""
    _require_job(job_id)
    video_path = _find_artifact(job_id, "final_video.mp4")
    if not video_path:
        raise HTTPException(status_code=404, detail="Final video not yet available")
    sha256 = _compute_sha256(video_path)
    return JobChecksumResponse(job_id=job_id, sha256=sha256, file_path=str(video_path))


@router.get("/{job_id}/report")
def get_pipeline_report(job_id: str) -> dict:
    """Return pipeline_report.json for AVON pipeline audit."""
    _require_job(job_id)
    report_path = _find_artifact(job_id, "pipeline_report.json")
    if not report_path:
        raise HTTPException(status_code=404, detail="Pipeline report not yet available")
    return json.loads(report_path.read_text(encoding="utf-8"))


@router.get("/{job_id}/identity")
def get_identity_state(job_id: str) -> dict:
    """Return identity_state.json for AVON identity drift validation."""
    _require_job(job_id)
    path = _find_artifact(job_id, "identity_state.json")
    if not path:
        raise HTTPException(status_code=404, detail="Identity state not yet available")
    return json.loads(path.read_text(encoding="utf-8"))


@router.get("/{job_id}/audio")
def get_audio_validation(job_id: str) -> dict:
    """Return audio_validation.json for AVON SNR + clipping validation."""
    _require_job(job_id)
    path = _find_artifact(job_id, "audio_validation.json")
    if not path:
        raise HTTPException(status_code=404, detail="Audio validation not yet available")
    return json.loads(path.read_text(encoding="utf-8"))


@router.get("/{job_id}/composition")
def get_composition_plan(job_id: str) -> dict:
    """Return composition_plan.json for AVON composition validation."""
    _require_job(job_id)
    path = _find_artifact(job_id, "composition_plan.json")
    if not path:
        raise HTTPException(status_code=404, detail="Composition plan not yet available")
    return json.loads(path.read_text(encoding="utf-8"))


@router.get("/{job_id}/temporal")
def get_continuity_report(job_id: str) -> dict:
    """Return continuity_report.json for AVON temporal health validation."""
    _require_job(job_id)
    path = _find_artifact(job_id, "continuity_report.json")
    if not path:
        raise HTTPException(status_code=404, detail="Continuity report not yet available")
    return json.loads(path.read_text(encoding="utf-8"))


# ─── Client Feedback Endpoint ─────────────────────────────────────────────────

@router.post("/{job_id}/client_report", response_model=ClientReportResponse)
def receive_client_report(job_id: str, report: ClientReportRequest) -> ClientReportResponse:
    """Accept structured feedback from AVON after every validation run.
    Stores the report for operator review and pipeline adaptation."""
    _require_job(job_id)
    _client_reports.setdefault(job_id, []).append(report.model_dump())

    # Update job health signals from client feedback
    job = _jobs[job_id]
    job["last_client_quality"] = report.quality_score
    job["last_client_confidence"] = report.confidence
    if report.recommended_adjustments:
        job["recommended_adjustments"] = report.recommended_adjustments

    return ClientReportResponse(
        accepted=True,
        job_id=job_id,
        message=f"Feedback accepted — quality={report.quality_score:.3f} confidence={report.confidence:.3f}",
    )


@router.get("/{job_id}/client_reports")
def list_client_reports(job_id: str) -> dict:
    """List all AVON feedback reports for a job."""
    _require_job(job_id)
    return {"job_id": job_id, "reports": _client_reports.get(job_id, [])}


# ─── Internal Helpers ─────────────────────────────────────────────────────────

def _require_job(job_id: str) -> dict:
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return job


def _build_status_response(job_id: str) -> JobStatusResponse:
    job = _jobs[job_id]
    return JobStatusResponse(
        job_id=job_id,
        status=job["status"],
        topic=job.get("topic", ""),
        current_stage=job.get("current_stage"),
        progress_percent=job.get("progress_percent", 0.0),
        health=job.get("health", "good"),
        warnings=job.get("warnings", []),
        stage_outputs_available=job.get("stage_outputs_available", []),
        stage_summary=job.get("stage_summary", {}),
        identity_drift=job.get("identity_drift", 0.0),
        temporal_health=job.get("temporal_health", 1.0),
    )


def _find_artifact(job_id: str, filename: str) -> Optional[Path]:
    """Locate a job artifact under OUTPUT_DIR/{job_id}/."""
    output_dir = settings.OUTPUT_DIR / job_id
    candidates = list(output_dir.rglob(filename)) if output_dir.exists() else []
    return candidates[0] if candidates else None


def _compute_sha256(path: Path) -> str:
    """Compute SHA-256 checksum of a file."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()
