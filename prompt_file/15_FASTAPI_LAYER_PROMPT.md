# Prompt 15: FastAPI API Layer
**Category:** API  
**Files:**
- `vga/api/main.py`
- `vga/api/routes/jobs.py`
- `vga/api/routes/hrg.py`
- `vga/api/routes/health.py`
- `vga/api/routes/temporal.py` [NEW v17.0]
- `vga/api/routes/identity.py` [NEW v17.0]
- `vga/api/routes/audio.py` [NEW v17.0]
- `vga/api/routes/composition.py` [NEW v17.0]
- `vga/api/middleware/auth.py`
- `vga/api/middleware/logging.py`
**Spec:** `01_VGA_SRD_v17.2.md` §7.2

## Key Endpoints

### All retained v16.0 endpoints:
- `POST /jobs` — submit new job
- `GET /jobs/{job_id}` — poll status + stage progress
- `GET /jobs/{job_id}/output` — stream final MP4
- `GET /jobs/{job_id}/report` — pipeline report
- `GET /health` — health check (returns {"status": "ok", "version": "17.2.0"})
- `POST /system/resume` — resume after HRG pause
- `GET /jobs/{job_id}/hrg/{checkpoint}` — HRG status
- `POST /jobs/{job_id}/hrg/{checkpoint}` — HRG decision (approve/edit/reject)

### NEW v17.0 endpoints:
```python
@router.get("/jobs/{job_id}/temporal/buffer")
async def get_temporal_buffer(job_id: str) -> TemporalBufferStatusResponse:
    """Current TemporalBuffer state: frame_count, segment_index, timestamp."""

@router.get("/jobs/{job_id}/identity/state")
async def get_identity_state(job_id: str) -> IdentityStateResponse:
    """IdentityState: drift_score, cumulative_drift, history."""

@router.get("/jobs/{job_id}/audio/validation")
async def get_audio_validation(job_id: str) -> AudioValidationResponse:
    """SNR and clipping validation results per scene."""

@router.get("/jobs/{job_id}/composition")
async def get_composition(job_id: str) -> CompositionPlanSchema:
    """Current CompositionPlan for the job."""

@router.post("/stage/execute")
async def execute_stage_api(request: StageExecutionRequest):
    """Full execute_stage() invocation via API (for testing/debugging)."""
```

## main.py startup
```python
@app.on_event("startup")
async def startup():
    """Initialize all singletons via vga.bootstrap.initialize_all_singletons()"""
    from vga.bootstrap import initialize_all_singletons
    initialize_all_singletons()
```

## Acceptance Criteria
- [ ] `GET /health` returns 200 with version field
- [ ] `GET /jobs/{job_id}/temporal/buffer` returns TemporalBufferStatusResponse
- [ ] `POST /jobs/{job_id}/hrg/HRG-4` accepts CompositionPlan edits
- [ ] All 11 HRG checkpoint routes registered
- [ ] API version string = "v6.5"
