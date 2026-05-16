"""
VGA FastAPI Application — main entry point.
All routes registered here. Bootstrap runs at startup.
Spec: VGA FastAPI Layer Spec v17.2; FR-700–FR-740
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from vga.api.routes import jobs, hrg, temporal, identity, audio, composition, report
from vga.api.middleware.auth import AuthMiddleware
from vga.api.middleware.logging import LoggingMiddleware


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: run VGA bootstrap to initialize all singletons."""
    try:
        from vga.bootstrap import run_bootstrap
        registry = run_bootstrap()
        app.state.registry = registry
    except Exception as exc:
        import logging
        logging.getLogger(__name__).error("VGA bootstrap failed: %s", exc)
        app.state.registry = {}
    yield
    # Shutdown: nothing to clean up currently


app = FastAPI(
    title="VGA API",
    description="Video Generation Automation v17.2 — Cinematic AI Motivation System",
    version="17.2.0",
    lifespan=lifespan,
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.add_middleware(LoggingMiddleware)
app.add_middleware(AuthMiddleware)

# Register all route modules
app.include_router(jobs.router, prefix="/api/v1/jobs", tags=["jobs"])
app.include_router(hrg.router, prefix="/api/v1/hrg", tags=["hrg"])
app.include_router(temporal.router, prefix="/api/v1/temporal", tags=["temporal"])
app.include_router(identity.router, prefix="/api/v1/identity", tags=["identity"])
app.include_router(audio.router, prefix="/api/v1/audio", tags=["audio"])
app.include_router(composition.router, prefix="/api/v1/composition", tags=["composition"])
app.include_router(report.router, prefix="/api/v1/reports", tags=["reports"])


@app.get("/health")
def health_check():
    return {"status": "ok", "version": "17.2.0"}
