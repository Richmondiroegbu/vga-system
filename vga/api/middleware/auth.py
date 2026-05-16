"""
Auth middleware — basic API key authentication for VGA endpoints.
Spec: VGA FastAPI Layer Spec v17.2 §middleware/auth.py
"""
from __future__ import annotations

import os

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

# Skip auth on health check and docs
_SKIP_PATHS = {"/health", "/docs", "/openapi.json", "/redoc"}
_API_KEY = os.getenv("VGA_API_KEY", "")


class AuthMiddleware(BaseHTTPMiddleware):
    """Optional API key authentication. Disabled if VGA_API_KEY is not set."""

    async def dispatch(self, request: Request, call_next):
        if not _API_KEY:
            return await call_next(request)

        if request.url.path in _SKIP_PATHS:
            return await call_next(request)

        key = request.headers.get("X-API-Key", "")
        if key != _API_KEY:
            return JSONResponse(
                status_code=401,
                content={"detail": "Invalid or missing X-API-Key header"},
            )
        return await call_next(request)
