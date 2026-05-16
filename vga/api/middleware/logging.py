"""
Logging middleware — structured request/response logging for all API calls.
Spec: VGA FastAPI Layer Spec v17.2 §middleware/logging.py
"""
from __future__ import annotations

import logging
import time

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger("vga.api")


class LoggingMiddleware(BaseHTTPMiddleware):
    """Logs every request with method, path, status, and elapsed time."""

    async def dispatch(self, request: Request, call_next):
        start = time.monotonic()
        response = await call_next(request)
        elapsed_ms = (time.monotonic() - start) * 1000

        logger.info(
            "API %s %s → %d (%.1fms)",
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
        )
        return response
