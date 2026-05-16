"""
Tracer — structured pipeline event tracing.
All stage events logged as JSON with consistent fields.
Spec: VGA Coding Standards RULE-18 (no print), RULE-19 (structured events)
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from vga.config.settings import settings

logger = logging.getLogger(__name__)


class Tracer:
    """Emits structured trace events for every pipeline stage transition."""

    def __init__(self) -> None:
        self._events: list[dict] = []
        logger.info("Tracer initialized")

    def trace_event(
        self,
        event_name: str,
        stage_id: str | None = None,
        scene_id: str | None = None,
        job_id: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Record a pipeline trace event."""
        event = {
            "event": event_name,
            "timestamp": time.time(),
            "stage_id": stage_id,
            "scene_id": scene_id,
            "job_id": job_id,
            **kwargs,
        }
        self._events.append(event)
        logger.debug("TRACE %s stage=%s scene=%s", event_name, stage_id, scene_id)

    def flush_to_disk(self, job_id: str) -> None:
        """Write all events to /workspace/logs/trace_{job_id}.jsonl."""
        try:
            log_dir = settings.LOGS_DIR
            log_dir.mkdir(parents=True, exist_ok=True)
            out = log_dir / f"trace_{job_id}.jsonl"
            with out.open("a", encoding="utf-8") as f:
                for ev in self._events:
                    f.write(json.dumps(ev, default=str) + "\n")
            self._events.clear()
        except Exception as exc:
            logger.warning("Tracer flush failed: %s", exc)
