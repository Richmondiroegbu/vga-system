"""
AuditSystem — structured audit log for all pipeline decisions.
Spec: VGA Observability Spec v17.2 §AuditSystem
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from vga.config.settings import settings

logger = logging.getLogger(__name__)


class AuditSystem:
    """Records every significant pipeline decision as a structured audit entry."""

    def __init__(self) -> None:
        self._entries: list[dict] = []

    def log(
        self,
        event: str,
        stage_id: str | None = None,
        scene_id: str | None = None,
        job_id: str | None = None,
        decision: str | None = None,
        **kwargs: Any,
    ) -> None:
        entry = {
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "event": event,
            "stage_id": stage_id,
            "scene_id": scene_id,
            "job_id": job_id,
            "decision": decision,
            **kwargs,
        }
        self._entries.append(entry)
        logger.debug("AUDIT %s: %s", event, decision or "")

    def flush(self, job_id: str) -> None:
        """Write all entries to the audit log file."""
        try:
            audit_dir = settings.LOGS_DIR
            audit_dir.mkdir(parents=True, exist_ok=True)
            audit_path = audit_dir / f"audit_{job_id}.jsonl"
            with audit_path.open("a", encoding="utf-8") as f:
                for entry in self._entries:
                    f.write(json.dumps(entry, default=str) + "\n")
            self._entries.clear()
        except Exception as exc:
            logger.warning("AuditSystem: flush failed: %s", exc)
