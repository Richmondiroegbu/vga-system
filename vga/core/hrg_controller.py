"""
HRGController — manages all 11 Human Review Gate checkpoints.
Blocks pipeline execution until human approves or rejects.
Spec: VGA System Architecture v17.2 §8 (HRG Subsystem); FR-300–FR-340
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional

from vga.config.settings import settings
from vga.core.exceptions import HRGRejectionError, HRGTimeoutError
from vga.models.enums import HRGCheckpoint
from vga.models.schemas import (
    HRG4DisplayData,
    HRG8DisplayData,
    HRG10DisplayData,
    HRG11DisplayData,
)

logger = logging.getLogger(__name__)


class HRGController:
    """Singleton managing 11 Human Review Gate checkpoints.

    Each checkpoint:
    1. Writes display data to /workspace/hrg/{checkpoint_id}.json
    2. Blocks until the API receives an approval or rejection response
    3. On approval → pipeline continues
    4. On rejection → HRGRejectionError raised (pipeline regenerates)
    5. On timeout → HRGTimeoutError raised
    """

    _instance: Optional["HRGController"] = None

    def __new__(cls) -> "HRGController":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self._hrg_dir = settings.HRG_DIR
        self._timeout = settings.HRG_APPROVAL_TIMEOUT_SECONDS
        self._enabled = settings.HRG_REVIEW_ENABLED
        self._outcomes: Dict[str, str] = {}
        self._initialized = True
        logger.info(
            "HRGController initialized — %d checkpoints, timeout=%ds, enabled=%s",
            settings.HRG_CHECKPOINT_COUNT, self._timeout, self._enabled,
        )

    # ─── Public API ───────────────────────────────────────────────────────────

    def checkpoint(
        self,
        checkpoint: HRGCheckpoint,
        display_data: Any,
        scene_id: str,
        job_id: str,
    ) -> None:
        """Execute an HRG checkpoint.

        Writes display_data to disk, then blocks for human approval.
        Raises HRGRejectionError on rejection, HRGTimeoutError on timeout.
        """
        logger.info("HRG checkpoint %s — scene=%s job=%s", checkpoint.value, scene_id, job_id)

        if not self._enabled:
            logger.warning("HRG disabled — auto-approving %s", checkpoint.value)
            self._outcomes[checkpoint.value] = "auto_approved"
            return

        self._write_display_data(checkpoint, display_data, scene_id, job_id)
        self._wait_for_approval(checkpoint, scene_id)

    def get_outcomes(self) -> Dict[str, str]:
        """Return dict of checkpoint_id → outcome for pipeline report."""
        return dict(self._outcomes)

    # ─── Internal ─────────────────────────────────────────────────────────────

    def _write_display_data(
        self,
        checkpoint: HRGCheckpoint,
        display_data: Any,
        scene_id: str,
        job_id: str,
    ) -> None:
        """Serialize display_data to /workspace/hrg/{checkpoint}_{scene_id}.json."""
        try:
            self._hrg_dir.mkdir(parents=True, exist_ok=True)
            filename = f"{checkpoint.value}_{scene_id}.json"
            output_path = self._hrg_dir / filename

            if hasattr(display_data, "model_dump"):
                payload = display_data.model_dump()
            else:
                payload = display_data if isinstance(display_data, dict) else str(display_data)

            envelope = {
                "checkpoint": checkpoint.value,
                "scene_id": scene_id,
                "job_id": job_id,
                "status": "pending",
                "display_data": payload,
            }
            output_path.write_text(json.dumps(envelope, indent=2, default=str), encoding="utf-8")
            logger.info("HRG display data written to %s", output_path)
        except Exception as exc:
            logger.error("HRG failed to write display data: %s", exc)

    def _wait_for_approval(self, checkpoint: HRGCheckpoint, scene_id: str) -> None:
        """Poll for human approval response. Block up to timeout seconds."""
        approval_file = self._hrg_dir / f"{checkpoint.value}_{scene_id}_response.json"
        deadline = time.monotonic() + self._timeout

        while time.monotonic() < deadline:
            if approval_file.exists():
                try:
                    response = json.loads(approval_file.read_text(encoding="utf-8"))
                    decision = response.get("decision", "").lower()
                    approval_file.unlink(missing_ok=True)

                    if decision == "approved":
                        self._outcomes[checkpoint.value] = "approved"
                        logger.info("HRG %s APPROVED for scene %s", checkpoint.value, scene_id)
                        return
                    elif decision == "rejected":
                        self._outcomes[checkpoint.value] = "rejected"
                        reason = response.get("reason", "No reason provided")
                        raise HRGRejectionError(
                            f"HRG {checkpoint.value} rejected for scene {scene_id}: {reason}"
                        )
                    else:
                        logger.warning("HRG response has unknown decision: %r", decision)
                except HRGRejectionError:
                    raise
                except Exception as exc:
                    logger.warning("HRG response parse error: %s", exc)

            time.sleep(2)

        raise HRGTimeoutError(
            f"HRG {checkpoint.value} timed out after {self._timeout}s waiting for "
            f"human review of scene {scene_id}"
        )

    def _prepare_hrg_display_data(
        self,
        checkpoint: HRGCheckpoint,
        raw_data: Dict[str, Any],
    ) -> Any:
        """Build typed display data for specific checkpoint types."""
        if checkpoint == HRGCheckpoint.HRG_4_COMPOSITION:
            return HRG4DisplayData(**raw_data)
        elif checkpoint == HRGCheckpoint.HRG_8_MOTION_QA:
            return HRG8DisplayData(**raw_data)
        elif checkpoint == HRGCheckpoint.HRG_10_LIPSYNC_QA:
            return HRG10DisplayData(**raw_data)
        elif checkpoint == HRGCheckpoint.HRG_11_FINAL_AUDIO_QA:
            return HRG11DisplayData(**raw_data)
        return raw_data
