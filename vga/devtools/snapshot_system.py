"""
SnapshotSystem — manages v15/v16/v17 candidate snapshots for regression testing.
Spec: VGA DevTools Spec v17.2 §SnapshotSystem
"""
from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

_SNAPSHOTS_ROOT = Path("snapshots")
_VALID_VERSIONS = {"v15_baseline", "v16_candidate", "v17_candidate"}


class SnapshotSystem:
    """Captures and compares pipeline output snapshots across versions."""

    def save(self, version: str, job_id: str, artifacts: dict) -> Path:
        """Save a snapshot of pipeline artifacts for a specific version.

        Args:
            version:   one of v15_baseline, v16_candidate, v17_candidate
            job_id:    job identifier
            artifacts: dict of artifact name → file path or data

        Returns:
            Path to the snapshot directory
        """
        if version not in _VALID_VERSIONS:
            raise ValueError(f"Invalid snapshot version: {version!r}. Must be one of {_VALID_VERSIONS}")

        snap_dir = _SNAPSHOTS_ROOT / version / job_id
        snap_dir.mkdir(parents=True, exist_ok=True)

        manifest = {
            "job_id": job_id,
            "version": version,
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "artifacts": {},
        }

        for name, value in artifacts.items():
            if isinstance(value, (str, Path)) and Path(value).exists():
                dest = snap_dir / Path(value).name
                shutil.copy2(str(value), str(dest))
                manifest["artifacts"][name] = str(dest)
            else:
                manifest["artifacts"][name] = value

        (snap_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, default=str), encoding="utf-8"
        )
        logger.info("SnapshotSystem: saved %s snapshot for job %s at %s", version, job_id, snap_dir)
        return snap_dir

    def load(self, version: str, job_id: str) -> dict:
        """Load a snapshot manifest."""
        manifest_path = _SNAPSHOTS_ROOT / version / job_id / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"No snapshot found: {version}/{job_id}")
        return json.loads(manifest_path.read_text(encoding="utf-8"))

    def list_snapshots(self, version: str) -> list[str]:
        """List all job IDs for a given version."""
        snap_dir = _SNAPSHOTS_ROOT / version
        if not snap_dir.exists():
            return []
        return [d.name for d in snap_dir.iterdir() if d.is_dir()]
