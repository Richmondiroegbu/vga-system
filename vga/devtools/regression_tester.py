"""
RegressionTester — runs regression checks against snapshot baselines.
Compares current pipeline outputs against v15/v16/v17 snapshots.
Spec: VGA DevTools Spec v17.2 §RegressionTester
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)

_SNAPSHOTS = Path("snapshots")


@dataclass
class RegressionResult:
    passed: bool
    version: str
    job_id: str
    mismatches: List[str] = field(default_factory=list)


class RegressionTester:
    """Compares pipeline outputs against stored snapshots to detect regressions."""

    def compare_to_baseline(
        self,
        current_report: dict,
        baseline_version: str = "v16_candidate",
        job_id: str = "baseline",
    ) -> RegressionResult:
        """Compare current pipeline report against a stored baseline."""
        baseline_path = _SNAPSHOTS / baseline_version / job_id / "manifest.json"
        if not baseline_path.exists():
            logger.warning("RegressionTester: no baseline at %s", baseline_path)
            return RegressionResult(passed=True, version=baseline_version, job_id=job_id)

        try:
            baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.error("RegressionTester: failed to load baseline: %s", exc)
            return RegressionResult(passed=False, version=baseline_version, job_id=job_id,
                                    mismatches=[str(exc)])

        mismatches = []

        # Compare schema versions
        b_schema = baseline.get("artifacts", {}).get("schema_version", "unknown")
        c_schema = current_report.get("schema_version", "unknown")
        if b_schema != c_schema:
            mismatches.append(f"schema_version: baseline={b_schema} current={c_schema}")

        # Compare HRG checkpoint count
        b_hrg = baseline.get("artifacts", {}).get("hrg_checkpoint_count", 0)
        c_hrg = current_report.get("hrg_checkpoint_count", 0)
        if b_hrg != c_hrg:
            mismatches.append(f"hrg_checkpoint_count: baseline={b_hrg} current={c_hrg}")

        passed = not mismatches
        if not passed:
            logger.warning("RegressionTester: %d mismatch(es) vs %s", len(mismatches), baseline_version)

        return RegressionResult(
            passed=passed,
            version=baseline_version,
            job_id=job_id,
            mismatches=mismatches,
        )
