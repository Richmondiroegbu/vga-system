"""
DevtoolsValidator — validates that devtools infrastructure itself is correctly set up.
Spec: VGA DevTools Spec v17.2 §validator.py
"""
from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class DevtoolsValidator:
    """Validates the devtools setup and pre-commit configuration."""

    def validate_all(self, project_root: Path) -> dict:
        """Run all devtools validation checks.

        Returns:
            dict with 'passed' (bool) and 'issues' (list of strings)
        """
        issues = []

        # Check 1: Architecture linter exists
        linter_path = project_root / "vga" / "devtools" / "architecture_linter.py"
        if not linter_path.exists():
            issues.append("architecture_linter.py missing")

        # Check 2: Pre-commit config exists
        precommit = project_root / ".pre-commit-config.yaml"
        if not precommit.exists():
            issues.append(".pre-commit-config.yaml missing")
        else:
            content = precommit.read_text(encoding="utf-8")
            if "architecture_linter" not in content:
                issues.append(".pre-commit-config.yaml does not reference architecture linter")

        # Check 3: DEVIATION_LOG.md exists
        deviation_log = project_root / "DEVIATION_LOG.md"
        if not deviation_log.exists():
            issues.append("DEVIATION_LOG.md missing")

        # Check 4: snapshots directories exist
        for version in ("v15_baseline", "v16_candidate", "v17_candidate"):
            snap_dir = project_root / "snapshots" / version
            if not snap_dir.exists():
                issues.append(f"snapshots/{version}/ directory missing")

        passed = not issues
        if passed:
            logger.info("DevtoolsValidator: all checks passed")
        else:
            logger.warning("DevtoolsValidator: %d issue(s): %s", len(issues), issues)

        return {"passed": passed, "issues": issues}
