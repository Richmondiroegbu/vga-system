"""
RuleChecker — per-rule assertion suite for CGRL-01 through CGRL-104.
Validates runtime compliance for each specific rule.
Spec: VGA DevTools Spec v17.2 §RuleChecker
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)


@dataclass
class RuleCheckResult:
    rule_id: str
    compliant: bool
    detail: str


class RuleChecker:
    """Validates specific architectural rules by inspecting source files and runtime state."""

    def check_rule_86(self, source_root: Path) -> RuleCheckResult:
        """RULE-86: TEMPORAL_BUFFER_SIZE used, never hardcoded 5 in temporal files."""
        violations = []
        for py_file in source_root.rglob("*.py"):
            if "temporal" in str(py_file) and ".venv" not in str(py_file):
                content = py_file.read_text(encoding="utf-8", errors="ignore")
                if "shape[0] == 5" in content and "TEMPORAL_BUFFER_SIZE" not in content:
                    violations.append(str(py_file))
        return RuleCheckResult(
            rule_id="RULE-86",
            compliant=not violations,
            detail=f"hardcoded buffer size in: {violations}" if violations else "OK",
        )

    def check_rule_106(self, source_root: Path) -> RuleCheckResult:
        """RULE-106: No direct agent.run() calls outside master_orchestrator."""
        violations = []
        for py_file in source_root.rglob("*.py"):
            if ".venv" in str(py_file) or "master_orchestrator" in str(py_file):
                continue
            content = py_file.read_text(encoding="utf-8", errors="ignore")
            if "agent.run(" in content.lower() or "_agent.run(" in content.lower():
                violations.append(str(py_file))
        return RuleCheckResult(
            rule_id="RULE-106",
            compliant=not violations,
            detail=f"direct agent.run() in: {violations}" if violations else "OK",
        )

    def check_rule_108(self, source_root: Path) -> RuleCheckResult:
        """RULE-108: No dict-based context access."""
        violations = []
        for py_file in source_root.rglob("*.py"):
            if ".venv" in str(py_file):
                continue
            content = py_file.read_text(encoding="utf-8", errors="ignore")
            if 'context["' in content or "context['" in content:
                violations.append(str(py_file))
        return RuleCheckResult(
            rule_id="RULE-108",
            compliant=not violations,
            detail=f"dict context access in: {violations}" if violations else "OK",
        )

    def run_all(self, source_root: Path) -> List[RuleCheckResult]:
        """Run all implemented rule checks."""
        return [
            self.check_rule_86(source_root),
            self.check_rule_106(source_root),
            self.check_rule_108(source_root),
        ]
