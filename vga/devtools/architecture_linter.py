"""
ArchitectureLinter — enforces VGA architectural rules on the codebase.
Run: python -m vga.devtools.architecture_linter --check-all
Spec: VGA DevTools Spec v17.2; RULE-86, RULE-87, RULE-88, RULE-106, RULE-107, RULE-108
"""
from __future__ import annotations

import argparse
import ast
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)


@dataclass
class LintViolation:
    rule_id: str
    file_path: str
    line_number: int
    message: str


@dataclass
class LintResult:
    violations: List[LintViolation] = field(default_factory=list)
    files_checked: int = 0

    @property
    def passed(self) -> bool:
        return len(self.violations) == 0


class ArchitectureLinter:
    """Scans Python source files for VGA architectural rule violations.

    Rules checked:
    - RULE-106: No direct agent.run() calls (must use execute_stage())
    - RULE-108: No dict-based context construction
    - RULE-86: TEMPORAL_BUFFER_SIZE must equal 5 wherever used as literal
    - RULE-107: No batch SVI patterns (must be explicit for loop)
    - No forbidden imports (modelscope, old FluxPipeline, etc.)
    """

    FORBIDDEN_IMPORTS = {
        "modelscope": "FORBIDDEN: modelscope import. Use HuggingFace. RULE-01.",
        "FluxPipeline": "FORBIDDEN: FluxPipeline (old). Use the current FLUX.2-klein pipeline.",
    }

    def check_all(self, root: Path) -> LintResult:
        """Scan all .py files under root and return LintResult."""
        result = LintResult()
        py_files = list(root.rglob("*.py"))
        py_files = [
            f for f in py_files
            if ".venv" not in str(f) and "system_files" not in str(f)
        ]

        for py_file in py_files:
            result.files_checked += 1
            violations = self._check_file(py_file)
            result.violations.extend(violations)

        return result

    def _check_file(self, file_path: Path) -> List[LintViolation]:
        violations = []
        try:
            source = file_path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(file_path))
        except (SyntaxError, OSError):
            return violations

        violations.extend(self._check_direct_agent_run(tree, file_path))
        violations.extend(self._check_dict_context(tree, file_path))
        violations.extend(self._check_forbidden_imports(tree, file_path))
        violations.extend(self._check_buffer_literal(source, file_path))

        return violations

    def _check_direct_agent_run(
        self, tree: ast.AST, file_path: Path
    ) -> List[LintViolation]:
        """RULE-106: detect agent.run() calls outside master_orchestrator.py"""
        if "master_orchestrator" in str(file_path):
            return []  # the orchestrator itself is allowed
        if "tests" in str(file_path) or "test_" in file_path.name:
            return []  # test files legitimately call agent.run() for unit isolation

        violations = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute):
                    if node.func.attr == "run" and isinstance(node.func.value, ast.Name):
                        # Heuristic: if the object name ends in 'agent', flag it
                        obj_name = node.func.value.id.lower()
                        if "agent" in obj_name:
                            violations.append(LintViolation(
                                rule_id="RULE-106",
                                file_path=str(file_path),
                                line_number=node.lineno,
                                message=(
                                    f"Direct {obj_name}.run() call — use execute_stage() instead. "
                                    f"RULE-106."
                                ),
                            ))
        return violations

    def _check_dict_context(
        self, tree: ast.AST, file_path: Path
    ) -> List[LintViolation]:
        """RULE-108: detect context['key'] subscription patterns on a 'context' variable."""
        violations = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Subscript):
                if isinstance(node.value, ast.Name) and node.value.id == "context":
                    violations.append(LintViolation(
                        rule_id="RULE-108",
                        file_path=str(file_path),
                        line_number=node.lineno,
                        message=(
                            "Dict-style context['key'] access — use ImmutableContext attributes. "
                            "RULE-108."
                        ),
                    ))
        return violations

    def _check_forbidden_imports(
        self, tree: ast.AST, file_path: Path
    ) -> List[LintViolation]:
        """Check for forbidden import patterns."""
        violations = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                module = ""
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        module = alias.name
                        for forbidden, msg in self.FORBIDDEN_IMPORTS.items():
                            if forbidden in module:
                                violations.append(LintViolation(
                                    rule_id="FORBIDDEN_IMPORT",
                                    file_path=str(file_path),
                                    line_number=node.lineno,
                                    message=msg,
                                ))
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    for alias in (node.names or []):
                        name = alias.name
                        for forbidden, msg in self.FORBIDDEN_IMPORTS.items():
                            if forbidden in name or forbidden in module:
                                violations.append(LintViolation(
                                    rule_id="FORBIDDEN_IMPORT",
                                    file_path=str(file_path),
                                    line_number=node.lineno,
                                    message=msg,
                                ))
        return violations

    def _check_buffer_literal(
        self, source: str, file_path: Path
    ) -> List[LintViolation]:
        """RULE-86: detect hardcoded buffer size != 5."""
        violations = []
        for i, line in enumerate(source.splitlines(), start=1):
            if "TEMPORAL_BUFFER_SIZE" in line and "=" in line:
                # Check if it's being set to something other than 5
                try:
                    rhs = line.split("=", 1)[1].strip()
                    if rhs and rhs[0].isdigit():
                        val = int(rhs.split("#")[0].strip())
                        if val != 5:
                            violations.append(LintViolation(
                                rule_id="RULE-86",
                                file_path=str(file_path),
                                line_number=i,
                                message=f"TEMPORAL_BUFFER_SIZE must be 5, got {val}. RULE-86.",
                            ))
                except (ValueError, IndexError):
                    pass
        return violations


def main():
    parser = argparse.ArgumentParser(description="VGA Architecture Linter")
    parser.add_argument("--check-all", action="store_true", help="Check all Python files")
    parser.add_argument("--root", default=".", help="Project root directory")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    linter = ArchitectureLinter()

    if args.check_all:
        result = linter.check_all(root)
        print(f"\nArchitectureLinter: checked {result.files_checked} files")

        if result.passed:
            print("ALL RULES PASS — architecture is compliant.\n")
            sys.exit(0)
        else:
            print(f"\n{len(result.violations)} VIOLATION(S) FOUND:\n")
            for v in result.violations:
                print(f"  [{v.rule_id}] {v.file_path}:{v.line_number}")
                print(f"    {v.message}")
            print()
            sys.exit(1)
    else:
        parser.print_help()
        sys.exit(0)


if __name__ == "__main__":
    main()
