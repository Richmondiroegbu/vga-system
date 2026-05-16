# Prompt 20: DevTools — Architecture Linter & Pre-commit Hooks
**Category:** DevTools  
**Files:**
- `vga/devtools/architecture_linter.py`
- `vga/devtools/rule_checker.py`
- `vga/devtools/snapshot_system.py`
- `.pre-commit-config.yaml` (update)
**Spec:** `10_VGA_Coding_Standards_and_Rules_v17.2.md`, v15.0 devtools

## ArchitectureLinter
```python
class ArchitectureLinter:
    """
    Detects architectural violations at commit time (NEVER at runtime).
    
    Checks:
    1. No dict context usage (context["key"] patterns)
    2. No direct agent.run() calls (must use execute_stage())
    3. No batch SVI generation
    4. No hardcoded threshold values (must use settings.)
    5. No forbidden imports (modelscope, FLUX.1-schnell, etc.)
    6. No TemporalBuffer construction outside TemporalBufferManager
    7. schema_version = "v6.0" in all new schema classes
    8. Every new agent file has stage_id class attribute
    """
    
    def check_all(self, source_root: str = "vga/") -> list[LintViolation]:
        violations = []
        for py_file in Path(source_root).rglob("*.py"):
            violations.extend(self.check_file(py_file))
        return violations
    
    def check_file(self, path: Path) -> list[LintViolation]:
        """Run all checks on a single file."""
        source = path.read_text()
        violations = []
        
        # Check 1: Dict context usage
        if re.search(r'context\[["\']\w+["\']', source):
            violations.append(LintViolation(path, "RULE-108", "Dict context forbidden"))
        
        # Check 2: Direct agent.run() calls
        if re.search(r'(?<!execute_stage\()agent\.run\(', source):
            violations.append(LintViolation(path, "RULE-106", "Direct agent.run() forbidden"))
        
        # Check 5: Forbidden imports
        for forbidden in ["modelscope", "FLUX.1-schnell", "wav2lip", "Wav2Lip"]:
            if forbidden.lower() in source.lower():
                violations.append(LintViolation(path, "REMOVED-MODEL", f"Forbidden: {forbidden}"))
        
        return violations
```

## pre-commit hook
```yaml
# .pre-commit-config.yaml addition:
- repo: local
  hooks:
    - id: vga-architecture-linter
      name: VGA Architecture Linter
      entry: python -m vga.devtools.architecture_linter --check-all
      language: system
      types: [python]
      fail_fast: true
```

## Acceptance Criteria
- [ ] `python -m vga.devtools.architecture_linter --check-all` catches dict context patterns
- [ ] Linter detects direct `agent.run()` calls
- [ ] Linter is in `vga/devtools/` with ZERO runtime imports (dev-only)
- [ ] Pre-commit hook runs linter on every commit
