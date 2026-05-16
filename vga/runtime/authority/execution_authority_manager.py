"""
ExecutionAuthorityManager — enforces the 6-level VGA authority hierarchy.
Verifies that only authorized components can trigger stage execution.
Spec: VGA Runtime Spec v17.2 §authority/execution_authority_manager.py; AGENT.md §Authority Model
"""
from __future__ import annotations

import logging

from vga.core.exceptions import ArchitectureGuardViolationError

logger = logging.getLogger(__name__)

# Authority level 3 — only MasterOrchestrator may execute stages
_AUTHORIZED_EXECUTORS = {
    "vga.core.master_orchestrator",
    "vga.temporal.temporal_engine",   # authorized for SVI subprocess only
}


class ExecutionAuthorityManager:
    """Enforces the VGA 6-level execution authority model.

    Level 1: System Mission (unchangeable)
    Level 2: VGA Spec Suite v17.2 (authoritative)
    Level 3: Claude Code Agent (implementation authority)
    Level 4: ArchitectureGuard (runtime enforcement) ← this class
    Level 5: SystemGuard (stage isolation)
    Level 6: HRGController (human-in-the-loop)
    """

    def assert_stage_execution_authority(self, caller_module: str) -> None:
        """Assert that the caller is authorized to execute pipeline stages. RULE-106."""
        authorized = any(auth in caller_module for auth in _AUTHORIZED_EXECUTORS)
        if not authorized:
            raise ArchitectureGuardViolationError(
                f"Unauthorized stage execution from '{caller_module}'. "
                f"Only MasterOrchestrator.execute_stage() may run pipeline stages. "
                f"RULE-106."
            )
        logger.debug("ExecutionAuthorityManager: execution authorized for %s", caller_module)

    def assert_context_authority(self, obj: object) -> None:
        """Assert that context is ImmutableContext, not a dict. RULE-108."""
        if isinstance(obj, dict):
            raise ArchitectureGuardViolationError(
                "Dict passed as context — ImmutableContext required. RULE-108."
            )
