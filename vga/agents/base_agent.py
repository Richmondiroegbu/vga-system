"""
BaseAgent — abstract base class for all VGA pipeline stage agents.
NEVER call agent.run() directly — always use execute_stage(). RULE-106.
Spec: VGA Engine Template Spec v17.2 §6; RULE-106, FR-220
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Tuple

from vga.core.exceptions import ArchitectureGuardViolationError
from vga.state.immutable_context import ImmutableContext

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """Abstract base for all VGA pipeline stage agents.

    Subclasses MUST implement run() following the mandatory 6-step pattern:
      1. Validate inputs against schema
      2. Check prerequisites (has_output())
      3. Execute with retry (≤ COMPOSITION_MAX_RETRIES)
      4. Validate outputs
      5. context.evolve() — ALL 5 dimensions updated
      6. tracer.trace_event()

    CRITICAL: run() is called ONLY by MasterOrchestrator.execute_stage().
    NEVER call agent.run() from anywhere else (ArchitectureGuardViolationError).
    """

    @property
    @abstractmethod
    def stage_id(self) -> str:
        """Return the pipeline stage identifier (e.g., 'S-01')."""

    @abstractmethod
    def run(
        self,
        input_data: Any,
        context: ImmutableContext,
    ) -> Tuple[Any, ImmutableContext]:
        """Execute the stage. Return (output, new_context).

        RULE-106: Called ONLY by MasterOrchestrator.execute_stage().
        FR-950: context.evolve() MUST be called before returning.
        """

    def _log_start(self, scene_id: str) -> None:
        logger.info("Agent %s START scene=%s", self.stage_id, scene_id)

    def _log_complete(self, scene_id: str) -> None:
        logger.info("Agent %s COMPLETE scene=%s", self.stage_id, scene_id)
