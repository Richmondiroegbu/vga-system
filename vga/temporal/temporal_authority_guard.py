"""
TemporalAuthorityGuard — runtime enforcement of TemporalEngine's exclusive authority.
Raises ArchitectureGuardViolationError on unauthorized segment iteration, buffer update,
or SVI invocation. CGRL-101, v17.2.
Spec: VGA File Responsibility Spec v17.2 §11.6
"""
from __future__ import annotations

import inspect
import logging

from vga.core.exceptions import ArchitectureGuardViolationError

logger = logging.getLogger(__name__)

# Only these callers are authorized to invoke each guarded operation
_AUTHORIZED_SEGMENT_ITERATORS = {"vga.temporal.temporal_engine"}
_AUTHORIZED_SVI_INVOKERS = {"vga.temporal.temporal_engine"}
_AUTHORIZED_BUFFER_UPDATERS = {
    "vga.temporal.temporal_buffer_manager",
    "vga.temporal.temporal_engine",
    "vga.agents.video_segment_generator",   # init only (Segment_1)
}


class TemporalAuthorityGuard:
    """Runtime gate enforcing TemporalEngine's exclusive operational authority.

    Call at the top of guarded functions before any work is done.
    """

    @staticmethod
    def guard_segment_iteration(caller_qualname: str | None = None) -> None:
        """Assert that only TemporalEngine may control segment iteration. CGRL-101."""
        caller = caller_qualname or _get_caller_module()
        authorized = any(auth in caller for auth in _AUTHORIZED_SEGMENT_ITERATORS)
        if not authorized:
            raise ArchitectureGuardViolationError(
                f"Unauthorized segment iteration from '{caller}'. "
                f"Only TemporalEngine may control the autoregressive loop. "
                f"CGRL-101, v17.2."
            )
        logger.debug("TemporalAuthorityGuard: segment iteration authorized for %s", caller)

    @staticmethod
    def guard_svi_invoke(caller_qualname: str | None = None) -> None:
        """Assert that only TemporalEngine may invoke SVI inference. CGRL-101."""
        caller = caller_qualname or _get_caller_module()
        authorized = any(auth in caller for auth in _AUTHORIZED_SVI_INVOKERS)
        if not authorized:
            raise ArchitectureGuardViolationError(
                f"Unauthorized SVI invocation from '{caller}'. "
                f"Only TemporalEngine may call SVI Pro 2. "
                f"CGRL-101, v17.2."
            )
        logger.debug("TemporalAuthorityGuard: SVI invocation authorized for %s", caller)

    @staticmethod
    def guard_buffer_update(caller_qualname: str | None = None) -> None:
        """Assert that only TemporalBufferManager/TemporalEngine may update the buffer. CGRL-101."""
        caller = caller_qualname or _get_caller_module()
        authorized = any(auth in caller for auth in _AUTHORIZED_BUFFER_UPDATERS)
        if not authorized:
            raise ArchitectureGuardViolationError(
                f"Unauthorized TemporalBuffer update from '{caller}'. "
                f"Only TemporalBufferManager (via TemporalEngine) may update the buffer. "
                f"CGRL-101, v17.2."
            )
        logger.debug("TemporalAuthorityGuard: buffer update authorized for %s", caller)

    @staticmethod
    def assert_authorized(operation: str, caller_qualname: str) -> None:
        """Generic authorization assertion — maps operation to the right guard."""
        match operation:
            case "segment_iteration":
                TemporalAuthorityGuard.guard_segment_iteration(caller_qualname)
            case "svi_invoke":
                TemporalAuthorityGuard.guard_svi_invoke(caller_qualname)
            case "buffer_update":
                TemporalAuthorityGuard.guard_buffer_update(caller_qualname)
            case _:
                logger.warning("TemporalAuthorityGuard: unknown operation '%s'", operation)


def _get_caller_module() -> str:
    """Inspect the call stack to determine the calling module name."""
    try:
        frame = inspect.stack()[2]
        module = frame.frame.f_globals.get("__name__", "unknown")
        return module
    except Exception:
        return "unknown"
