"""
OutputIntegrityChecker — verifies stage outputs are not None/empty/corrupt.
Spec: VGA Runtime Spec v17.2 §failure/output_integrity_checker.py
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from vga.core.exceptions import OutputValidationError

logger = logging.getLogger(__name__)


class OutputIntegrityChecker:
    """Validates that stage outputs pass basic integrity checks."""

    def check(self, stage_id: str, output: Any) -> None:
        """Assert output is not None and passes basic type checks.

        Raises:
            OutputValidationError on any integrity failure
        """
        if output is None:
            raise OutputValidationError(
                f"Stage {stage_id} returned None — output contract violated",
                stage_id=stage_id,
            )

        # For dict outputs: must not be empty
        if isinstance(output, dict) and not output:
            raise OutputValidationError(
                f"Stage {stage_id} returned empty dict",
                stage_id=stage_id,
            )

        # For path outputs: file must exist
        if isinstance(output, (str, Path)):
            path = Path(output)
            if not path.exists():
                raise OutputValidationError(
                    f"Stage {stage_id} output path does not exist: {path}",
                    stage_id=stage_id,
                )

        logger.debug("OutputIntegrityChecker: %s output OK", stage_id)

    def check_schema_version(self, stage_id: str, output: Any) -> None:
        """Assert schema_version = 'v6.0' on artifact outputs."""
        if hasattr(output, "schema_version"):
            if output.schema_version != "v6.0":
                raise OutputValidationError(
                    f"Stage {stage_id}: schema_version={output.schema_version!r} "
                    f"(expected 'v6.0')",
                    stage_id=stage_id,
                )
