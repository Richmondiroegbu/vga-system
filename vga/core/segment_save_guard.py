"""
SegmentSaveGuard — prevents writing incomplete/corrupt video segments to disk.
Validates segment before saving and enforces atomic write pattern.
Spec: VGA Codebase Structure Design v17.2 §core/segment_save_guard.py
"""
from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path

from vga.config.settings import settings
from vga.core.exceptions import OutputValidationError

logger = logging.getLogger(__name__)

_MIN_SEGMENT_SIZE_BYTES = 10_000   # 10 KB minimum — filters empty/corrupt segments


class SegmentSaveGuard:
    """Validates and atomically writes video segment files to disk.

    Uses a temp-then-rename pattern to ensure the final file is never partially written.
    """

    def save(self, segment_data: bytes | Path, dest_path: Path) -> Path:
        """Atomically save a video segment with size validation.

        Args:
            segment_data: bytes or path to source file
            dest_path:    final destination path

        Returns:
            dest_path on success

        Raises:
            OutputValidationError if segment is too small (corrupt/empty)
        """
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = dest_path.with_suffix(".tmp")

        try:
            if isinstance(segment_data, (str, Path)):
                src = Path(segment_data)
                if not src.exists():
                    raise OutputValidationError(
                        f"SegmentSaveGuard: source file not found: {src}"
                    )
                shutil.copy2(str(src), str(tmp_path))
            else:
                tmp_path.write_bytes(segment_data)

            # Validate minimum size
            size = tmp_path.stat().st_size
            if size < _MIN_SEGMENT_SIZE_BYTES:
                tmp_path.unlink(missing_ok=True)
                raise OutputValidationError(
                    f"SegmentSaveGuard: segment too small ({size} bytes < {_MIN_SEGMENT_SIZE_BYTES}) "
                    f"— likely corrupt or empty"
                )

            # Atomic rename
            tmp_path.rename(dest_path)
            logger.debug("SegmentSaveGuard: saved %s (%d bytes)", dest_path.name, size)
            return dest_path

        except OutputValidationError:
            raise
        except Exception as exc:
            tmp_path.unlink(missing_ok=True)
            raise OutputValidationError(
                f"SegmentSaveGuard: failed to save segment: {exc}"
            ) from exc
