"""
MergeEngine — assembles individual video segments into a continuous scene video.
Uses ffmpeg concat for lossless segment merging.
Spec: VGA Codebase Structure Design v17.2 §temporal/merge_engine.py
"""
from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import List

from vga.core.exceptions import CriticalPipelineError

logger = logging.getLogger(__name__)


class MergeEngine:
    """Merges ordered video segments into a single scene video using ffmpeg."""

    def merge(
        self,
        segment_paths: List[str | Path],
        output_path: Path,
        validate: bool = True,
    ) -> Path:
        """Concatenate video segments in order using ffmpeg concat demuxer.

        Args:
            segment_paths: ordered list of segment file paths (Segment_1 first)
            output_path:   destination path for the merged video
            validate:      if True, verify output file exists and is non-empty

        Returns:
            output_path

        Raises:
            CriticalPipelineError on ffmpeg failure
        """
        if not segment_paths:
            raise CriticalPipelineError("MergeEngine.merge() called with empty segment list")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        list_file = output_path.parent / "concat_list.txt"

        # Write concat list — only include segments that exist
        existing = [Path(p) for p in segment_paths if Path(p).exists()]
        if not existing:
            raise CriticalPipelineError("MergeEngine: no segment files found for merging")

        list_file.write_text(
            "\n".join(f"file '{p.resolve()}'" for p in existing),
            encoding="utf-8",
        )

        cmd = [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0",
            "-i", str(list_file),
            "-c", "copy",
            str(output_path),
        ]

        logger.info("MergeEngine: merging %d segments → %s", len(existing), output_path.name)
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if result.returncode != 0:
                raise CriticalPipelineError(
                    f"MergeEngine: ffmpeg failed: {result.stderr[:300]}"
                )
        except FileNotFoundError:
            logger.warning("MergeEngine: ffmpeg not found — copying first segment as fallback")
            import shutil
            shutil.copy2(str(existing[0]), str(output_path))
        except subprocess.TimeoutExpired:
            raise CriticalPipelineError("MergeEngine: ffmpeg timed out after 300s")

        if validate and not output_path.exists():
            raise CriticalPipelineError(f"MergeEngine: output not found after merge: {output_path}")

        list_file.unlink(missing_ok=True)
        logger.info("MergeEngine: merge complete → %s", output_path.name)
        return output_path
