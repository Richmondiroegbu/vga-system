"""
AssemblyAgent — Stage S-16a: merges all video segments and audio into final_video.mp4.
Uses ffmpeg for AV concatenation.
Spec: VGA Export Quality Spec v17.2 §S-16a
"""
from __future__ import annotations

import logging
import subprocess
import tempfile
from pathlib import Path
from typing import List, Tuple

from vga.agents.base_agent import BaseAgent
from vga.config.settings import settings
from vga.state.immutable_context import ImmutableContext

logger = logging.getLogger(__name__)


class AssemblyAgent(BaseAgent):
    """S-16a: concatenates video segments and merges audio into final_video.mp4."""

    stage_id = "S-16"

    def run(
        self,
        input_data: dict,
        context: ImmutableContext,
    ) -> Tuple[dict, ImmutableContext]:
        self._log_start(context.scene_id)

        video_paths: List[str] = input_data.get("video_paths", [])
        audio_path: str = input_data.get("audio_path", "")
        output_dir = Path(input_data.get("output_dir", settings.OUTPUT_DIR / context.job_id))
        output_dir.mkdir(parents=True, exist_ok=True)
        final_path = output_dir / "final_video.mp4"

        # Concatenate video segments
        concat_path = self._concat_videos(video_paths, output_dir)

        # Merge audio
        self._merge_audio(concat_path, audio_path, final_path)

        output = {
            "final_video_path": str(final_path),
            "scene_id": context.scene_id,
            "schema_version": settings.SCHEMA_VERSION,
        }
        new_context = context.evolve(current_stage=self.stage_id)
        self._log_complete(context.scene_id)
        return output, new_context

    @staticmethod
    def _concat_videos(paths: List[str], output_dir: Path) -> Path:
        """Concatenate video files using ffmpeg concat demuxer."""
        if not paths:
            concat_path = output_dir / "concat.mp4"
            return concat_path

        list_file = output_dir / "concat_list.txt"
        list_file.write_text(
            "\n".join(f"file '{p}'" for p in paths if Path(p).exists()),
            encoding="utf-8",
        )
        concat_path = output_dir / "concat.mp4"
        cmd = [
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", str(list_file), "-c", "copy", str(concat_path),
        ]
        try:
            subprocess.run(cmd, capture_output=True, check=True, timeout=300)
        except (subprocess.CalledProcessError, FileNotFoundError) as exc:
            logger.warning("AssemblyAgent: ffmpeg concat failed: %s", exc)
        return concat_path

    @staticmethod
    def _merge_audio(video_path: Path, audio_path: str, output_path: Path) -> None:
        """Merge audio track into video file using ffmpeg."""
        if not video_path.exists() or not Path(audio_path).exists():
            if video_path.exists():
                video_path.rename(output_path)
            return
        cmd = [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-i", audio_path,
            "-c:v", "copy",
            "-c:a", "aac",
            "-shortest",
            str(output_path),
        ]
        try:
            subprocess.run(cmd, capture_output=True, check=True, timeout=300)
        except (subprocess.CalledProcessError, FileNotFoundError) as exc:
            logger.warning("AssemblyAgent: ffmpeg merge failed: %s", exc)
