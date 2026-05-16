"""
MusicAgent — Stage S-14: generates background music via MusicGen-medium.
Spec: VGA Audio Pipeline Spec v17.2 §S-14
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

from vga.agents.base_agent import BaseAgent
from vga.config.settings import settings
from vga.models.wrappers.musicgen_wrapper import MusicGenWrapper
from vga.state.immutable_context import ImmutableContext

logger = logging.getLogger(__name__)


class MusicAgent(BaseAgent):
    """S-14: generates inspirational background music for the scene."""

    stage_id = "S-14"

    def __init__(self, musicgen: MusicGenWrapper | None = None) -> None:
        self._musicgen = musicgen or MusicGenWrapper()

    def run(
        self,
        input_data: dict,
        context: ImmutableContext,
    ) -> Tuple[dict, ImmutableContext]:
        self._log_start(context.scene_id)

        music_prompt: str = input_data.get(
            "music_prompt",
            "emotional orchestral music, inspiring, uplifting, cinematic",
        )
        duration_s: float = input_data.get("duration_s", 30.0)
        output_dir = Path(input_data.get("output_dir", settings.OUTPUT_DIR / context.job_id / "music"))
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / f"music_{context.scene_id}.wav"

        music_path = self._musicgen.generate(
            prompt=music_prompt,
            duration_s=duration_s,
            output_path=out_path,
        )

        output = {
            "music_path": music_path,
            "duration_s": duration_s,
            "scene_id": context.scene_id,
            "schema_version": settings.SCHEMA_VERSION,
        }
        new_context = context.evolve(current_stage=self.stage_id)
        self._log_complete(context.scene_id)
        return output, new_context
