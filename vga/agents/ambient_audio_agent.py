"""
AmbientAudioAgent — Stage S-13: generates ambient soundscapes via MMAudio.
Spec: VGA Audio Pipeline Spec v17.2 §S-13
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple

from vga.agents.base_agent import BaseAgent
from vga.config.settings import settings
from vga.models.wrappers.mmaudio_wrapper import MMAudioWrapper
from vga.state.immutable_context import ImmutableContext

logger = logging.getLogger(__name__)


class AmbientAudioAgent(BaseAgent):
    """S-13: generates per-scene ambient audio synchronized to video."""

    stage_id = "S-13"

    def __init__(self, mmaudio: MMAudioWrapper | None = None) -> None:
        self._mmaudio = mmaudio or MMAudioWrapper()

    def run(
        self,
        input_data: dict,
        context: ImmutableContext,
    ) -> Tuple[dict, ImmutableContext]:
        self._log_start(context.scene_id)

        video_paths: List[str] = input_data.get("video_paths", [])
        ambient_prompt: str = input_data.get("ambient_prompt", "subtle ambient environment sound")
        output_dir = Path(input_data.get("output_dir", settings.OUTPUT_DIR / context.job_id / "ambient"))
        output_dir.mkdir(parents=True, exist_ok=True)

        ambient_paths = []
        for i, video_path in enumerate(video_paths):
            out_path = output_dir / f"ambient_{i+1:03d}.wav"
            duration_s = input_data.get("durations", [4.0] * len(video_paths))[i]
            audio_path = self._mmaudio.generate(
                video_path=video_path,
                prompt=ambient_prompt,
                duration_s=duration_s,
                output_path=out_path,
            )
            ambient_paths.append(audio_path)

        output = {
            "ambient_paths": ambient_paths,
            "scene_id": context.scene_id,
            "schema_version": settings.SCHEMA_VERSION,
        }
        new_context = context.evolve(current_stage=self.stage_id)
        self._log_complete(context.scene_id)
        return output, new_context
