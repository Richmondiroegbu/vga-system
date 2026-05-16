"""
DialogueAgent — Stage S-11: synthesizes dialogue audio via CosyVoice3.
Timing ±0.10s (RULE-96). HRG-9 follows.
Spec: VGA Audio Pipeline Spec v17.2 §S-11; RULE-96
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple

from vga.agents.base_agent import BaseAgent
from vga.config.settings import settings
from vga.models.schemas import ScenePlanSchema
from vga.models.wrappers.cosyvoice_wrapper import CosyVoiceWrapper
from vga.state.immutable_context import ImmutableContext

logger = logging.getLogger(__name__)


class DialogueAgent(BaseAgent):
    """S-11: generates per-segment dialogue audio aligned to video timing."""

    stage_id = "S-11"

    def __init__(self, cosyvoice: CosyVoiceWrapper | None = None) -> None:
        self._cosyvoice = cosyvoice or CosyVoiceWrapper()

    def run(
        self,
        input_data: dict,
        context: ImmutableContext,
    ) -> Tuple[dict, ImmutableContext]:
        """Generate dialogue audio for each segment.

        Args:
            input_data: dict with scene_plan (ScenePlanSchema), output_dir
            context:    current ImmutableContext

        Returns:
            (output dict with audio_paths and durations, new_context)
        """
        self._log_start(context.scene_id)

        scene_plan: ScenePlanSchema = input_data["scene_plan"]
        output_dir = Path(input_data.get("output_dir", settings.OUTPUT_DIR / context.job_id / "audio"))
        output_dir.mkdir(parents=True, exist_ok=True)

        audio_paths = []
        actual_durations = []

        for seg in scene_plan.segments:
            if not seg.dialogue:
                # Write silence for segments without dialogue
                silence_path = str(output_dir / f"{seg.segment_id}_silence.wav")
                CosyVoiceWrapper._write_silence(Path(silence_path), seg.duration_s)
                audio_paths.append(silence_path)
                actual_durations.append(seg.duration_s)
                continue

            audio_path, actual_dur = self._cosyvoice.synthesize(
                text=seg.dialogue,
                target_duration_s=seg.duration_s,
            )
            audio_paths.append(audio_path)
            actual_durations.append(actual_dur)
            logger.info(
                "DialogueAgent: segment=%s duration=%.2fs", seg.segment_id, actual_dur
            )

        output = {
            "audio_paths": audio_paths,
            "actual_durations": actual_durations,
            "scene_id": context.scene_id,
            "schema_version": settings.SCHEMA_VERSION,
        }

        new_context = context.evolve(current_stage=self.stage_id)
        self._log_complete(context.scene_id)
        return output, new_context
