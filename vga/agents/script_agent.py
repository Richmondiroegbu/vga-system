"""
ScriptAgent — Stage S-01: generates the narrative script.
Calls Qwen2.5-14B to produce a structured ScriptSchema. HRG-1 follows.
Spec: VGA Narrative Agents Spec v17.2 §S-01
"""
from __future__ import annotations

import logging
from typing import Any, Tuple

from vga.agents.base_agent import BaseAgent
from vga.models.schemas import ScriptSchema
from vga.models.wrappers.qwen_wrapper import QwenWrapper
from vga.state.immutable_context import ImmutableContext

logger = logging.getLogger(__name__)

_SCRIPT_SYSTEM_PROMPT = """You are a professional screenwriter for VGA cinematic motivation videos.
Write an inspirational story about a person overcoming adversity.
Mission: inspire audiences by telling stories of people who overcame adversity — restoring hope and faith.
Return ONLY valid JSON matching the ScriptSchema structure."""


class ScriptAgent(BaseAgent):
    """S-01: generates narrative script from a creative brief."""

    stage_id = "S-01"

    def __init__(self, qwen: QwenWrapper | None = None) -> None:
        self._qwen = qwen or QwenWrapper()

    def run(
        self,
        input_data: dict,
        context: ImmutableContext,
    ) -> Tuple[ScriptSchema, ImmutableContext]:
        """Generate script from creative brief.

        Args:
            input_data: dict with keys: topic, protagonist_description, theme, duration_s
            context:    current ImmutableContext

        Returns:
            (ScriptSchema, new_context)
        """
        self._log_start(context.scene_id)

        topic = input_data.get("topic", "overcoming adversity")
        protagonist = input_data.get("protagonist_description", "")
        theme = input_data.get("theme", "hope and resilience")
        duration_s = input_data.get("duration_s", 60.0)

        prompt = (
            f"Write an inspirational script about: {topic}\n"
            f"Protagonist: {protagonist}\n"
            f"Theme: {theme}\n"
            f"Target duration: {duration_s} seconds\n"
            f"Output as JSON with fields: job_id, title, logline, characters, scenes, "
            f"total_duration_estimate_s, schema_version"
        )

        script = self._qwen.generate_structured(
            prompt=prompt,
            output_schema=ScriptSchema,
            system_prompt=_SCRIPT_SYSTEM_PROMPT,
        )

        new_context = context.evolve(current_stage=self.stage_id)
        self._log_complete(context.scene_id)
        return script, new_context
