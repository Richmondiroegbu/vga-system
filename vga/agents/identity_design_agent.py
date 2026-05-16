"""
IdentityDesignAgent — Stage S-03: generates character identity and environment design.
Produces IdentityDesignSchema with reference_strategy. HRG-3 follows.
Spec: VGA Narrative Agents Spec v17.2 §S-03
"""
from __future__ import annotations

import logging
from typing import Tuple

from vga.agents.base_agent import BaseAgent
from vga.models.schemas import IdentityDesignSchema, ScriptSchema
from vga.models.wrappers.qwen_wrapper import QwenWrapper
from vga.state.immutable_context import ImmutableContext

logger = logging.getLogger(__name__)


class IdentityDesignAgent(BaseAgent):
    """S-03: Generates character visual identity, environment, and reference strategy."""

    stage_id = "S-03"

    def __init__(self, qwen: QwenWrapper | None = None) -> None:
        self._qwen = qwen or QwenWrapper()

    def run(
        self,
        input_data: dict,
        context: ImmutableContext,
    ) -> Tuple[IdentityDesignSchema, ImmutableContext]:
        """Generate identity design from script and scene plan data.

        Args:
            input_data: dict with script (ScriptSchema) and character_id
            context:    current ImmutableContext

        Returns:
            (IdentityDesignSchema, new_context)
        """
        self._log_start(context.scene_id)

        script: ScriptSchema = input_data["script"]
        character_id: str = input_data.get("character_id", "main_character")
        scene_id = context.scene_id

        prompt = (
            f"Design the visual identity for character '{character_id}' in an inspirational "
            f"cinematic video about: {script.logline}\n\n"
            f"Return JSON with fields:\n"
            f"  job_id: str\n"
            f"  scene_id: str (use '{scene_id}')\n"
            f"  character_id: str\n"
            f"  character_identity: str (detailed visual description for FLUX image generation)\n"
            f"  environment_description: str (scene environment prompt)\n"
            f"  reference_strategy: str (how to maintain visual consistency across shots)\n"
            f"  negative_prompt: str (what to avoid in generation)\n"
            f"  schema_version: str (use 'v6.0')"
        )

        design = self._qwen.generate_structured(
            prompt=prompt,
            output_schema=IdentityDesignSchema,
        )

        new_context = context.evolve(current_stage=self.stage_id)
        self._log_complete(scene_id)
        return design, new_context
