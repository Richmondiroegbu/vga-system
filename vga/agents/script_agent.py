"""
ScriptAgent — Stage S-01: generates the narrative script.
Calls Qwen3-14B to produce a structured ScriptSchema. HRG-1 follows.
Spec: VGA Narrative Agents Spec v17.2 §S-01

Action Density Rule (motion evaluation requirement):
  The first 20 seconds of Scene 1 MUST contain ≥4 distinct, coherent, specific
  physical action beats. This is the motion quality evaluation window — it determines
  how natural or drifted character movement appears vs. real human motion.
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
Mission: inspire audiences by telling stories of people who overcame adversity — restoring hope and faith.
Return ONLY valid JSON. No markdown, no explanation, no code blocks.

═══════════════════════════════════════════════════════════════════
CRITICAL: ACTION DENSITY RULE — FIRST 20 SECONDS
═══════════════════════════════════════════════════════════════════
Scene 1 must open with ≥4 DISTINCT, COHERENT, SPECIFIC physical action beats
packed into the first ~20 seconds (4 segments × 5 seconds each).

WHY: The first 20 seconds is the motion quality evaluation window. It determines
whether the AI-generated character movement looks natural like a real human or
drifts into unnatural, robotic, or frozen poses. Dense, varied, coherent actions
stress-test the full range of motion generation.

WHAT COUNTS AS A GOOD ACTION BEAT:
Each action beat must specify ALL THREE of:
  1. BODY PART engaged (full body / upper body / lower body / head+face / hands)
  2. SPECIFIC MOVEMENT (not "walks" → "strides forward with shoulders rolling,
     arms swinging loosely at her sides, heels striking first")
  3. EMOTIONAL STATE visible in the movement (not "sad" → "jaw clenched, eyes
     fixed on the ground, each step heavier than the last")

WHAT IS FORBIDDEN IN ACTION BEATS:
  ✗ Generic: "walks", "stands", "looks", "sits"
  ✗ Static: "stands still looking at camera"
  ✗ Invisible: "thinks about her past" (not observable in motion)
  ✗ Disconnected: action 2 has no physical relationship to action 1

WHAT IS REQUIRED — COHERENT FLOW:
Each action must physically lead into the next, as if you are choreographing
a single continuous scene of human motion:
  ✓ Beat 1 → Beat 2 → Beat 3 → Beat 4 must read as one unbroken movement sequence
  ✓ A person stopping must transition from running/walking, not appear from nothing
  ✓ A gesture must relate to what the character just did or will do next

REQUIRED ACTION VARIETY IN FIRST 20 SECONDS:
Include at least 3 of these 5 categories across the 4 opening beats:
  [A] LOCOMOTION: walking, running, stumbling, climbing, stopping abruptly
  [B] UPPER BODY: reaching, lifting, pressing, throwing, gripping, releasing
  [C] POSTURE SHIFT: crouching, standing, kneeling, leaning, turning
  [D] HEAD/FACE: looking up/down/sideways, nodding, shaking, eyes widening
  [E] WHOLE-BODY EMOTION: slumping defeat, straightening resolve, trembling fear

EXAMPLES — GOOD vs BAD:

BAD (too generic, no motion detail):
"She walks into the room and sits down, looking sad."

GOOD (specific, evaluable, coherent):
"She pushes open the heavy gym door with her shoulder [B+C], steps inside and
stops dead when she sees the empty equipment — her hands drop to her sides,
shoulders collapsing forward [C+E]; she exhales, raises her chin slowly [D],
and forces one foot forward, then the other, jaw tight with resolve [A+E]."

─────────────────────────────────────────────────────────────────
Include the 4 opening action beats in the scene's opening_action_sequence field:
  "opening_action_sequence": [
    "Beat 1 — full specific description of seconds 0–5",
    "Beat 2 — full specific description of seconds 5–10",
    "Beat 3 — full specific description of seconds 10–15",
    "Beat 4 — full specific description of seconds 15–20"
  ]
═══════════════════════════════════════════════════════════════════"""


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
            f"\n"
            f"CRITICAL — characters field MUST be a list of objects, NOT strings.\n"
            f'CORRECT:   "characters": [{{"name": "Maya", "role": "protagonist", "age_range": "late 20s", "appearance": "...", "emotional_arc": "...", "description": "..."}}]\n'
            f'WRONG:     "characters": ["Maya (protagonist)"]\n'
            f"\n"
            f"MANDATORY — ACTION DENSITY FOR MOTION EVALUATION:\n"
            f"Scene 1 must open with exactly 4 specific, coherent physical action beats\n"
            f"in the first 20 seconds. Fill opening_action_sequence with these 4 beats.\n"
            f"Each beat: [BODY PART] + [EXACT MOVEMENT] + [EMOTIONAL STATE in body].\n"
            f"The 4 beats must flow as one unbroken physical sequence — each leads into the next.\n"
            f"Cover at least 3 of: locomotion, upper-body gesture, posture shift, head/face, whole-body emotion.\n"
            f"\n"
            f"OPENING ACTION SEQUENCE FORMAT EXAMPLE (adapt to your story):\n"
            f'  "opening_action_sequence": [\n'
            f'    "She sprints down the empty track [A locomotion], arms pistoning, '
            f'breath ragged — the starting blocks shrinking behind her",\n'
            f'    "She slows at the 200m mark, leg buckling slightly [C posture+A], '
            f'grabs her thigh with both hands [B upper-body], teeth gritted",\n'
            f'    "She forces herself upright [C posture], lifts her chin [D head], '
            f'stares at the finish line ahead — eyes narrowing with calculation",\n'
            f'    "She takes one deliberate stride [A], then another, arms rising into '
            f'race form [B+E], the hesitation replaced by controlled fury"\n'
            f'  ]\n'
            f"\n"
            f"Output JSON fields: job_id, title, logline, characters, scenes "
            f"(each scene has: scene_id, scene_number, title, description, emotional_tone, "
            f"duration_hint_s, opening_action_sequence), "
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
