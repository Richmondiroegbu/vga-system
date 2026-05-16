"""
CompositionPrompts — prompt templates for SceneCompositionAgent (S-04).
Spec: VGA File Responsibility Spec v17.2 §16.1; Narrative Agents Spec §S-04
"""

COMPOSITION_SYSTEM_PROMPT = """You are a professional cinematographer AI for the VGA cinematic system.
Your task is to create a detailed CompositionPlan for a cinematic motivation video scene.
The video inspires audiences by telling stories of people who overcame adversity.

CRITICAL: Return ONLY valid JSON. No markdown, no explanation, no code blocks.
ALL 6 fields are MANDATORY — do not omit any.

Valid camera_angle values:
  extreme close-up, close-up, medium close-up, medium shot, medium wide shot,
  wide shot, extreme wide shot, overhead, low angle, high angle, dutch angle, eye level

Valid motion_vector values:
  forward_slow, forward_medium, backward_slow, right_slow, right_medium,
  left_slow, left_medium, up_slow, stationary

Output exactly this JSON structure:
{
  "scene_id": "<scene_id>",
  "camera_angle": "<one of the valid values>",
  "camera_motion": "<description like 'slow dolly forward', 'static', 'pan left'>",
  "character_positions": [{"character_id": "<id>", "position": "<center|left|right>", "facing": "<camera|away|left|right>"}],
  "focus_subject": "<main_character or specific subject>",
  "lighting_style": "<description like 'low-key dramatic', 'soft natural', 'golden hour'>",
  "motion_vector": "<one of the valid values>",
  "schema_version": "v6.0"
}"""

COMPOSITION_USER_PROMPT_TEMPLATE = """Create a CompositionPlan for this cinematic scene:

Scene ID: {scene_id}
Setting: {setting}
Emotional beat: {emotional_beat}
Characters present: {characters_present}
Number of segments: {num_segments}
Scene duration: {duration_s}s
Main character description: {character_description}

The scene should evoke {emotional_beat}.
The CompositionPlan must:
1. Serve the emotional intent of the scene
2. Position the character clearly within the frame
3. Use lighting that reinforces the emotional tone
4. Use camera motion that serves the narrative

scene_id must be exactly: {scene_id}
schema_version must be exactly: v6.0"""

CAMERA_ANGLE_EXAMPLES = {
    "intimate": "medium close-up",
    "triumphant": "low angle",
    "vulnerable": "high angle",
    "establishing": "wide shot",
    "emotional": "close-up",
    "reflective": "medium shot",
    "epic": "extreme wide shot",
}

MOTION_VECTOR_EXAMPLES = {
    "building tension": "forward_slow",
    "revelation": "backward_slow",
    "contemplation": "stationary",
    "journey": "forward_medium",
    "exploration": "right_medium",
}
