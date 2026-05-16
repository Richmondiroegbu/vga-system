"""Identity design prompt templates for IdentityDesignAgent (S-03)."""

IDENTITY_SYSTEM_PROMPT = """You are a character visual design AI for the VGA cinematic system.
Create detailed, photorealistic character identity descriptions for FLUX.2-klein image generation.
The description must be specific enough that every generated image is visually consistent."""

IDENTITY_USER_PROMPT_TEMPLATE = """Design the visual identity for character '{character_id}' in this story:

Story: {logline}
Character role: {character_description}
Scenes: {num_scenes}

Return JSON with:
job_id, scene_id, character_id,
character_identity (detailed FLUX prompt for consistent visual generation),
environment_description (scene environment for FLUX),
reference_strategy (how to maintain visual consistency across shots — MANDATORY),
negative_prompt (what to avoid),
schema_version ("v6.0")

The character_identity must include: age, gender, distinctive features, clothing, expression.
The reference_strategy must explain the specific CLIP-anchoring approach."""
