"""Unit tests for CompositionValidator. RULE-88."""
from __future__ import annotations

import pytest

from vga.core.exceptions import CompositionPlanValidationError
from vga.models.schemas import CompositionPlanSchema
from vga.state.context_factory import ContextFactory
from vga.validation.composition_validator import CompositionValidator


def test_assert_in_context_raises_when_no_plan(valid_context):
    """assert_in_context raises CompositionPlanValidationError when plan is None."""
    validator = CompositionValidator()
    with pytest.raises(CompositionPlanValidationError):
        validator.assert_in_context(valid_context)


def test_assert_in_context_passes_with_valid_plan(valid_context):
    """assert_in_context succeeds when all 6 required fields are present."""
    plan = CompositionPlanSchema(
        scene_id="sc_001",
        camera_angle="medium shot",
        camera_motion="static",
        character_positions=[{"character_id": "hero", "position": "center"}],
        focus_subject="main_character",
        lighting_style="soft natural",
        motion_vector="stationary",
    )
    ctx_with_plan = valid_context.evolve(composition_plan=plan)
    validator = CompositionValidator()
    result = validator.assert_in_context(ctx_with_plan)
    assert result.scene_id == "sc_001"


def test_validate_schema_raises_on_invalid_angle():
    """validate_schema raises CompositionPlanValidationError for bad camera_angle."""
    validator = CompositionValidator()
    with pytest.raises(CompositionPlanValidationError):
        validator.validate_schema(
            {
                "camera_angle": "bad_angle",
                "camera_motion": "static",
                "character_positions": [{"character_id": "hero"}],
                "focus_subject": "hero",
                "lighting_style": "natural",
                "motion_vector": "stationary",
            },
            scene_id="sc_001",
        )
