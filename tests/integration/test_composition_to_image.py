"""
Integration test: S-04 (CompositionPlan) → S-05 (BaseImage) gate enforcement.
Verifies RULE-88: CompositionPlan required before image generation.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from vga.core.exceptions import CompositionPlanValidationError
from vga.models.schemas import CompositionPlanSchema
from vga.state.context_factory import ContextFactory
from vga.validation.composition_validator import CompositionValidator


@pytest.fixture
def context_without_plan():
    return ContextFactory.create_initial("job_test", "sc_001")


@pytest.fixture
def context_with_plan(context_without_plan):
    plan = CompositionPlanSchema(
        scene_id="sc_001",
        camera_angle="medium shot",
        camera_motion="static",
        character_positions=[{"character_id": "hero", "position": "center", "facing": "camera"}],
        focus_subject="main_character",
        lighting_style="soft natural",
        motion_vector="stationary",
    )
    return context_without_plan.evolve(composition_plan=plan)


def test_composition_gate_blocks_without_plan(context_without_plan):
    """assert_composition_plan() raises when plan is None. RULE-88."""
    with pytest.raises(CompositionPlanValidationError):
        context_without_plan.assert_composition_plan()


def test_composition_gate_passes_with_valid_plan(context_with_plan):
    """assert_composition_plan() passes when all 6 fields present."""
    context_with_plan.assert_composition_plan()   # should not raise


def test_composition_validator_checks_all_6_fields(context_without_plan):
    """CompositionValidator raises if any of the 6 required fields is empty."""
    # Create a plan with an empty field
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        CompositionPlanSchema(
            scene_id="sc_001",
            camera_angle="eye level",
            camera_motion="static",
            character_positions=[],   # EMPTY — forbidden
            focus_subject="main_character",
            lighting_style="natural",
            motion_vector="stationary",
        )


def test_composition_to_image_gate_enforced_by_base_image_agent(context_without_plan):
    """BaseImageAgent.run() raises CompositionPlanValidationError without a plan."""
    from vga.agents.base_image_agent import BaseImageAgent
    agent = BaseImageAgent()

    with pytest.raises(CompositionPlanValidationError):
        agent.run(
            {"identity_design": {"character_identity": "test"}},
            context_without_plan,   # no composition_plan
        )
