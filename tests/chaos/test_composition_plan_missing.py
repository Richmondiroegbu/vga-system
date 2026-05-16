"""
Chaos test: CompositionPlan missing before image/video generation.
RULE-88 — NO generation without a valid CompositionPlan.
"""
from __future__ import annotations

import pytest

from vga.core.exceptions import CompositionPlanValidationError
from vga.state.context_factory import ContextFactory
from vga.validation.composition_validator import CompositionValidator


@pytest.fixture
def context_no_plan():
    return ContextFactory.create_initial("job_test", "sc_001")


def test_assert_composition_plan_raises_when_none(context_no_plan):
    """Context with no plan raises CompositionPlanValidationError. RULE-88."""
    with pytest.raises(CompositionPlanValidationError):
        context_no_plan.assert_composition_plan()


def test_composition_validator_raises_on_missing_context(context_no_plan):
    """CompositionValidator.assert_in_context raises when plan is None."""
    validator = CompositionValidator()
    with pytest.raises(CompositionPlanValidationError):
        validator.assert_in_context(context_no_plan)


def test_base_image_agent_blocked_without_plan(context_no_plan):
    """BaseImageAgent.run() is blocked by CompositionPlan gate. RULE-88."""
    from vga.agents.base_image_agent import BaseImageAgent
    agent = BaseImageAgent()
    with pytest.raises(CompositionPlanValidationError):
        agent.run({"identity_design": {}}, context_no_plan)


def test_video_segment_generator_blocked_without_plan(context_no_plan):
    """VideoSegmentGenerator.run() blocked without CompositionPlan. RULE-88."""
    from vga.agents.video_segment_generator import VideoSegmentGenerator
    agent = VideoSegmentGenerator()
    with pytest.raises(CompositionPlanValidationError):
        agent.run({"refined_image": None, "output_dir": "/tmp"}, context_no_plan)


def test_composition_plan_schema_requires_all_6_fields():
    """CompositionPlanSchema with any missing field raises ValidationError. RULE-88."""
    from pydantic import ValidationError
    from vga.models.schemas import CompositionPlanSchema

    # Missing motion_vector
    with pytest.raises(ValidationError):
        CompositionPlanSchema(
            scene_id="sc_001",
            camera_angle="eye level",
            camera_motion="static",
            character_positions=[{"character_id": "hero"}],
            focus_subject="hero",
            lighting_style="natural",
            # motion_vector MISSING
        )
