"""Unit tests for SceneCompositionAgent. RULE-88, S-04."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from vga.agents.scene_composition_agent import SceneCompositionAgent
from vga.core.exceptions import CompositionPlanValidationError
from vga.models.schemas import CompositionPlanSchema, ScenePlanSchema, SegmentPlanSchema
from vga.state.context_factory import ContextFactory


def _make_scene_plan(scene_id="sc_001"):
    return ScenePlanSchema(
        job_id="job_test",
        scene_id=scene_id,
        scene_number=1,
        duration_s=20.0,
        segments=[
            SegmentPlanSchema(
                segment_id=f"{scene_id}_seg1",
                scene_id=scene_id,
                segment_number=1,
                duration_s=5.0,
                action_description="protagonist walks forward",
                camera_instruction="eye level, static",
            ),
            SegmentPlanSchema(
                segment_id=f"{scene_id}_seg2",
                scene_id=scene_id,
                segment_number=2,
                duration_s=5.0,
                action_description="protagonist looks determined",
                camera_instruction="close-up, slow push-in",
            ),
            SegmentPlanSchema(
                segment_id=f"{scene_id}_seg3",
                scene_id=scene_id,
                segment_number=3,
                duration_s=5.0,
                action_description="wide shot of environment",
                camera_instruction="wide shot, slow pullback",
            ),
            SegmentPlanSchema(
                segment_id=f"{scene_id}_seg4",
                scene_id=scene_id,
                segment_number=4,
                duration_s=5.0,
                action_description="close up of face triumphant",
                camera_instruction="close-up, steady",
            ),
        ],
        setting="urban environment, sunrise",
        characters_present=["main_character"],
        emotional_beat="determination",
    )


@pytest.fixture
def valid_context():
    return ContextFactory.create_initial("job_test", "sc_001")


@pytest.fixture
def mock_qwen():
    mock = MagicMock()
    mock.generate_structured.return_value = CompositionPlanSchema(
        scene_id="sc_001",
        camera_angle="medium shot",
        camera_motion="slow dolly forward",
        character_positions=[{"character_id": "main_character", "position": "center", "facing": "camera"}],
        focus_subject="main_character",
        lighting_style="soft natural",
        motion_vector="forward_slow",
    )
    return mock


def test_scene_composition_agent_produces_composition_plan(valid_context, mock_qwen, tmp_path):
    """SceneCompositionAgent produces a valid CompositionPlan with all 6 fields."""
    agent = SceneCompositionAgent(qwen=mock_qwen)
    scene_plan = _make_scene_plan()

    with patch("vga.agents.scene_composition_agent.settings") as mock_settings:
        mock_settings.COMPOSITION_MAX_RETRIES = 3
        mock_settings.SCHEMA_VERSION = "v6.0"
        mock_settings.HRG_DIR = tmp_path

        output, new_context = agent.run(
            {"scene_plan": scene_plan, "identity_design": {"character_identity": "test"}},
            valid_context,
        )

    assert isinstance(output, CompositionPlanSchema)
    assert output.camera_angle == "medium shot"
    assert output.motion_vector == "forward_slow"
    assert new_context.composition_plan is not None


def test_scene_composition_agent_raises_after_max_retries(valid_context, tmp_path):
    """SceneCompositionAgent raises CompositionPlanValidationError after COMPOSITION_MAX_RETRIES failures."""
    failing_qwen = MagicMock()
    failing_qwen.generate_structured.side_effect = Exception("LLM failed")

    agent = SceneCompositionAgent(qwen=failing_qwen)
    scene_plan = _make_scene_plan()

    with patch("vga.agents.scene_composition_agent.settings") as mock_settings:
        mock_settings.COMPOSITION_MAX_RETRIES = 3
        mock_settings.SCHEMA_VERSION = "v6.0"
        mock_settings.HRG_DIR = tmp_path

        with pytest.raises(CompositionPlanValidationError):
            agent.run(
                {"scene_plan": scene_plan, "identity_design": {}},
                valid_context,
            )


def test_scene_composition_agent_evolves_context_with_plan(valid_context, mock_qwen, tmp_path):
    """Context returned from SceneCompositionAgent has composition_plan, camera_state, lighting_state."""
    agent = SceneCompositionAgent(qwen=mock_qwen)
    scene_plan = _make_scene_plan()

    with patch("vga.agents.scene_composition_agent.settings") as mock_settings:
        mock_settings.COMPOSITION_MAX_RETRIES = 3
        mock_settings.SCHEMA_VERSION = "v6.0"
        mock_settings.HRG_DIR = tmp_path

        _, new_context = agent.run(
            {"scene_plan": scene_plan, "identity_design": {}},
            valid_context,
        )

    assert new_context.composition_plan is not None
    assert new_context.camera_state.angle == "medium shot"
    assert new_context.lighting_state.style == "soft natural"
