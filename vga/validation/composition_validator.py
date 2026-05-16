"""
CompositionPlanValidator — validates CompositionPlan schema completeness.
SOLE owner of CompositionPlan validation logic. RULE-88.
Spec: VGA Validation Spec v17.2; RULE-88, FR-520–FR-525
"""
from __future__ import annotations

import logging

from pydantic import ValidationError

from vga.core.exceptions import CompositionPlanValidationError
from vga.models.schemas import CompositionPlanSchema
from vga.state.immutable_context import ImmutableContext

logger = logging.getLogger(__name__)

# All 6 mandatory fields of a CompositionPlan (RULE-88)
REQUIRED_FIELDS = (
    "camera_angle",
    "camera_motion",
    "character_positions",
    "focus_subject",
    "lighting_style",
    "motion_vector",
)


class CompositionValidator:
    """Validates CompositionPlan presence and schema completeness.

    Usage:
        validator = CompositionValidator()
        validator.assert_in_context(context)   # before image/video generation
        validated_plan = validator.validate_schema(raw_dict, scene_id)
    """

    def assert_in_context(self, context: ImmutableContext) -> CompositionPlanSchema:
        """Assert CompositionPlan is present and valid in context. RULE-88.

        Raises CompositionPlanValidationError if plan is missing or invalid.
        Returns the validated CompositionPlanSchema.
        """
        context.assert_composition_plan()   # raises if None

        plan = context.composition_plan
        assert plan is not None  # type narrowing

        # Verify all 6 required fields are non-empty
        for field in REQUIRED_FIELDS:
            value = getattr(plan, field, None)
            if value is None or (isinstance(value, str) and not value.strip()):
                raise CompositionPlanValidationError(
                    f"CompositionPlan field '{field}' is empty or missing for "
                    f"scene {context.scene_id}. RULE-88 requires all 6 fields."
                )
            if isinstance(value, list) and len(value) == 0:
                raise CompositionPlanValidationError(
                    f"CompositionPlan field '{field}' is an empty list for "
                    f"scene {context.scene_id}. RULE-88 requires all 6 fields."
                )

        logger.debug(
            "CompositionValidator: plan validated for scene %s", context.scene_id
        )
        return plan

    def validate_schema(self, raw: dict, scene_id: str) -> CompositionPlanSchema:
        """Parse and validate a raw dict as CompositionPlanSchema.

        Raises CompositionPlanValidationError on schema failure.
        """
        try:
            plan = CompositionPlanSchema(scene_id=scene_id, **raw)
        except ValidationError as exc:
            raise CompositionPlanValidationError(
                f"CompositionPlan schema validation failed for scene {scene_id}: {exc}"
            ) from exc

        return plan
