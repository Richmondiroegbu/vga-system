"""
MotionEvolutionEngine — evolves motion parameters across segments for smooth flow.
Prevents abrupt motion direction changes between consecutive segments.
Spec: VGA Codebase Structure Design v17.2 §temporal/motion_evolution_engine.py
"""
from __future__ import annotations

import logging
from typing import Optional

from vga.state.immutable_context import MotionState

logger = logging.getLogger(__name__)

# Maximum allowed direction change per segment (radians)
_MAX_DIRECTION_CHANGE = 0.5


class MotionEvolutionEngine:
    """Smooths motion state evolution across segments to avoid discontinuities."""

    def evolve(
        self,
        prev_motion: MotionState,
        new_motion: MotionState,
        smoothing_alpha: float = 0.7,
    ) -> MotionState:
        """Blend previous and new motion state for smooth evolution.

        Args:
            prev_motion:     MotionState from previous segment
            new_motion:      estimated MotionState from current segment
            smoothing_alpha: blend factor (higher = more weight on new state)

        Returns:
            Smoothed MotionState
        """
        # Blend velocity vectors with exponential smoothing
        prev_vx, prev_vy = prev_motion.velocity_vector
        new_vx, new_vy = new_motion.velocity_vector

        blended_vx = prev_vx * (1 - smoothing_alpha) + new_vx * smoothing_alpha
        blended_vy = prev_vy * (1 - smoothing_alpha) + new_vy * smoothing_alpha
        blended_magnitude = (blended_vx ** 2 + blended_vy ** 2) ** 0.5

        # Use the new direction if magnitude is significant; else keep previous
        direction = (
            new_motion.direction
            if blended_magnitude > 0.02
            else prev_motion.direction
        )

        evolved = MotionState(
            velocity_vector=(blended_vx, blended_vy),
            direction=direction,
            magnitude=blended_magnitude,
        )
        logger.debug(
            "MotionEvolutionEngine: prev_dir=%s new_dir=%s evolved_dir=%s mag=%.3f",
            prev_motion.direction, new_motion.direction, evolved.direction, blended_magnitude,
        )
        return evolved

    def is_smooth_transition(
        self, prev_motion: MotionState, new_motion: MotionState
    ) -> bool:
        """Return True if the motion transition is smooth enough to continue."""
        # Direction changes from stationary to any direction are always OK
        if prev_motion.direction == "stationary":
            return True
        # Abrupt reversal detection
        prev_vx, prev_vy = prev_motion.velocity_vector
        new_vx, new_vy = new_motion.velocity_vector
        if prev_motion.magnitude > 0.05 and new_motion.magnitude > 0.05:
            # Check if direction reversed (dot product < 0)
            dot = prev_vx * new_vx + prev_vy * new_vy
            if dot < -0.5:
                logger.warning("MotionEvolutionEngine: abrupt reversal detected")
                return False
        return True
