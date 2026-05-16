"""
MotionStateTracker — estimates motion from TemporalBuffer via optical flow.
SLA: estimate() must complete in ≤ 1.0 second. NFR-168.
Spec: VGA Temporal Engine Spec v17.2 §MotionStateTracker; NFR-168
"""
from __future__ import annotations

import logging
import time

import numpy as np

from vga.models.enums import MotionDirection
from vga.state.immutable_context import MotionState
from vga.temporal.temporal_buffer_manager import TemporalBuffer

logger = logging.getLogger(__name__)

_STATIONARY_THRESHOLD = 0.02   # pixels/frame below which motion is "stationary"


class MotionStateTracker:
    """Estimates velocity_vector, direction, and magnitude from TemporalBuffer.

    Uses dense optical flow (Farneback method via OpenCV) between
    consecutive frame pairs in the buffer.
    SLA: estimate() ≤ 1.0 second.
    """

    def estimate(self, buffer: TemporalBuffer) -> MotionState:
        """Compute MotionState from the 5 buffer frames.

        Args:
            buffer: TemporalBuffer with exactly 5 frames

        Returns:
            MotionState with velocity_vector (dx, dy), direction, magnitude
        """
        t0 = time.monotonic()

        try:
            velocity_vector, magnitude = self._compute_optical_flow(buffer.frames)
            direction = self._classify_direction(velocity_vector, magnitude)
        except Exception as exc:
            logger.warning("MotionStateTracker: optical flow failed: %s", exc)
            velocity_vector = (0.0, 0.0)
            magnitude = 0.0
            direction = MotionDirection.STATIONARY.value

        elapsed = time.monotonic() - t0
        if elapsed > 1.0:
            logger.warning(
                "MotionStateTracker: estimate() took %.2fs (SLA ≤ 1.0s)", elapsed
            )
        else:
            logger.debug("MotionStateTracker: %.3fs velocity=%s mag=%.3f dir=%s",
                         elapsed, velocity_vector, magnitude, direction)

        return MotionState(
            velocity_vector=velocity_vector,
            direction=direction,
            magnitude=magnitude,
        )

    @staticmethod
    def _compute_optical_flow(
        frames: np.ndarray,
    ) -> tuple[tuple[float, float], float]:
        """Compute mean optical flow across consecutive frame pairs.

        Args:
            frames: np.ndarray of shape (5, H, W, 3), float32 normalized

        Returns:
            (velocity_vector, magnitude) where velocity_vector = (mean_dx, mean_dy)
        """
        try:
            import cv2

            # Denormalize to uint8 for optical flow
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            frames_uint8 = np.clip(
                (frames * std + mean) * 255.0, 0, 255
            ).astype(np.uint8)

            flows = []
            for i in range(len(frames_uint8) - 1):
                prev_gray = cv2.cvtColor(frames_uint8[i], cv2.COLOR_RGB2GRAY)
                next_gray = cv2.cvtColor(frames_uint8[i + 1], cv2.COLOR_RGB2GRAY)
                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray, next_gray,
                    None,
                    pyr_scale=0.5,
                    levels=3,
                    winsize=15,
                    iterations=3,
                    poly_n=5,
                    poly_sigma=1.2,
                    flags=0,
                )
                flows.append(flow)

            if not flows:
                return (0.0, 0.0), 0.0

            mean_flow = np.mean(flows, axis=0)   # (H, W, 2)
            mean_velocity = np.mean(mean_flow, axis=(0, 1))   # (2,) — (dx, dy)
            magnitude = float(np.sqrt(mean_velocity[0] ** 2 + mean_velocity[1] ** 2))
            return (float(mean_velocity[0]), float(mean_velocity[1])), magnitude

        except ImportError:
            # Fallback without OpenCV
            return (0.0, 0.0), 0.0

    @staticmethod
    def _classify_direction(
        velocity_vector: tuple[float, float],
        magnitude: float,
    ) -> str:
        """Classify dominant motion direction from mean flow vector."""
        if magnitude < _STATIONARY_THRESHOLD:
            return MotionDirection.STATIONARY.value

        dx, dy = velocity_vector
        abs_dx, abs_dy = abs(dx), abs(dy)

        if abs_dx > abs_dy:
            return MotionDirection.RIGHT.value if dx > 0 else MotionDirection.LEFT.value
        else:
            return MotionDirection.FORWARD.value if dy > 0 else MotionDirection.BACKWARD.value
