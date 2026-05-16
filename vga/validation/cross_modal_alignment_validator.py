"""
CrossModalAlignmentValidator — validates video↔audio duration alignment.
SOLE owner of cross-modal sync validation. FR-972, FR-973.
Spec: VGA Validation Spec v17.2; RULE-99 (Audio Realism), FR-972–FR-973
"""
from __future__ import annotations

import logging
from pathlib import Path

from vga.config.settings import settings
from vga.core.exceptions import CrossModalAlignmentError
from vga.models.schemas import CrossModalAlignmentRecord

logger = logging.getLogger(__name__)


class CrossModalAlignmentValidator:
    """Ensures video and audio track durations match within ±0.10 s tolerance. FR-972."""

    def validate(
        self,
        video_path: str | Path,
        audio_path: str | Path,
        scene_id: str,
        segment_id: str = "",
    ) -> CrossModalAlignmentRecord:
        """Measure and validate video↔audio duration alignment.

        Args:
            video_path: path to video file
            audio_path: path to audio file
            scene_id:   scene identifier
            segment_id: segment identifier (optional)

        Returns:
            CrossModalAlignmentRecord

        Raises:
            CrossModalAlignmentError if |alignment_error| > TIMING_TOLERANCE_S
        """
        video_duration = self._get_video_duration(Path(video_path))
        audio_duration = self._get_audio_duration(Path(audio_path))
        error = abs(video_duration - audio_duration)
        within_tolerance = error <= settings.TIMING_TOLERANCE_S

        logger.info(
            "CrossModalAlignment: scene=%s video=%.3fs audio=%.3fs error=%.3fs ok=%s",
            scene_id, video_duration, audio_duration, error, within_tolerance,
        )

        record = CrossModalAlignmentRecord(
            scene_id=scene_id,
            segment_id=segment_id,
            video_duration_s=video_duration,
            audio_duration_s=audio_duration,
            alignment_error_s=error,
            within_tolerance=within_tolerance,
        )

        if not within_tolerance:
            raise CrossModalAlignmentError(
                f"Video-audio alignment error {error:.3f}s exceeds tolerance "
                f"{settings.TIMING_TOLERANCE_S}s for scene {scene_id}. FR-972."
            )

        return record

    def _get_video_duration(self, path: Path) -> float:
        """Return video duration in seconds."""
        if not path.exists():
            logger.warning("CrossModalAlignment: video not found %s", path)
            return 0.0
        try:
            import cv2
            cap = cv2.VideoCapture(str(path))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            cap.release()
            return float(frame_count / fps) if fps > 0 else 0.0
        except Exception as exc:
            logger.error("CrossModalAlignment video duration error: %s", exc)
            return 0.0

    def _get_audio_duration(self, path: Path) -> float:
        """Return audio duration in seconds."""
        if not path.exists():
            logger.warning("CrossModalAlignment: audio not found %s", path)
            return 0.0
        try:
            from pydub import AudioSegment
            audio = AudioSegment.from_file(str(path))
            return len(audio) / 1000.0   # pydub returns milliseconds
        except Exception as exc:
            logger.error("CrossModalAlignment audio duration error: %s", exc)
            return 0.0
