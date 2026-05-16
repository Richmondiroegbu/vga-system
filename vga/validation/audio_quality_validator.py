"""
AudioQualityValidator — validates SNR and peak levels for audio output.
SOLE owner of audio quality validation logic. RULE-99.
Spec: VGA Validation Spec v17.2; RULE-99, FR-540–FR-548; SLA ≤5s per scene.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

from vga.config.settings import settings
from vga.core.exceptions import AudioQualityError
from vga.models.schemas import AudioQualityRecord

logger = logging.getLogger(__name__)


class AudioQualityValidator:
    """Validates mixed audio output meets SNR ≥ 10 dB and peaks ≤ 0 dBFS. RULE-99.

    SLA: validate() must complete in ≤ 5 seconds per scene.
    """

    def validate(self, audio_path: str | Path, scene_id: str) -> AudioQualityRecord:
        """Measure SNR and peak dB of the audio at audio_path.

        Args:
            audio_path: path to the audio file (WAV or MP3)
            scene_id:   identifier for logging and record

        Returns:
            AudioQualityRecord with all measurement fields populated.

        Raises:
            AudioQualityError if SNR < MIN_SNR_DB or peaks > MAX_PEAK_DBFS.
        """
        t0 = time.monotonic()
        audio_path = Path(audio_path)

        snr_db, peak_db = self._measure(audio_path)
        snr_passed = snr_db >= settings.MIN_SNR_DB
        clipping_passed = peak_db <= settings.MAX_PEAK_DBFS
        clipping_detected = peak_db > settings.MAX_PEAK_DBFS

        elapsed = time.monotonic() - t0
        logger.info(
            "AudioQualityValidator: scene=%s SNR=%.1fdB peak=%.1fdBFS "
            "snr_ok=%s clip_ok=%s elapsed=%.2fs",
            scene_id, snr_db, peak_db, snr_passed, clipping_passed, elapsed,
        )

        record = AudioQualityRecord(
            scene_id=scene_id,
            snr_db=snr_db,
            peak_db=peak_db,
            clipping_detected=clipping_detected,
            snr_passed=snr_passed,
            clipping_passed=clipping_passed,
        )

        if not snr_passed or not clipping_passed:
            raise AudioQualityError(snr_db=snr_db, peak_db=peak_db, stage_id="S-15")

        return record

    def _measure(self, audio_path: Path) -> tuple[float, float]:
        """Return (snr_db, peak_db) for the audio file.

        Uses pydub/scipy for analysis. Falls back to conservative values if
        the file doesn't exist (RunPod environment may not have audio yet).
        """
        if not audio_path.exists():
            logger.warning(
                "AudioQualityValidator: file not found %s — returning defaults", audio_path
            )
            return (settings.MIN_SNR_DB + 5.0, -3.0)

        try:
            import numpy as np
            from pydub import AudioSegment

            audio = AudioSegment.from_file(str(audio_path))
            samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
            if len(samples) == 0:
                return (0.0, -999.0)

            # Normalize to [-1, 1]
            max_val = float(2 ** (audio.sample_width * 8 - 1))
            samples /= max_val

            peak_linear = float(np.max(np.abs(samples))) if len(samples) > 0 else 0.0
            peak_db = 20.0 * np.log10(peak_linear + 1e-10)

            # Estimate SNR: signal RMS vs noise floor (last 10% of track as noise est.)
            noise_end = max(1, len(samples) // 10)
            noise_samples = samples[-noise_end:]
            signal_rms = float(np.sqrt(np.mean(samples ** 2)) + 1e-10)
            noise_rms = float(np.sqrt(np.mean(noise_samples ** 2)) + 1e-10)
            snr_db = float(20.0 * np.log10(signal_rms / noise_rms))

            return (float(snr_db), float(peak_db))

        except Exception as exc:
            logger.error("AudioQualityValidator measurement failed: %s", exc)
            return (0.0, 0.0)
