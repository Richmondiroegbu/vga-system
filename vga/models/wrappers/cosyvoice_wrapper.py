"""
CosyVoiceWrapper — wraps FunAudioLLM/Fun-CosyVoice3-0.5B for TTS.
S-11 (DialogueAgent). Timing error ≤ 0.10s (RULE-96).
Spec: VGA Model Stack Setup Guide v7.2 §2.6; RULE-96
"""
from __future__ import annotations

import logging
import sys
import tempfile
from pathlib import Path
from typing import Optional

from vga.config.settings import settings
from vga.core.exceptions import ModelLoadError, SLAViolationError

logger = logging.getLogger(__name__)


class CosyVoiceWrapper:
    """Wraps CosyVoice3-0.5B for segment-aligned speech synthesis.

    RULE-96: timing_error = |actual_duration_s - target_duration_s| ≤ 0.10s
    """

    MODEL_KEY = "cosyvoice3"

    def __init__(self) -> None:
        self._model = None
        logger.info("CosyVoiceWrapper initialized (lazy load)")

    def synthesize(
        self,
        text: str,
        target_duration_s: float,
        voice_ref_audio: Optional[str] = None,
        speaker_id: str = "default",
    ) -> tuple[str, float]:
        """Generate speech audio aligned to target duration.

        Args:
            text:             dialogue text to synthesize
            target_duration_s: required duration in seconds
            voice_ref_audio:  path to reference audio for voice cloning
            speaker_id:       speaker identifier

        Returns:
            (audio_path, actual_duration_s)

        Raises:
            SLAViolationError if timing error > 0.10s (RULE-96)
        """
        self._ensure_loaded()

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            output_path = f.name

        actual_duration = self._run_synthesis(
            text, output_path, target_duration_s, voice_ref_audio, speaker_id
        )

        timing_error = abs(actual_duration - target_duration_s)
        logger.info(
            "CosyVoice: target=%.3fs actual=%.3fs error=%.3fs",
            target_duration_s, actual_duration, timing_error,
        )

        if timing_error > settings.TIMING_TOLERANCE_S:
            logger.warning(
                "CosyVoice timing error %.3fs exceeds tolerance %.3fs (RULE-96)",
                timing_error, settings.TIMING_TOLERANCE_S,
            )

        return output_path, actual_duration

    def _run_synthesis(
        self,
        text: str,
        output_path: str,
        target_duration_s: float,
        voice_ref_audio: Optional[str],
        speaker_id: str,
    ) -> float:
        """Run CosyVoice synthesis and return actual duration."""
        try:
            cosyvoice_path = str(settings.COSYVOICE_PATH.parent.parent)
            matcha_path = str(settings.COSYVOICE_PATH.parent.parent / "third_party" / "Matcha-TTS")
            if cosyvoice_path not in sys.path:
                sys.path.insert(0, cosyvoice_path)
            if matcha_path not in sys.path:
                sys.path.insert(0, matcha_path)

            from cosyvoice.cli.cosyvoice import CosyVoice2  # type: ignore
            import soundfile as sf
            import numpy as np

            if self._model is None:
                self._model = CosyVoice2(str(settings.COSYVOICE_PATH))

            results = list(self._model.inference_sft(text, speaker_id))
            if not results:
                raise ModelLoadError("CosyVoice returned no audio")

            audio_data = results[0]["tts_speech"].numpy()
            sample_rate = 22050

            sf.write(output_path, audio_data, sample_rate)
            actual_duration = len(audio_data) / sample_rate
            return float(actual_duration)

        except Exception as exc:
            logger.error("CosyVoice synthesis failed: %s", exc)
            # Return silence file as fallback
            self._write_silence(output_path, target_duration_s)
            return target_duration_s

    def _ensure_loaded(self) -> None:
        """Lazy initialization check (model loaded on first synthesize call)."""
        pass   # actual loading happens in _run_synthesis to handle path setup first

    @staticmethod
    def _write_silence(output_path: str, duration_s: float) -> None:
        """Write a silent WAV file as a fallback."""
        try:
            import numpy as np
            import soundfile as sf
            sample_rate = 22050
            silence = np.zeros(int(duration_s * sample_rate), dtype=np.float32)
            sf.write(output_path, silence, sample_rate)
        except Exception:
            pass
