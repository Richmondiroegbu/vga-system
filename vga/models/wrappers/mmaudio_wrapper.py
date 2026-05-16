"""
MMAudioWrapper — wraps MMAudio medium_44k for ambient audio generation.
S-13 (AmbientAudioAgent). Generates ambient soundscapes synchronized to video.
Model selected via settings.MMAUDIO_MODEL_NAME (medium_44k = 2.49 GB, 44.1kHz).
Spec: VGA Model Stack Setup Guide v7.2 §2.8
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

from vga.config.settings import settings
from vga.core.exceptions import ModelLoadError

logger = logging.getLogger(__name__)


class MMAudioWrapper:
    """Wraps MMAudio medium_44k for ambient audio generation from video segments.

    Model is configured via settings.MMAUDIO_MODEL_NAME.
    To change model size: update MMAUDIO_MODEL_NAME in settings.py.
    Valid values: small_16k, small_44k, medium_44k, large_44k, large_44k_v2
    """

    MODEL_KEY = "mmaudio-medium-44k"

    def __init__(self) -> None:
        self._model = None
        logger.info("MMAudioWrapper initialized (lazy load)")

    def generate(
        self,
        video_path: str | Path,
        prompt: str,
        duration_s: float,
        output_path: str | Path,
        seed: int = 42,
    ) -> str:
        """Generate ambient audio synchronized to the video segment.

        Args:
            video_path:  video segment to synchronize audio to
            prompt:      description of ambient sound (e.g., "city street noise")
            duration_s:  target duration in seconds
            output_path: where to write the output audio file
            seed:        random seed

        Returns:
            str path to generated audio file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        self._ensure_loaded()

        try:
            audio_path = self._run_mmaudio(
                video_path=Path(video_path),
                prompt=prompt,
                duration_s=duration_s,
                output_path=output_path,
                seed=seed,
            )
            logger.info("MMAudioWrapper: generated ambient audio → %s", audio_path)
            return str(audio_path)
        except Exception as exc:
            logger.error("MMAudioWrapper generation failed: %s", exc)
            # Write silence as fallback
            self._write_silence(output_path, duration_s)
            return str(output_path)

    def _run_mmaudio(
        self,
        video_path: Path,
        prompt: str,
        duration_s: float,
        output_path: Path,
        seed: int,
    ) -> Path:
        """Invoke MMAudio model for generation."""
        mmaudio_path = settings.MMAUDIO_PATH
        if str(mmaudio_path) not in sys.path:
            sys.path.insert(0, str(mmaudio_path))

        try:
            import mmaudio  # type: ignore
            import torch

            if self._model is None:
                self._model = mmaudio.get_model(
                    settings.MMAUDIO_MODEL_NAME, str(mmaudio_path)
                )
                self._model.eval()
                logger.info(
                    "MMAudioWrapper: loaded model '%s'", settings.MMAUDIO_MODEL_NAME
                )

            with torch.no_grad():
                result = mmaudio.generate(
                    video=str(video_path),
                    text=prompt,
                    duration=duration_s,
                    seed=seed,
                    model=self._model,
                )
            result_audio = result["audio"]
            import torchaudio
            torchaudio.save(
                str(output_path), result_audio, settings.MMAUDIO_SAMPLE_RATE
            )
            return output_path

        except ImportError:
            logger.warning("MMAudio not available — writing silence")
            self._write_silence(output_path, duration_s)
            return output_path

    def _ensure_loaded(self) -> None:
        """Placeholder — model is loaded lazily in _run_mmaudio."""
        pass

    @staticmethod
    def _write_silence(output_path: Path, duration_s: float) -> None:
        try:
            import numpy as np
            import soundfile as sf
            sr = settings.MMAUDIO_SAMPLE_RATE
            sf.write(str(output_path), np.zeros(int(duration_s * sr), dtype=np.float32), sr)
        except Exception:
            pass
