"""
MusicGenWrapper — wraps MusicGen-medium for background music generation.
S-14 (MusicAgent). Generates scene-appropriate instrumental music.
Spec: VGA Model Stack Setup Guide v7.2 §2.9
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from vga.config.settings import settings
from vga.core.exceptions import ModelLoadError

logger = logging.getLogger(__name__)


class MusicGenWrapper:
    """Wraps MusicGen-medium for background music generation per scene."""

    MODEL_KEY = "musicgen-medium"

    def __init__(self) -> None:
        self._model = None
        self._processor = None
        logger.info("MusicGenWrapper initialized (lazy load)")

    def generate(
        self,
        prompt: str,
        duration_s: float,
        output_path: str | Path,
        seed: int = 42,
    ) -> str:
        """Generate instrumental background music.

        Args:
            prompt:      music description (e.g., "emotional orchestral, inspiring")
            duration_s:  target duration in seconds
            output_path: where to write the output audio file
            seed:        random seed

        Returns:
            str path to generated music file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        self._ensure_loaded()

        try:
            audio = self._run_musicgen(prompt, duration_s, seed)
            self._save_audio(audio, output_path, sample_rate=32000)
            logger.info("MusicGenWrapper: generated music → %s", output_path)
            return str(output_path)
        except Exception as exc:
            logger.error("MusicGenWrapper generation failed: %s", exc)
            self._write_silence(output_path, duration_s)
            return str(output_path)

    def _run_musicgen(self, prompt: str, duration_s: float, seed: int) -> "torch.Tensor":
        """Run MusicGen model."""
        import torch

        if self._model is None:
            self._ensure_loaded()

        with torch.no_grad():
            inputs = self._processor(
                text=[prompt],
                padding=True,
                return_tensors="pt",
            )
            audio_values = self._model.generate(
                **inputs,
                max_new_tokens=int(duration_s * 50),   # ~50 tokens/second
            )
        return audio_values[0, 0]

    def _ensure_loaded(self) -> None:
        """Lazy-load MusicGen model."""
        if self._model is not None:
            return
        try:
            from transformers import AutoProcessor, MusicgenForConditionalGeneration
            path = str(settings.MUSICGEN_MODEL_PATH)
            logger.info("MusicGenWrapper: loading from %s", path)
            self._processor = AutoProcessor.from_pretrained(path)
            self._model = MusicgenForConditionalGeneration.from_pretrained(path)
            logger.info("MusicGenWrapper: MusicGen-medium loaded")
        except Exception as exc:
            raise ModelLoadError(f"MusicGenWrapper failed to load: {exc}") from exc

    @staticmethod
    def _save_audio(audio: "torch.Tensor", output_path: Path, sample_rate: int) -> None:
        try:
            import torchaudio
            torchaudio.save(str(output_path), audio.unsqueeze(0), sample_rate)
        except Exception as exc:
            logger.error("MusicGen save failed: %s", exc)

    @staticmethod
    def _write_silence(output_path: Path, duration_s: float) -> None:
        try:
            import numpy as np
            import soundfile as sf
            sr = 32000
            sf.write(str(output_path), np.zeros(int(duration_s * sr), dtype=np.float32), sr)
        except Exception:
            pass
