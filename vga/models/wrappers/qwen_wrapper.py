"""
QwenWrapper — wraps Qwen2.5-14B-Instruct (unsloth 4bit).
Used at S-01 (ScriptAgent) and S-04 (SceneCompositionAgent).
Spec: VGA Model Stack Setup Guide v7.2 §2.1
"""
from __future__ import annotations

import json
import logging
import time
from typing import Type, TypeVar

from pydantic import BaseModel, ValidationError

from vga.config.settings import settings
from vga.core.exceptions import ModelLoadError, RetryExhaustedError, SchemaValidationError

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

# System prompt that instructs Qwen to return valid JSON
_JSON_SYSTEM_PROMPT = (
    "You are a professional screenwriter AI assistant for the VGA cinematic system. "
    "ALWAYS respond with valid JSON only. No markdown, no explanation, no code blocks."
)


class QwenWrapper:
    """Wraps Qwen2.5-14B-Instruct (unsloth 4bit) for structured text generation.

    All inference requests use generate_structured() which binds output to a
    Pydantic schema and retries up to max_retries on validation failure.
    """

    MODEL_KEY = "qwen2.5-14b"

    def __init__(self) -> None:
        self._tokenizer = None
        self._model = None
        logger.info("QwenWrapper initialized (lazy load)")

    def generate_structured(
        self,
        prompt: str,
        output_schema: Type[T],
        max_retries: int | None = None,
        system_prompt: str = _JSON_SYSTEM_PROMPT,
    ) -> T:
        """Generate structured output bound to a Pydantic schema.

        Retries up to max_retries on JSON parse or schema validation failure.

        Args:
            prompt:        user prompt
            output_schema: Pydantic model class to validate against
            max_retries:   number of retry attempts (defaults to settings.MAX_RETRIES)
            system_prompt: system message for the model

        Returns:
            Validated Pydantic model instance

        Raises:
            RetryExhaustedError if all retries fail
        """
        max_retries = max_retries or settings.MAX_RETRIES
        last_error: Exception | None = None

        for attempt in range(max_retries):
            try:
                raw_text = self._generate_raw(prompt, system_prompt)
                data = self._extract_json(raw_text)
                return output_schema.model_validate(data)
            except (ValidationError, ValueError, json.JSONDecodeError) as exc:
                last_error = exc
                logger.warning(
                    "QwenWrapper structured attempt %d/%d failed: %s",
                    attempt + 1, max_retries, exc,
                )
                backoff = settings.BACKOFF_SECONDS[min(attempt, len(settings.BACKOFF_SECONDS) - 1)]
                time.sleep(backoff)

        raise RetryExhaustedError(
            f"QwenWrapper failed after {max_retries} attempts: {last_error}"
        )

    def generate_text(self, prompt: str, max_tokens: int = 2048) -> str:
        """Generate plain text response (no schema binding)."""
        return self._generate_raw(prompt, _JSON_SYSTEM_PROMPT, max_tokens=max_tokens)

    def _generate_raw(
        self,
        prompt: str,
        system_prompt: str,
        max_tokens: int = 4096,
    ) -> str:
        """Load model, run inference, unload, return raw text."""
        self._ensure_loaded()
        try:
            import torch
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]
            text = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self._tokenizer(text, return_tensors="pt").to(self._model.device)
            with torch.no_grad():
                output_ids = self._model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=0.7,
                    do_sample=True,
                )
            generated = output_ids[0][inputs["input_ids"].shape[1]:]
            return self._tokenizer.decode(generated, skip_special_tokens=True).strip()
        except Exception as exc:
            raise ModelLoadError(f"QwenWrapper inference failed: {exc}") from exc

    def _ensure_loaded(self) -> None:
        """Lazy-load the Qwen model using BitsAndBytesConfig for 4-bit quantization.
        Note: bitsandbytes may use CPU backend on RTX 5090 due to a driver detection
        quirk. Qwen inference takes ~75s on CPU, which is acceptable for testing.
        All other GPU models (FLUX, CLIP, Wan2.2) still run on GPU.
        """
        if self._model is not None:
            return
        try:
            from transformers import (
                AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
            )
            path = str(settings.QWEN_MODEL_PATH)
            logger.info("QwenWrapper: loading model from %s", path)
            self._tokenizer = AutoTokenizer.from_pretrained(path)
            import torch
            # Force GPU device map — bypasses bitsandbytes broken device detection
            # on RTX 5090 (sm_120). {"": "cuda:0"} tells transformers to place ALL
            # layers on GPU without going through bnb's cuDeviceGet detection.
            gpu_available = torch.cuda.is_available()
            device_map = {"": "cuda:0"} if gpu_available else "cpu"
            logger.info("QwenWrapper: device_map=%s (GPU available: %s)", device_map, gpu_available)

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            self._model = AutoModelForCausalLM.from_pretrained(
                path,
                quantization_config=bnb_config,
                device_map=device_map,
            )
            logger.info("QwenWrapper: model loaded (4-bit BNB)")
        except Exception as exc:
            raise ModelLoadError(f"QwenWrapper failed to load: {exc}") from exc

    @staticmethod
    def _extract_json(text: str) -> dict:
        """Extract the first valid JSON object from raw model text."""
        text = text.strip()
        # Strip markdown code fences if present
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(l for l in lines if not l.startswith("```")).strip()
        # Find first '{' and last '}'
        start = text.find("{")
        end = text.rfind("}") + 1
        if start == -1 or end == 0:
            raise ValueError(f"No JSON object found in model output: {text[:200]!r}")
        return json.loads(text[start:end])
