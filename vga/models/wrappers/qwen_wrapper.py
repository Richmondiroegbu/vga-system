"""
QwenWrapper — wraps Qwen3-14B-Instruct (BF16, direct GPU load).
Used at S-01 (ScriptAgent) and S-04 (SceneCompositionAgent).
Spec: VGA Model Stack Setup Guide v7.2 §2.1
Upgrade: Qwen2.5-14B → Qwen3-14B (Apache 2.0, better structured JSON output).
Download: huggingface-cli download Qwen/Qwen3-14B --local-dir /workspace/models/qwen
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
    """Wraps Qwen3-14B-Instruct (BF16) for structured text generation.

    Thinking mode is explicitly disabled (enable_thinking=False) so the model
    returns JSON directly without a <think>...</think> prefix block, which
    would break the JSON extraction and add unnecessary latency.

    All inference requests use generate_structured() which binds output to a
    Pydantic schema and retries up to max_retries on validation failure.
    """

    MODEL_KEY = "qwen3-14b"

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
        max_tokens: int = 1024,  # was 4096 — a 60s script needs ~500 tokens max
    ) -> str:
        """Load model, run inference, unload, return raw text."""
        self._ensure_loaded()
        try:
            import torch
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]
            # enable_thinking=False: Qwen3 defaults to chain-of-thought mode which
            # prepends a <think>...</think> block before the answer. This breaks
            # JSON extraction and adds 5-30s of unnecessary compute per call.
            text = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            inputs = self._tokenizer(text, return_tensors="pt").to(self._model.device)
            with torch.no_grad():
                output_ids = self._model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=0.7,
                    do_sample=True,
                    repetition_penalty=1.1,   # prevents loops which slow CPU inference
                )
            generated = output_ids[0][inputs["input_ids"].shape[1]:]
            return self._tokenizer.decode(generated, skip_special_tokens=True).strip()
        except Exception as exc:
            raise ModelLoadError(f"QwenWrapper inference failed: {exc}") from exc

    def _ensure_loaded(self) -> None:
        """Lazy-load Qwen3-14B-Instruct in bfloat16 directly on GPU.

        Uses standard BF16 loading — no bitsandbytes required.
        Model: Qwen/Qwen3-14B (~28 GB BF16, fits comfortably in RTX PRO 6000 96GB).
        Generation speed: ~80-120 tokens/second on GPU.
        """
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            path = str(settings.QWEN_MODEL_PATH)
            logger.info("QwenWrapper: loading model from %s (bfloat16, GPU)", path)
            self._tokenizer = AutoTokenizer.from_pretrained(path)
            self._model = AutoModelForCausalLM.from_pretrained(
                path,
                torch_dtype=torch.bfloat16,
                device_map="cuda",   # direct GPU load — no bitsandbytes needed
            )
            logger.info("QwenWrapper: model loaded on GPU (bfloat16) — device: %s",
                       next(self._model.parameters()).device)
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
