"""
CLIPValidator — sole owner of character identity CLIP scoring.
ALWAYS uses frozen char_identity_ref from ImmutableContext. NEVER recomputes.
Spec: VGA Validation Spec v17.2; RULE-89, RULE-92, RULE-95, FR-500–FR-510
"""
from __future__ import annotations

import logging
from typing import List, Optional

from vga.config.settings import settings
from vga.core.exceptions import CLIPValidationError

logger = logging.getLogger(__name__)


class CLIPValidator:
    """Validates character identity consistency using CLIP cosine similarity.

    Critical invariants:
    - The reference embedding is ALWAYS taken from context.identity_state.embedding_vector
    - The embedding is NEVER recomputed mid-pipeline (frozen at S-07, RULE-95)
    - Threshold is ALWAYS read from settings.CLIP_IDENTITY_THRESHOLD (never hardcoded)
    """

    def __init__(self) -> None:
        self._model = None
        self._preprocess = None
        self._device = "cpu"
        logger.info("CLIPValidator initialized (lazy model load)")

    def _ensure_loaded(self) -> None:
        """Lazy-load CLIP model on first use."""
        if self._model is not None:
            return
        try:
            import open_clip
            import torch
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            self._model, _, self._preprocess = open_clip.create_model_and_transforms(
                "ViT-L-14", pretrained="openai"
            )
            self._model = self._model.to(self._device).eval()
            logger.info("CLIPValidator: CLIP ViT-L/14 loaded on %s", self._device)
        except ImportError:
            logger.warning("CLIPValidator: open_clip not available — using mock scorer")

    def encode_image(self, image: "PIL.Image.Image") -> List[float]:
        """Encode an image to a CLIP embedding vector. Returns list of floats."""
        self._ensure_loaded()
        if self._model is None:
            return [0.0] * 768   # fallback for environments without CLIP

        import torch
        tensor = self._preprocess(image).unsqueeze(0).to(self._device)
        with torch.no_grad():
            features = self._model.encode_image(tensor)
            features = features / features.norm(dim=-1, keepdim=True)
        return features.squeeze(0).cpu().tolist()

    def score(
        self,
        image_or_frame: "PIL.Image.Image",
        reference_embedding: Optional[List[float]],
    ) -> float:
        """Compute cosine similarity between image and frozen reference embedding.

        RULE-95: reference_embedding MUST come from context.identity_state.embedding_vector.
        Never pass a freshly computed embedding as the reference.

        Args:
            image_or_frame: PIL Image of the current frame/image to validate
            reference_embedding: frozen char_identity_ref from ImmutableContext

        Returns:
            float in [0, 1] — cosine similarity score
        """
        if reference_embedding is None:
            raise CLIPValidationError(
                "Cannot compute CLIP score — reference_embedding is None. "
                "Identity must be frozen before calling score(). RULE-95."
            )

        # Guard against zero-vector embedding (fallback when CLIP failed to load).
        # A zero vector would silently pass the None check but produce invalid scores.
        if all(v == 0.0 for v in reference_embedding[:8]):  # check first 8 dims
            raise CLIPValidationError(
                "reference_embedding appears to be a zero vector — "
                "CLIP model likely failed to load. Cannot validate identity. RULE-95."
            )

        current_embedding = self.encode_image(image_or_frame)
        return self._cosine_similarity(current_embedding, reference_embedding)

    def assert_above_threshold(
        self,
        score: float,
        stage_id: str,
        scene_id: str,
        threshold: Optional[float] = None,
    ) -> None:
        """Raise CLIPValidationError if score is below threshold. RULE-92."""
        effective_threshold = threshold or settings.CLIP_IDENTITY_THRESHOLD
        if score < effective_threshold:
            raise CLIPValidationError(
                f"CLIP identity score {score:.4f} below threshold {effective_threshold:.4f} "
                f"at stage {stage_id} scene {scene_id}. RULE-92.",
                stage_id=stage_id,
            )

    @staticmethod
    def _cosine_similarity(a: List[float], b: List[float]) -> float:
        """Compute cosine similarity between two float vectors."""
        if len(a) != len(b):
            raise ValueError(f"Embedding dimension mismatch: {len(a)} vs {len(b)}")
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)
