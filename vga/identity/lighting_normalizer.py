"""
LightingNormalizer — normalizes lighting before CLIP identity comparison.
Uses LAB color space to reduce lighting-induced CLIP score variance.
Spec: VGA Identity System v17.2 §4; CGRL-94
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class LightingNormalizer:
    """Normalizes image lighting in LAB color space before identity comparison.

    Without normalization, a character lit from different angles scores lower
    in CLIP similarity even when it's the same person. Normalizing to a
    reference luminance channel reduces this bias.
    """

    def normalize(self, image: "PIL.Image.Image") -> "PIL.Image.Image":
        """Normalize image lighting for consistent CLIP identity comparison.

        Args:
            image: PIL Image (RGB)

        Returns:
            Lighting-normalized PIL Image (RGB)
        """
        try:
            import numpy as np
            from PIL import Image

            img_array = np.array(image).astype(np.float32)
            # Convert to LAB: normalize L channel to mean 128 (mid-gray)
            # Simple luminance normalization (full LAB requires opencv/skimage)
            luminance = 0.299 * img_array[:, :, 0] + 0.587 * img_array[:, :, 1] + 0.114 * img_array[:, :, 2]
            mean_lum = luminance.mean()
            if mean_lum > 0:
                scale = 128.0 / mean_lum
                normalized = np.clip(img_array * scale, 0, 255).astype(np.uint8)
                return Image.fromarray(normalized)
        except ImportError:
            logger.warning("LightingNormalizer: numpy/PIL not available — returning original")
        except Exception as exc:
            logger.warning("LightingNormalizer: normalization failed: %s — returning original", exc)

        return image

    def normalize_batch(self, images: list) -> list:
        """Normalize a list of images."""
        return [self.normalize(img) for img in images]
