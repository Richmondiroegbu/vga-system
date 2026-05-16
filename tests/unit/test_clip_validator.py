"""Unit tests for CLIPValidator. RULE-89, RULE-92, RULE-95."""
from __future__ import annotations

import pytest

from vga.core.exceptions import CLIPValidationError
from vga.validation.clip_validator import CLIPValidator


def test_cosine_similarity_identical_vectors():
    """Identical vectors should have similarity 1.0."""
    v = [1.0, 0.0, 0.0]
    score = CLIPValidator._cosine_similarity(v, v)
    assert abs(score - 1.0) < 1e-6


def test_cosine_similarity_orthogonal_vectors():
    """Orthogonal vectors should have similarity 0.0."""
    a = [1.0, 0.0, 0.0]
    b = [0.0, 1.0, 0.0]
    score = CLIPValidator._cosine_similarity(a, b)
    assert abs(score) < 1e-6


def test_cosine_similarity_opposite_vectors():
    """Opposite vectors should have similarity -1.0."""
    a = [1.0, 0.0]
    b = [-1.0, 0.0]
    score = CLIPValidator._cosine_similarity(a, b)
    assert abs(score + 1.0) < 1e-6


def test_score_raises_on_none_reference():
    """score() raises CLIPValidationError when reference is None. RULE-95."""
    pytest.importorskip("PIL", reason="Pillow not installed — skipping image test")
    from PIL import Image
    validator = CLIPValidator()
    dummy_image = Image.new("RGB", (64, 64))
    with pytest.raises(CLIPValidationError):
        validator.score(dummy_image, None)


def test_assert_above_threshold_passes():
    """assert_above_threshold() does not raise when score meets threshold."""
    validator = CLIPValidator()
    validator.assert_above_threshold(0.95, "S-05", "sc_001")


def test_assert_above_threshold_raises_below_threshold():
    """assert_above_threshold() raises CLIPValidationError when below 0.93. RULE-92."""
    validator = CLIPValidator()
    with pytest.raises(CLIPValidationError):
        validator.assert_above_threshold(0.90, "S-05", "sc_001")


def test_assert_above_threshold_custom_threshold():
    """Custom threshold can be passed."""
    validator = CLIPValidator()
    with pytest.raises(CLIPValidationError):
        validator.assert_above_threshold(0.85, "S-09", "sc_001", threshold=0.90)


def test_dimension_mismatch_raises():
    """Mismatched embedding dimensions raises ValueError."""
    with pytest.raises(ValueError, match="dimension mismatch"):
        CLIPValidator._cosine_similarity([1.0, 0.0], [1.0, 0.0, 0.0])
