"""
Unit tests for watermark filtering.

Tests the WatermarkFilter class and convenience functions for detecting
and removing watermark text from PDF pages.
"""

import pytest
from tables.detector.watermark_filter import (
    WatermarkFilter,
    WatermarkCandidate,
    FilterResult,
    filter_watermarks,
    is_likely_watermark,
    WATERMARK_PATTERNS,
)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def default_filter():
    """Create a default WatermarkFilter."""
    return WatermarkFilter()


@pytest.fixture
def sample_words():
    """Sample word dictionaries for testing."""
    return [
        {"text": "Date", "x0": 50, "x1": 80, "top": 100, "bottom": 112, "upright": True},
        {"text": "Description", "x0": 100, "x1": 180, "top": 100, "bottom": 112, "upright": True},
        {"text": "Amount", "x0": 200, "x1": 250, "top": 100, "bottom": 112, "upright": True},
        {"text": "01/01/2024", "x0": 50, "x1": 100, "top": 120, "bottom": 132, "upright": True},
        {"text": "Transaction", "x0": 100, "x1": 180, "top": 120, "bottom": 132, "upright": True},
        {"text": "1000.00", "x0": 200, "x1": 250, "top": 120, "bottom": 132, "upright": True},
    ]


@pytest.fixture
def words_with_watermark():
    """Words including a watermark."""
    return [
        {"text": "Date", "x0": 50, "x1": 80, "top": 100, "bottom": 112, "upright": True},
        {"text": "DRAFT", "x0": 200, "x1": 400, "top": 300, "bottom": 350, "upright": False},
        {"text": "Amount", "x0": 200, "x1": 250, "top": 100, "bottom": 112, "upright": True},
    ]


@pytest.fixture
def rotated_words():
    """Words with rotated text."""
    return [
        {"text": "Normal", "x0": 50, "x1": 100, "top": 100, "bottom": 112, "upright": True},
        {"text": "Rotated", "x0": 150, "x1": 200, "top": 200, "bottom": 250, "upright": False},
        {"text": "AlsoNormal", "x0": 250, "x1": 320, "top": 100, "bottom": 112, "upright": True},
    ]


# =============================================================================
# WatermarkFilter Initialization Tests
# =============================================================================

class TestWatermarkFilterInit:
    """Tests for WatermarkFilter initialization."""

    def test_default_init(self):
        """Test default initialization."""
        filter = WatermarkFilter()
        assert filter.filter_rotated is True
        assert filter.filter_patterns is True
        assert filter.max_font_size is None
        assert filter.custom_patterns == []

    def test_custom_init(self):
        """Test custom initialization."""
        filter = WatermarkFilter(
            filter_rotated=False,
            filter_patterns=False,
            max_font_size=40.0,
            custom_patterns=[r"custom_pattern"],
        )
        assert filter.filter_rotated is False
        assert filter.filter_patterns is False
        assert filter.max_font_size == 40.0
        assert len(filter.custom_patterns) == 1

    def test_custom_patterns_compiled(self):
        """Test that custom patterns are compiled."""
        filter = WatermarkFilter(custom_patterns=[r"test\d+"])
        assert len(filter._custom_compiled) == 1


# =============================================================================
# Pattern Matching Tests
# =============================================================================

class TestPatternMatching:
    """Tests for watermark pattern matching."""

    @pytest.mark.parametrize("text", [
        "DRAFT",
        "draft",
        "  DRAFT  ",
        "CONFIDENTIAL",
        "confidential",
        "COPY",
        "SAMPLE",
        "SPECIMEN",
        "DUPLICATE",
        "VOID",
        "CANCELLED",
        "ORIGINAL",
        "NOT VALID",
        "FOR REFERENCE ONLY",
        "UNOFFICIAL",
    ])
    def test_matches_known_watermark_patterns(self, default_filter, text):
        """Test that known watermark patterns are detected."""
        word = {"text": text, "upright": True}
        assert default_filter._matches_watermark_pattern(text)

    @pytest.mark.parametrize("text", [
        "Date",
        "Description",
        "Amount",
        "Transaction",
        "Balance",
        "01/01/2024",
        "1000.00",
        "HDFC Bank",
        "Account Number",
    ])
    def test_does_not_match_normal_text(self, default_filter, text):
        """Test that normal text is not matched."""
        assert not default_filter._matches_watermark_pattern(text)

    def test_pattern_case_insensitive(self, default_filter):
        """Test that pattern matching is case insensitive."""
        assert default_filter._matches_watermark_pattern("draft")
        assert default_filter._matches_watermark_pattern("DRAFT")
        assert default_filter._matches_watermark_pattern("Draft")
        assert default_filter._matches_watermark_pattern("DrAfT")


# =============================================================================
# Rotation Detection Tests
# =============================================================================

class TestRotationDetection:
    """Tests for rotated text detection."""

    def test_detects_rotated_text(self, default_filter):
        """Test that non-upright text is detected."""
        word = {"text": "Rotated", "upright": False}
        candidate = default_filter._check_word(word, {})
        assert candidate is not None
        assert "rotated_text" in candidate.reasons
        assert candidate.confidence >= 0.8

    def test_allows_upright_text(self, default_filter):
        """Test that upright text is allowed."""
        word = {"text": "Normal", "upright": True}
        candidate = default_filter._check_word(word, {})
        assert candidate is None

    def test_rotation_filtering_disabled(self):
        """Test that rotation filtering can be disabled."""
        filter = WatermarkFilter(filter_rotated=False)
        word = {"text": "Rotated", "upright": False}
        candidate = filter._check_word(word, {})
        # Should not be flagged for rotation alone
        assert candidate is None


# =============================================================================
# Filter Words Tests
# =============================================================================

class TestFilterWords:
    """Tests for the filter_words method."""

    def test_filter_empty_list(self, default_filter):
        """Test filtering an empty list."""
        result = default_filter.filter_words([])
        assert result.filtered_words == []
        assert result.removed_candidates == []
        assert result.removal_count == 0

    def test_filter_no_watermarks(self, default_filter, sample_words):
        """Test filtering words with no watermarks."""
        result = default_filter.filter_words(sample_words)
        assert len(result.filtered_words) == len(sample_words)
        assert result.removal_count == 0

    def test_filter_removes_watermark(self, default_filter, words_with_watermark):
        """Test that watermarks are removed."""
        result = default_filter.filter_words(words_with_watermark)
        assert len(result.filtered_words) == 2
        assert result.removal_count == 1
        assert result.removed_candidates[0].text == "DRAFT"

    def test_filter_removes_rotated_text(self, default_filter, rotated_words):
        """Test that rotated text is removed."""
        result = default_filter.filter_words(rotated_words)
        assert len(result.filtered_words) == 2
        assert result.removal_count == 1
        # Check that "Rotated" was removed
        removed_texts = [c.text for c in result.removed_candidates]
        assert "Rotated" in removed_texts

    def test_filter_preserves_order(self, default_filter, sample_words):
        """Test that filtered words maintain order."""
        result = default_filter.filter_words(sample_words)
        for i, word in enumerate(result.filtered_words):
            assert word == sample_words[i]


# =============================================================================
# is_watermark Tests
# =============================================================================

class TestIsWatermark:
    """Tests for the is_watermark method."""

    def test_is_watermark_rotated(self, default_filter):
        """Test is_watermark with rotated text."""
        word = {"text": "SomeText", "upright": False}
        assert default_filter.is_watermark(word) is True

    def test_is_watermark_pattern(self, default_filter):
        """Test is_watermark with pattern match."""
        word = {"text": "DRAFT", "upright": True}
        assert default_filter.is_watermark(word) is True

    def test_is_watermark_normal(self, default_filter):
        """Test is_watermark with normal text."""
        word = {"text": "Date", "upright": True}
        assert default_filter.is_watermark(word) is False


# =============================================================================
# Custom Patterns Tests
# =============================================================================

class TestCustomPatterns:
    """Tests for custom pattern matching."""

    def test_custom_pattern_match(self):
        """Test that custom patterns are matched."""
        filter = WatermarkFilter(custom_patterns=[r"INTERNAL\s+USE"])
        word = {"text": "INTERNAL USE", "upright": True}
        candidate = filter._check_word(word, {})
        assert candidate is not None
        assert "custom_pattern" in candidate.reasons

    def test_custom_pattern_regex(self):
        """Test regex custom patterns."""
        filter = WatermarkFilter(custom_patterns=[r"REF-\d{6}"])
        word = {"text": "REF-123456", "upright": True}
        candidate = filter._check_word(word, {})
        assert candidate is not None

    def test_multiple_custom_patterns(self):
        """Test multiple custom patterns."""
        filter = WatermarkFilter(custom_patterns=[
            r"PATTERN1",
            r"PATTERN2",
        ])
        word1 = {"text": "PATTERN1", "upright": True}
        word2 = {"text": "PATTERN2", "upright": True}
        assert filter._check_word(word1, {}) is not None
        assert filter._check_word(word2, {}) is not None


# =============================================================================
# Font Size Tests
# =============================================================================

class TestFontSizeFiltering:
    """Tests for font size-based filtering."""

    def test_large_font_detected(self):
        """Test that large fonts are detected with char data."""
        filter = WatermarkFilter(max_font_size=30.0)
        chars = [
            {"x0": 100.0, "top": 200.0, "size": 50.0},  # Large font
        ]
        word = {"text": "LARGE", "x0": 100.0, "top": 200.0, "upright": True}
        font_sizes = filter._build_font_size_lookup(chars)
        candidate = filter._check_word(word, font_sizes)
        assert candidate is not None
        assert any("large_font" in r for r in candidate.reasons)

    def test_normal_font_allowed(self):
        """Test that normal fonts pass."""
        filter = WatermarkFilter(max_font_size=30.0)
        chars = [
            {"x0": 100.0, "top": 200.0, "size": 12.0},  # Normal font
        ]
        word = {"text": "Normal", "x0": 100.0, "top": 200.0, "upright": True}
        font_sizes = filter._build_font_size_lookup(chars)
        candidate = filter._check_word(word, font_sizes)
        assert candidate is None

    def test_font_size_disabled_by_default(self):
        """Test that font size filtering is disabled by default."""
        filter = WatermarkFilter()
        assert filter.max_font_size is None


# =============================================================================
# Convenience Function Tests
# =============================================================================

class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_filter_watermarks_function(self, words_with_watermark):
        """Test the filter_watermarks convenience function."""
        filtered = filter_watermarks(words_with_watermark)
        assert len(filtered) == 2
        # DRAFT should be removed
        texts = [w["text"] for w in filtered]
        assert "DRAFT" not in texts

    def test_is_likely_watermark_function(self):
        """Test the is_likely_watermark convenience function."""
        assert is_likely_watermark({"text": "DRAFT", "upright": True}) is True
        assert is_likely_watermark({"text": "Normal", "upright": True}) is False
        assert is_likely_watermark({"text": "Rotated", "upright": False}) is True


# =============================================================================
# Edge Cases Tests
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_text(self, default_filter):
        """Test handling of empty text."""
        word = {"text": "", "upright": True}
        candidate = default_filter._check_word(word, {})
        assert candidate is None

    def test_whitespace_only_text(self, default_filter):
        """Test handling of whitespace-only text."""
        word = {"text": "   ", "upright": True}
        candidate = default_filter._check_word(word, {})
        assert candidate is None

    def test_missing_upright_field(self, default_filter):
        """Test handling of missing upright field."""
        word = {"text": "Normal"}  # No upright field
        candidate = default_filter._check_word(word, {})
        # Should default to upright=True
        assert candidate is None

    def test_watermark_with_multiple_reasons(self, default_filter):
        """Test watermark with multiple detection reasons."""
        # Both rotated AND matches pattern
        word = {"text": "DRAFT", "upright": False}
        candidate = default_filter._check_word(word, {})
        assert candidate is not None
        assert len(candidate.reasons) >= 2
        assert "rotated_text" in candidate.reasons
        assert "watermark_pattern" in candidate.reasons


# =============================================================================
# WatermarkCandidate Tests
# =============================================================================

class TestWatermarkCandidate:
    """Tests for WatermarkCandidate dataclass."""

    def test_text_property(self):
        """Test text property extraction."""
        candidate = WatermarkCandidate(
            word={"text": "DRAFT", "x0": 100},
            reasons=["pattern"],
            confidence=0.9,
        )
        assert candidate.text == "DRAFT"

    def test_empty_text_property(self):
        """Test text property with missing text."""
        candidate = WatermarkCandidate(
            word={"x0": 100},  # No text field
            reasons=[],
            confidence=0.0,
        )
        assert candidate.text == ""


# =============================================================================
# FilterResult Tests
# =============================================================================

class TestFilterResult:
    """Tests for FilterResult dataclass."""

    def test_removal_count_property(self):
        """Test removal_count property."""
        result = FilterResult(
            filtered_words=[{"text": "a"}, {"text": "b"}],
            removed_candidates=[
                WatermarkCandidate(word={"text": "c"}, reasons=["test"]),
                WatermarkCandidate(word={"text": "d"}, reasons=["test"]),
            ],
        )
        assert result.removal_count == 2

    def test_empty_result(self):
        """Test empty result."""
        result = FilterResult()
        assert result.filtered_words == []
        assert result.removed_candidates == []
        assert result.removal_count == 0
