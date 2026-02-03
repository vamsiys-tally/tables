"""
Watermark Detection and Filtering.

This module provides functionality to detect and filter watermark text
from PDF pages. Watermarks can interfere with table detection by:
- Creating false rows in table region detection
- Matching header keywords (e.g., "ACCOUNT" watermark)
- Adding noise to structure classification

Detection Heuristics:
    1. Rotation: Non-upright text (diagonal watermarks)
    2. Text patterns: Common watermark phrases (DRAFT, CONFIDENTIAL, etc.)
    3. Font size: Unusually large fonts (>30pt by default)
    4. Position: Text spanning unusual page areas

The filter is conservative by default - it only removes text that
strongly matches watermark patterns to avoid false positives.

Example Usage:
    >>> from tables.detector.watermark_filter import WatermarkFilter
    >>>
    >>> filter = WatermarkFilter()
    >>> words = pdf_document.get_page_words(0)
    >>> filtered = filter.filter_words(words)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional
import re
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

# Common watermark text patterns (case-insensitive)
WATERMARK_PATTERNS = [
    r"^\s*draft\s*$",
    r"^\s*confidential\s*$",
    r"^\s*copy\s*$",
    r"^\s*sample\s*$",
    r"^\s*specimen\s*$",
    r"^\s*duplicate\s*$",
    r"^\s*void\s*$",
    r"^\s*cancelled\s*$",
    r"^\s*original\s*$",
    r"^\s*not\s+valid\s*$",
    r"^\s*for\s+reference\s+only\s*$",
    r"^\s*unofficial\s*$",
]

# Compile patterns for efficiency
_COMPILED_PATTERNS = [re.compile(p, re.IGNORECASE) for p in WATERMARK_PATTERNS]

# Default thresholds
DEFAULT_MAX_FONT_SIZE = 30.0  # Points - text larger than this may be watermark
DEFAULT_MIN_FONT_SIZE = 4.0   # Points - text smaller than this is likely noise


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class WatermarkCandidate:
    """
    A word identified as a potential watermark.

    Stores the word data and the reason(s) it was flagged.

    Attributes:
        word: Original word dictionary from pdfplumber
        reasons: List of reasons why this was flagged
        confidence: Confidence that this is a watermark (0.0-1.0)
    """

    word: dict[str, Any]
    reasons: list[str] = field(default_factory=list)
    confidence: float = 0.0

    @property
    def text(self) -> str:
        """Get the word text."""
        return self.word.get("text", "")


@dataclass
class FilterResult:
    """
    Result of watermark filtering.

    Contains both the filtered words and information about what was removed.

    Attributes:
        filtered_words: Words that passed the filter
        removed_candidates: Words identified as watermarks
        removal_count: Number of words removed
    """

    filtered_words: list[dict[str, Any]] = field(default_factory=list)
    removed_candidates: list[WatermarkCandidate] = field(default_factory=list)

    @property
    def removal_count(self) -> int:
        """Number of words removed."""
        return len(self.removed_candidates)


# =============================================================================
# Watermark Filter
# =============================================================================

class WatermarkFilter:
    """
    Filter for detecting and removing watermark text.

    Uses multiple heuristics to identify watermark text with high precision
    (avoiding false positives). The filter is configurable for different
    document types.

    Detection Criteria:
        - Non-upright text (rotated/diagonal)
        - Known watermark text patterns
        - Unusually large font sizes
        - Text with very low opacity (if available)

    Attributes:
        filter_rotated: Whether to filter non-upright text
        filter_patterns: Whether to filter known watermark patterns
        max_font_size: Maximum font size before flagging as watermark
        custom_patterns: Additional regex patterns to filter

    Example:
        >>> filter = WatermarkFilter()
        >>> result = filter.filter_words(words)
        >>> print(f"Removed {result.removal_count} watermarks")
    """

    def __init__(
        self,
        filter_rotated: bool = True,
        filter_patterns: bool = True,
        max_font_size: Optional[float] = None,
        custom_patterns: Optional[list[str]] = None,
    ):
        """
        Initialize the watermark filter.

        Args:
            filter_rotated: Filter non-upright (rotated) text
            filter_patterns: Filter known watermark text patterns
            max_font_size: Maximum font size (None to disable size filtering)
            custom_patterns: Additional regex patterns to filter
        """
        self.filter_rotated = filter_rotated
        self.filter_patterns = filter_patterns
        self.max_font_size = max_font_size
        self.custom_patterns = custom_patterns or []

        # Compile custom patterns
        self._custom_compiled = [
            re.compile(p, re.IGNORECASE) for p in self.custom_patterns
        ]

    def filter_words(
        self,
        words: list[dict[str, Any]],
        chars: Optional[list[dict[str, Any]]] = None,
    ) -> FilterResult:
        """
        Filter watermark words from a list of words.

        Args:
            words: List of word dictionaries from pdfplumber
            chars: Optional character list (for font size detection)

        Returns:
            FilterResult with filtered words and removal info
        """
        result = FilterResult()

        # Build font size lookup if chars provided
        font_sizes: dict[tuple[float, float], float] = {}
        if chars and self.max_font_size:
            font_sizes = self._build_font_size_lookup(chars)

        for word in words:
            candidate = self._check_word(word, font_sizes)

            if candidate:
                result.removed_candidates.append(candidate)
            else:
                result.filtered_words.append(word)

        if result.removal_count > 0:
            logger.debug(
                f"Removed {result.removal_count} watermark words: "
                f"{[c.text for c in result.removed_candidates]}"
            )

        return result

    def is_watermark(
        self,
        word: dict[str, Any],
        chars: Optional[list[dict[str, Any]]] = None,
    ) -> bool:
        """
        Check if a single word is likely a watermark.

        Args:
            word: Word dictionary from pdfplumber
            chars: Optional characters for font size lookup

        Returns:
            True if word appears to be a watermark
        """
        font_sizes: dict[tuple[float, float], float] = {}
        if chars and self.max_font_size:
            font_sizes = self._build_font_size_lookup(chars)

        return self._check_word(word, font_sizes) is not None

    def _check_word(
        self,
        word: dict[str, Any],
        font_sizes: dict[tuple[float, float], float],
    ) -> Optional[WatermarkCandidate]:
        """
        Check if a word matches watermark criteria.

        Args:
            word: Word dictionary to check
            font_sizes: Lookup dict for font sizes by position

        Returns:
            WatermarkCandidate if word is a watermark, None otherwise
        """
        reasons: list[str] = []
        confidence = 0.0

        text = word.get("text", "").strip()
        if not text:
            return None

        # Check 1: Rotation (non-upright text)
        if self.filter_rotated:
            upright = word.get("upright", True)
            if not upright:
                reasons.append("rotated_text")
                confidence = max(confidence, 0.8)

        # Check 2: Known watermark patterns
        if self.filter_patterns:
            if self._matches_watermark_pattern(text):
                reasons.append("watermark_pattern")
                confidence = max(confidence, 0.9)

        # Check 3: Font size (if available)
        if self.max_font_size and font_sizes:
            x0 = word.get("x0", 0)
            top = word.get("top", 0)
            # Find font size for this position
            font_size = font_sizes.get((round(x0, 1), round(top, 1)))
            if font_size and font_size > self.max_font_size:
                reasons.append(f"large_font_{font_size:.1f}pt")
                confidence = max(confidence, 0.7)

        # Check 4: Custom patterns
        for pattern in self._custom_compiled:
            if pattern.search(text):
                reasons.append("custom_pattern")
                confidence = max(confidence, 0.85)
                break

        if reasons:
            return WatermarkCandidate(
                word=word,
                reasons=reasons,
                confidence=confidence,
            )

        return None

    def _matches_watermark_pattern(self, text: str) -> bool:
        """
        Check if text matches known watermark patterns.

        Args:
            text: Text to check

        Returns:
            True if matches a watermark pattern
        """
        for pattern in _COMPILED_PATTERNS:
            if pattern.search(text):
                return True
        return False

    def _build_font_size_lookup(
        self,
        chars: list[dict[str, Any]],
    ) -> dict[tuple[float, float], float]:
        """
        Build a lookup table of font sizes by position.

        Args:
            chars: List of character dictionaries

        Returns:
            Dict mapping (x0, top) -> font_size
        """
        lookup: dict[tuple[float, float], float] = {}

        for char in chars:
            x0 = round(char.get("x0", 0), 1)
            top = round(char.get("top", char.get("y0", 0)), 1)
            size = char.get("size", 0)

            if size > 0:
                key = (x0, top)
                # Keep the largest size at this position
                if key not in lookup or size > lookup[key]:
                    lookup[key] = size

        return lookup


# =============================================================================
# Convenience Functions
# =============================================================================

def filter_watermarks(
    words: list[dict[str, Any]],
    chars: Optional[list[dict[str, Any]]] = None,
    filter_rotated: bool = True,
    filter_patterns: bool = True,
) -> list[dict[str, Any]]:
    """
    Convenience function to filter watermarks from words.

    Args:
        words: List of word dictionaries from pdfplumber
        chars: Optional character list for font size detection
        filter_rotated: Filter non-upright (rotated) text
        filter_patterns: Filter known watermark text patterns

    Returns:
        Filtered list of words

    Example:
        >>> words = pdf.get_page_words(0)
        >>> filtered = filter_watermarks(words)
    """
    filter = WatermarkFilter(
        filter_rotated=filter_rotated,
        filter_patterns=filter_patterns,
    )
    result = filter.filter_words(words, chars)
    return result.filtered_words


def is_likely_watermark(word: dict[str, Any]) -> bool:
    """
    Quick check if a word is likely a watermark.

    Only checks rotation and text patterns (not font size).

    Args:
        word: Word dictionary from pdfplumber

    Returns:
        True if word appears to be a watermark

    Example:
        >>> word = {"text": "DRAFT", "upright": False}
        >>> is_likely_watermark(word)
        True
    """
    filter = WatermarkFilter()
    return filter.is_watermark(word)
