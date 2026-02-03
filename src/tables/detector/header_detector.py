"""
Header Detection for Table Columns.

This module provides functionality to identify and classify table headers
in bank statement PDFs. It uses a two-tier approach:

1. Primary: Keyword-based matching (deterministic, fast)
2. Fallback: SLM-based semantic similarity (for ambiguous cases)

The header detector identifies:
- Which row(s) contain headers
- The semantic type of each header (date, description, debit, credit, etc.)
- Confidence scores for each classification

Design Principles:
    - Keyword matching is preferred for speed and reliability
    - SLM is only invoked when confidence is below threshold
    - No content-based validation (pure structural detection)

Example Usage:
    >>> from tables.detector.header_detector import HeaderDetector
    >>>
    >>> detector = HeaderDetector()
    >>> matches = detector.detect_headers(["Date", "Particulars", "Debit", "Credit"])
    >>> for match in matches:
    ...     print(f"{match.text} -> {match.semantic_type} ({match.confidence:.2f})")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Any
import logging

from tables.detector.keywords import (
    HEADER_KEYWORDS,
    normalize_header_text,
    find_semantic_type,
    is_likely_header_row,
    HEADER_EXCLUSIONS,
)
from tables.models.table import ColumnDefinition, DataType

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

# Confidence threshold below which SLM fallback is used
SLM_FALLBACK_THRESHOLD = 0.5

# Minimum confidence to accept a header match
MIN_HEADER_CONFIDENCE = 0.3

# SLM model name (sentence-transformers)
DEFAULT_SLM_MODEL = "all-MiniLM-L6-v2"


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class HeaderMatch:
    """
    Result of header detection for a single cell.

    Represents the classification of a potential header cell
    with its semantic type and confidence score.

    Attributes:
        text: Original header text
        normalized_text: Normalized text used for matching
        semantic_type: Detected semantic type (date, debit, etc.)
        confidence: Confidence score (0.0-1.0)
        match_method: Method used for matching (keyword, slm, none)
        column_index: Index of this header in the row

    Example:
        >>> match = HeaderMatch(
        ...     text="Transaction Date",
        ...     normalized_text="transaction date",
        ...     semantic_type="date",
        ...     confidence=1.0,
        ...     match_method="keyword",
        ...     column_index=0,
        ... )
    """

    text: str
    normalized_text: str = ""
    semantic_type: Optional[str] = None
    confidence: float = 0.0
    match_method: str = "none"  # "keyword", "slm", "none"
    column_index: int = 0

    def __post_init__(self):
        """Normalize text if not already done."""
        if not self.normalized_text:
            self.normalized_text = normalize_header_text(self.text)

    @property
    def is_matched(self) -> bool:
        """Check if a semantic type was matched."""
        return self.semantic_type is not None and self.confidence >= MIN_HEADER_CONFIDENCE

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "text": self.text,
            "normalized_text": self.normalized_text,
            "semantic_type": self.semantic_type,
            "confidence": self.confidence,
            "match_method": self.match_method,
            "column_index": self.column_index,
        }


@dataclass
class HeaderRow:
    """
    Complete header row detection result.

    Contains all header matches for a potential header row
    along with overall metrics.

    Attributes:
        row_index: Index of this row (0-indexed)
        matches: List of HeaderMatch for each cell
        is_header: Whether this row is classified as a header
        overall_confidence: Average confidence of matched headers
        matched_count: Number of cells with valid semantic matches

    Example:
        >>> row = HeaderRow(
        ...     row_index=0,
        ...     matches=[match1, match2, match3],
        ...     is_header=True,
        ... )
    """

    row_index: int = 0
    matches: list[HeaderMatch] = field(default_factory=list)
    is_header: bool = False
    overall_confidence: float = 0.0
    matched_count: int = 0

    def __post_init__(self):
        """Calculate metrics if not provided."""
        if self.matches and self.matched_count == 0:
            self._calculate_metrics()

    def _calculate_metrics(self) -> None:
        """Calculate matched count and overall confidence."""
        matched = [m for m in self.matches if m.is_matched]
        self.matched_count = len(matched)

        if matched:
            self.overall_confidence = sum(m.confidence for m in matched) / len(matched)
        else:
            self.overall_confidence = 0.0

    def get_column_definitions(
        self,
        column_positions: list[tuple[float, float]],
    ) -> list[ColumnDefinition]:
        """
        Convert header matches to column definitions.

        Args:
            column_positions: List of (x0, x1) tuples for each column

        Returns:
            List of ColumnDefinition objects

        Raises:
            ValueError: If column_positions length doesn't match matches
        """
        if len(column_positions) != len(self.matches):
            raise ValueError(
                f"Column positions ({len(column_positions)}) must match "
                f"number of matches ({len(self.matches)})"
            )

        columns: list[ColumnDefinition] = []
        for i, (match, (x0, x1)) in enumerate(zip(self.matches, column_positions)):
            # Determine data type from semantic type
            data_type = DataType.UNKNOWN
            if match.semantic_type == "date":
                data_type = DataType.DATE
            elif match.semantic_type in ("debit", "credit", "balance", "amount"):
                data_type = DataType.NUMERIC
            elif match.semantic_type in ("description", "reference"):
                data_type = DataType.TEXT

            columns.append(ColumnDefinition(
                column_id=i,
                x0=x0,
                x1=x1,
                header_text=match.text,
                semantic_type=match.semantic_type,
                data_type=data_type,
                confidence=match.confidence,
                source_header_row=self.row_index,
            ))

        return columns

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "row_index": self.row_index,
            "matches": [m.to_dict() for m in self.matches],
            "is_header": self.is_header,
            "overall_confidence": self.overall_confidence,
            "matched_count": self.matched_count,
        }


# =============================================================================
# Header Detector
# =============================================================================

class HeaderDetector:
    """
    Detector for table headers using keyword matching and SLM fallback.

    The detector analyzes text cells to identify header rows and
    classify each header's semantic type (date, description, debit, etc.).

    Detection Strategy:
        1. Normalize and clean header text
        2. Attempt keyword-based matching
        3. If confidence < threshold, use SLM fallback (if enabled)
        4. Aggregate results and determine if row is a header

    Attributes:
        slm_threshold: Confidence threshold for SLM fallback
        use_slm: Whether to enable SLM fallback
        slm_model: Sentence transformer model (lazy loaded)

    Example:
        >>> detector = HeaderDetector(use_slm=True)
        >>> row = detector.detect_header_row(["Date", "Particulars", "Amount"])
        >>> if row.is_header:
        ...     print(f"Found header with {row.matched_count} matches")
    """

    def __init__(
        self,
        slm_threshold: float = SLM_FALLBACK_THRESHOLD,
        use_slm: bool = True,
        min_matches_for_header: int = 2,
    ):
        """
        Initialize the header detector.

        Args:
            slm_threshold: Confidence below which SLM fallback is used
            use_slm: Whether to enable SLM-based fallback
            min_matches_for_header: Minimum semantic matches for header row
        """
        self.slm_threshold = slm_threshold
        self.use_slm = use_slm
        self.min_matches_for_header = min_matches_for_header
        self._slm_model = None
        self._header_embeddings: Optional[dict[str, Any]] = None

    @property
    def slm_model(self):
        """Lazy-load the SLM model."""
        if self._slm_model is None and self.use_slm:
            try:
                from sentence_transformers import SentenceTransformer
                logger.info(f"Loading SLM model: {DEFAULT_SLM_MODEL}")
                self._slm_model = SentenceTransformer(DEFAULT_SLM_MODEL)
                self._precompute_header_embeddings()
            except ImportError:
                logger.warning(
                    "sentence-transformers not installed. SLM fallback disabled."
                )
                self.use_slm = False
            except Exception as e:
                logger.warning(f"Failed to load SLM model: {e}. SLM fallback disabled.")
                self.use_slm = False
        return self._slm_model

    def _precompute_header_embeddings(self) -> None:
        """
        Precompute embeddings for all known header keywords.

        This is done once when the model is loaded to speed up
        subsequent similarity comparisons.
        """
        if self._slm_model is None:
            return

        import numpy as np

        self._header_embeddings = {}

        for semantic_type, keywords in HEADER_KEYWORDS.items():
            # Embed all keywords for this type
            embeddings = self._slm_model.encode(keywords, convert_to_numpy=True)
            # Store mean embedding as representative
            self._header_embeddings[semantic_type] = {
                "keywords": keywords,
                "embeddings": embeddings,
                "mean_embedding": np.mean(embeddings, axis=0),
            }

        logger.info(f"Precomputed embeddings for {len(self._header_embeddings)} semantic types")

    def detect_headers(self, texts: list[str]) -> list[HeaderMatch]:
        """
        Detect headers from a list of text cells.

        This is the main entry point for header detection.
        Returns a HeaderMatch for each input text.

        Args:
            texts: List of potential header texts

        Returns:
            List of HeaderMatch objects (one per input text)

        Example:
            >>> matches = detector.detect_headers(["Date", "Description", "Amount"])
            >>> for m in matches:
            ...     print(f"{m.text}: {m.semantic_type} ({m.confidence:.2f})")
        """
        matches: list[HeaderMatch] = []

        for i, text in enumerate(texts):
            match = self._detect_single_header(text, column_index=i)
            matches.append(match)

        return matches

    def detect_header_row(
        self,
        texts: list[str],
        row_index: int = 0,
    ) -> HeaderRow:
        """
        Detect whether a row of texts constitutes a header row.

        Analyzes all cells in the row and determines if the row
        should be classified as a header based on match count.

        Args:
            texts: List of cell texts in the row
            row_index: Index of this row (for reference)

        Returns:
            HeaderRow with detection results

        Example:
            >>> row = detector.detect_header_row(["Date", "Particulars", "Dr", "Cr", "Balance"])
            >>> row.is_header
            True
            >>> row.matched_count
            5
        """
        matches = self.detect_headers(texts)

        header_row = HeaderRow(
            row_index=row_index,
            matches=matches,
        )
        header_row._calculate_metrics()

        # Determine if this is a header row
        header_row.is_header = (
            header_row.matched_count >= self.min_matches_for_header
        )

        return header_row

    def find_header_rows(
        self,
        rows: list[list[str]],
        max_header_rows: int = 2,
    ) -> list[HeaderRow]:
        """
        Find header rows from multiple candidate rows.

        Typically searches the first few rows of a table to find headers.
        Handles multi-row headers where header text spans multiple rows.

        Args:
            rows: List of rows, each row is a list of cell texts
            max_header_rows: Maximum number of header rows to find

        Returns:
            List of identified HeaderRow objects

        Example:
            >>> rows = [
            ...     ["Date", "Description", "Debit", "Credit", "Balance"],
            ...     ["01/01/24", "Opening Balance", "", "", "10000.00"],
            ... ]
            >>> headers = detector.find_header_rows(rows)
            >>> len(headers)
            1
        """
        header_rows: list[HeaderRow] = []

        for i, row in enumerate(rows[:max_header_rows + 2]):  # Check a few extra
            header_row = self.detect_header_row(row, row_index=i)

            if header_row.is_header:
                header_rows.append(header_row)

                # Stop if we've found enough
                if len(header_rows) >= max_header_rows:
                    break
            elif header_rows:
                # If we had header rows but this one isn't, stop searching
                break

        return header_rows

    def _detect_single_header(
        self,
        text: str,
        column_index: int = 0,
    ) -> HeaderMatch:
        """
        Detect semantic type for a single header text.

        Uses keyword matching first, then SLM fallback if needed.

        Args:
            text: Header text to classify
            column_index: Index of this column

        Returns:
            HeaderMatch with classification result
        """
        normalized = normalize_header_text(text)

        # Check for exclusions first
        if any(excl in normalized for excl in HEADER_EXCLUSIONS):
            return HeaderMatch(
                text=text,
                normalized_text=normalized,
                semantic_type=None,
                confidence=0.0,
                match_method="excluded",
                column_index=column_index,
            )

        # Try keyword matching
        keyword_result = find_semantic_type(text, threshold=0.0)

        if keyword_result:
            semantic_type, confidence = keyword_result

            # If confidence is high enough, return keyword match
            if confidence >= self.slm_threshold:
                return HeaderMatch(
                    text=text,
                    normalized_text=normalized,
                    semantic_type=semantic_type,
                    confidence=confidence,
                    match_method="keyword",
                    column_index=column_index,
                )

        # Try SLM fallback if enabled and keyword match was weak
        if self.use_slm:
            slm_result = self._slm_match(normalized)
            if slm_result:
                semantic_type, confidence = slm_result

                # Use SLM result if better than keyword result
                if keyword_result:
                    kw_type, kw_conf = keyword_result
                    if confidence > kw_conf:
                        return HeaderMatch(
                            text=text,
                            normalized_text=normalized,
                            semantic_type=semantic_type,
                            confidence=confidence,
                            match_method="slm",
                            column_index=column_index,
                        )
                    else:
                        return HeaderMatch(
                            text=text,
                            normalized_text=normalized,
                            semantic_type=kw_type,
                            confidence=kw_conf,
                            match_method="keyword",
                            column_index=column_index,
                        )
                else:
                    return HeaderMatch(
                        text=text,
                        normalized_text=normalized,
                        semantic_type=semantic_type,
                        confidence=confidence,
                        match_method="slm",
                        column_index=column_index,
                    )

        # Return keyword result if we have one (even if low confidence)
        if keyword_result:
            semantic_type, confidence = keyword_result
            return HeaderMatch(
                text=text,
                normalized_text=normalized,
                semantic_type=semantic_type,
                confidence=confidence,
                match_method="keyword",
                column_index=column_index,
            )

        # No match found
        return HeaderMatch(
            text=text,
            normalized_text=normalized,
            semantic_type=None,
            confidence=0.0,
            match_method="none",
            column_index=column_index,
        )

    def _slm_match(
        self,
        text: str,
        threshold: float = 0.4,
    ) -> Optional[tuple[str, float]]:
        """
        Match header text using SLM semantic similarity.

        Compares the text embedding against precomputed header embeddings
        to find the best semantic type match.

        Args:
            text: Normalized header text
            threshold: Minimum similarity threshold

        Returns:
            Tuple of (semantic_type, confidence) or None if no match
        """
        if self.slm_model is None or self._header_embeddings is None:
            return None

        if not text or len(text) < 2:
            return None

        try:
            import numpy as np
            from numpy.linalg import norm

            # Encode the input text
            text_embedding = self._slm_model.encode([text], convert_to_numpy=True)[0]

            best_type: Optional[str] = None
            best_similarity = 0.0

            # Compare against each semantic type
            for semantic_type, data in self._header_embeddings.items():
                # Calculate cosine similarity with mean embedding
                mean_emb = data["mean_embedding"]
                similarity = float(
                    np.dot(text_embedding, mean_emb) /
                    (norm(text_embedding) * norm(mean_emb))
                )

                if similarity > best_similarity:
                    best_similarity = similarity
                    best_type = semantic_type

            if best_type and best_similarity >= threshold:
                # Convert similarity to confidence (0.4-1.0 -> 0.3-0.8)
                confidence = 0.3 + (best_similarity - 0.4) * (0.5 / 0.6)
                confidence = min(0.8, max(0.3, confidence))  # Cap at 0.8 for SLM
                return (best_type, confidence)

        except Exception as e:
            logger.warning(f"SLM matching failed: {e}")

        return None


# =============================================================================
# Convenience Functions
# =============================================================================

def detect_headers(
    texts: list[str],
    use_slm: bool = False,
) -> list[HeaderMatch]:
    """
    Convenience function to detect headers.

    Args:
        texts: List of potential header texts
        use_slm: Whether to enable SLM fallback

    Returns:
        List of HeaderMatch objects

    Example:
        >>> matches = detect_headers(["Date", "Amount", "Balance"])
        >>> [m.semantic_type for m in matches if m.is_matched]
        ['date', 'amount', 'balance']
    """
    detector = HeaderDetector(use_slm=use_slm)
    return detector.detect_headers(texts)


def is_header_row(
    texts: list[str],
    min_matches: int = 2,
) -> bool:
    """
    Quick check if a row appears to be a header row.

    Args:
        texts: List of cell texts in the row
        min_matches: Minimum semantic matches required

    Returns:
        True if row appears to be a header

    Example:
        >>> is_header_row(["Date", "Description", "Amount"])
        True
        >>> is_header_row(["01/01/24", "ATM Withdrawal", "500.00"])
        False
    """
    return is_likely_header_row(texts, min_matches=min_matches)
