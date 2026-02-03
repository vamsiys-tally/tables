"""
Unit tests for header keywords module.

Tests cover:
- Keyword normalization
- Semantic type detection from text
- Header row identification
- Multi-word and abbreviated keywords
"""

import pytest

from tables.detector.keywords import (
    HEADER_KEYWORDS,
    SEMANTIC_TYPES,
    normalize_header_text,
    get_keywords_for_type,
    find_semantic_type,
    is_likely_header_row,
    HEADER_EXCLUSIONS,
)


class TestNormalizeHeaderText:
    """Tests for the normalize_header_text function."""

    def test_lowercase_conversion(self):
        """Test that text is converted to lowercase."""
        assert normalize_header_text("DATE") == "date"
        assert normalize_header_text("Transaction Date") == "transaction date"

    def test_whitespace_normalization(self):
        """Test that multiple whitespaces are collapsed."""
        assert normalize_header_text("  Transaction   DATE  ") == "transaction date"
        assert normalize_header_text("Debit\t\tAmount") == "debit amount"

    def test_empty_and_none_handling(self):
        """Test handling of empty strings."""
        assert normalize_header_text("") == ""
        assert normalize_header_text("   ") == ""

    def test_preserves_parentheses(self):
        """Test that parentheses are preserved."""
        # Period inside parentheses is preserved
        assert normalize_header_text("Debit (Dr.)") == "debit (dr.)"
        assert normalize_header_text("Amount (INR)") == "amount (inr)"

    def test_removes_trailing_periods(self):
        """Test that trailing periods are removed."""
        assert normalize_header_text("Sr. No.") == "sr. no"  # Internal periods kept


class TestGetKeywordsForType:
    """Tests for the get_keywords_for_type function."""

    def test_valid_types(self):
        """Test that valid types return keyword lists."""
        date_keywords = get_keywords_for_type("date")
        assert isinstance(date_keywords, list)
        assert len(date_keywords) > 0
        assert "date" in date_keywords
        assert "transaction date" in date_keywords

    def test_invalid_type(self):
        """Test that invalid types return empty list."""
        assert get_keywords_for_type("invalid_type") == []
        assert get_keywords_for_type("") == []

    def test_all_semantic_types_have_keywords(self):
        """Test that all defined semantic types have keywords."""
        for semantic_type in SEMANTIC_TYPES.keys():
            keywords = get_keywords_for_type(semantic_type)
            assert len(keywords) > 0, f"No keywords for type: {semantic_type}"


class TestFindSemanticType:
    """Tests for the find_semantic_type function."""

    def test_exact_match_high_confidence(self):
        """Test that exact matches return high confidence."""
        result = find_semantic_type("date")
        assert result is not None
        semantic_type, confidence = result
        assert semantic_type == "date"
        assert confidence == 1.0

    def test_contains_match(self):
        """Test matching when text contains keyword."""
        result = find_semantic_type("Transaction Date Column")
        assert result is not None
        semantic_type, confidence = result
        assert semantic_type == "date"
        assert 0.7 <= confidence < 1.0

    def test_common_header_matches(self):
        """Test detection of common bank statement headers."""
        test_cases = [
            ("Date", "date"),
            ("Txn Date", "date"),
            ("Particulars", "description"),
            ("Description", "description"),
            ("Narration", "description"),
            ("Debit", "debit"),
            ("Withdrawal", "debit"),
            ("Dr", "debit"),
            ("Credit", "credit"),
            ("Deposit", "credit"),
            ("Cr", "credit"),
            ("Balance", "balance"),
            ("Closing Balance", "balance"),
            ("Reference", "reference"),
            ("Chq No", "reference"),
            ("Amount", "amount"),
            ("Sl No", "serial"),
        ]

        for text, expected_type in test_cases:
            result = find_semantic_type(text)
            assert result is not None, f"No match for: {text}"
            semantic_type, _ = result
            assert semantic_type == expected_type, f"Wrong type for {text}: got {semantic_type}, expected {expected_type}"

    def test_no_match_returns_none(self):
        """Test that unrecognized text returns None."""
        result = find_semantic_type("xyz123abc", threshold=0.3)
        assert result is None

    def test_threshold_filtering(self):
        """Test that threshold filters low-confidence matches."""
        # This should match but with low confidence
        result = find_semantic_type("partial date info", threshold=0.9)
        assert result is None  # Below threshold

    def test_case_insensitive(self):
        """Test that matching is case-insensitive."""
        result1 = find_semantic_type("DATE")
        result2 = find_semantic_type("date")
        result3 = find_semantic_type("Date")

        assert result1 is not None and result2 is not None and result3 is not None
        assert result1[0] == result2[0] == result3[0]


class TestIsLikelyHeaderRow:
    """Tests for the is_likely_header_row function."""

    def test_typical_header_row(self):
        """Test detection of typical bank statement header row."""
        headers = ["Date", "Description", "Debit", "Credit", "Balance"]
        assert is_likely_header_row(headers) is True

    def test_header_with_abbreviations(self):
        """Test detection with abbreviated headers."""
        headers = ["Txn Date", "Particulars", "Dr", "Cr", "Bal"]
        assert is_likely_header_row(headers) is True

    def test_data_row_not_header(self):
        """Test that data rows are not identified as headers."""
        data_row = ["01/01/2024", "ATM Withdrawal", "500.00", "", "10000.00"]
        assert is_likely_header_row(data_row) is False

    def test_partial_header(self):
        """Test rows with some but not enough header matches."""
        partial = ["Date", "Some Text", "More Text", "Random"]
        # Only one match, default min is 2
        assert is_likely_header_row(partial, min_matches=2) is False

    def test_min_matches_parameter(self):
        """Test the min_matches parameter."""
        headers = ["Date", "Balance"]
        assert is_likely_header_row(headers, min_matches=2) is True
        assert is_likely_header_row(headers, min_matches=3) is False

    def test_exclusions_filtered(self):
        """Test that exclusion patterns are filtered out."""
        # "Opening Balance" should be excluded as it's a summary row indicator
        row_with_exclusion = ["Opening Balance", "Total", "Summary"]
        assert is_likely_header_row(row_with_exclusion) is False


class TestKeywordCompleteness:
    """Tests for keyword coverage and completeness."""

    def test_all_semantic_types_defined(self):
        """Test that all expected semantic types are defined."""
        expected_types = {"date", "description", "reference", "debit", "credit", "balance", "amount", "serial"}
        actual_types = set(HEADER_KEYWORDS.keys())
        assert expected_types <= actual_types

    def test_keywords_are_lowercase(self):
        """Test that all keywords are lowercase."""
        for semantic_type, keywords in HEADER_KEYWORDS.items():
            for keyword in keywords:
                assert keyword == keyword.lower(), f"Keyword '{keyword}' in {semantic_type} is not lowercase"

    def test_no_duplicate_keywords_within_type(self):
        """Test that there are no duplicate keywords within a type."""
        for semantic_type, keywords in HEADER_KEYWORDS.items():
            unique = set(keywords)
            assert len(unique) == len(keywords), f"Duplicate keywords in {semantic_type}"
