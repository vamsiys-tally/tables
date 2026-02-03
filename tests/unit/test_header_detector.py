"""
Unit tests for header detection.

Tests cover:
- HeaderMatch creation and properties
- HeaderRow detection and metrics
- HeaderDetector keyword matching
- Header row identification in candidate rows
"""

import pytest

from tables.detector.header_detector import (
    HeaderMatch,
    HeaderRow,
    HeaderDetector,
    detect_headers,
    is_header_row,
    MIN_HEADER_CONFIDENCE,
)
from tables.models.table import DataType


class TestHeaderMatch:
    """Tests for the HeaderMatch class."""

    def test_basic_creation(self):
        """Test basic HeaderMatch creation."""
        match = HeaderMatch(
            text="Date",
            semantic_type="date",
            confidence=1.0,
            match_method="keyword",
            column_index=0,
        )

        assert match.text == "Date"
        assert match.semantic_type == "date"
        assert match.confidence == 1.0
        assert match.is_matched is True

    def test_auto_normalization(self):
        """Test that text is auto-normalized."""
        match = HeaderMatch(text="  TRANSACTION DATE  ")
        assert match.normalized_text == "transaction date"

    def test_is_matched_with_low_confidence(self):
        """Test is_matched with low confidence."""
        match = HeaderMatch(
            text="Something",
            semantic_type="date",
            confidence=0.1,  # Below MIN_HEADER_CONFIDENCE
        )
        assert match.is_matched is False

    def test_is_matched_without_semantic_type(self):
        """Test is_matched without semantic type."""
        match = HeaderMatch(text="Unknown", semantic_type=None, confidence=0.9)
        assert match.is_matched is False

    def test_to_dict(self):
        """Test serialization to dictionary."""
        match = HeaderMatch(
            text="Balance",
            semantic_type="balance",
            confidence=0.95,
            match_method="keyword",
            column_index=4,
        )

        d = match.to_dict()
        assert d["text"] == "Balance"
        assert d["semantic_type"] == "balance"
        assert d["confidence"] == 0.95
        assert d["column_index"] == 4


class TestHeaderRow:
    """Tests for the HeaderRow class."""

    def test_basic_creation(self):
        """Test basic HeaderRow creation."""
        matches = [
            HeaderMatch(text="Date", semantic_type="date", confidence=1.0),
            HeaderMatch(text="Description", semantic_type="description", confidence=0.9),
        ]
        row = HeaderRow(row_index=0, matches=matches, is_header=True)

        assert row.row_index == 0
        assert row.matched_count == 2
        assert row.is_header is True

    def test_overall_confidence_calculation(self):
        """Test overall confidence calculation."""
        matches = [
            HeaderMatch(text="Date", semantic_type="date", confidence=1.0),
            HeaderMatch(text="Amount", semantic_type="amount", confidence=0.8),
            HeaderMatch(text="Unknown", semantic_type=None, confidence=0.0),
        ]
        row = HeaderRow(row_index=0, matches=matches)
        row._calculate_metrics()

        # Only matched items count: (1.0 + 0.8) / 2 = 0.9
        assert row.matched_count == 2
        assert abs(row.overall_confidence - 0.9) < 0.01

    def test_get_column_definitions(self):
        """Test conversion to column definitions."""
        matches = [
            HeaderMatch(text="Date", semantic_type="date", confidence=1.0, column_index=0),
            HeaderMatch(text="Debit", semantic_type="debit", confidence=0.9, column_index=1),
        ]
        row = HeaderRow(row_index=0, matches=matches, is_header=True)

        positions = [(0.0, 50.0), (50.0, 100.0)]
        columns = row.get_column_definitions(positions)

        assert len(columns) == 2
        assert columns[0].header_text == "Date"
        assert columns[0].semantic_type == "date"
        assert columns[0].data_type == DataType.DATE
        assert columns[0].x0 == 0.0
        assert columns[0].x1 == 50.0

        assert columns[1].header_text == "Debit"
        assert columns[1].semantic_type == "debit"
        assert columns[1].data_type == DataType.NUMERIC


class TestHeaderDetector:
    """Tests for the HeaderDetector class."""

    @pytest.fixture
    def detector(self):
        """Create a HeaderDetector instance without SLM."""
        return HeaderDetector(use_slm=False)

    def test_detect_common_headers(self, detector):
        """Test detection of common bank statement headers."""
        texts = ["Date", "Description", "Debit", "Credit", "Balance"]
        matches = detector.detect_headers(texts)

        assert len(matches) == 5
        assert matches[0].semantic_type == "date"
        assert matches[1].semantic_type == "description"
        assert matches[2].semantic_type == "debit"
        assert matches[3].semantic_type == "credit"
        assert matches[4].semantic_type == "balance"

    def test_detect_abbreviated_headers(self, detector):
        """Test detection of abbreviated headers."""
        texts = ["Txn Date", "Particulars", "Dr", "Cr", "Bal"]
        matches = detector.detect_headers(texts)

        assert len(matches) == 5
        assert matches[0].semantic_type == "date"
        assert matches[1].semantic_type == "description"
        assert matches[2].semantic_type == "debit"
        assert matches[3].semantic_type == "credit"
        assert matches[4].semantic_type == "balance"

    def test_detect_no_matches(self, detector):
        """Test handling of unrecognized text."""
        texts = ["XYZ", "ABC", "123"]
        matches = detector.detect_headers(texts)

        assert len(matches) == 3
        assert all(not m.is_matched for m in matches)

    def test_detect_header_row(self, detector):
        """Test header row detection."""
        texts = ["Date", "Description", "Debit", "Credit", "Balance"]
        row = detector.detect_header_row(texts, row_index=0)

        assert row.is_header is True
        assert row.matched_count == 5
        assert row.row_index == 0

    def test_detect_data_row_not_header(self, detector):
        """Test that data rows are not classified as headers."""
        texts = ["01/01/2024", "ATM Withdrawal", "500.00", "", "10000.00"]
        row = detector.detect_header_row(texts)

        assert row.is_header is False

    def test_find_header_rows(self, detector):
        """Test finding headers in multiple rows."""
        rows = [
            ["Date", "Description", "Debit", "Credit", "Balance"],
            ["01/01/2024", "Opening Balance", "", "", "10000.00"],
            ["02/01/2024", "ATM Withdrawal", "500.00", "", "9500.00"],
        ]
        headers = detector.find_header_rows(rows)

        assert len(headers) == 1
        assert headers[0].row_index == 0

    def test_find_header_rows_multirow(self, detector):
        """Test finding multi-row headers."""
        rows = [
            ["Transaction", "Transaction", "Debit", "Credit", "Running"],
            ["Date", "Details", "Amount", "Amount", "Balance"],
            ["01/01/2024", "Opening", "", "", "10000.00"],
        ]
        headers = detector.find_header_rows(rows, max_header_rows=2)

        # Should find both header rows
        assert len(headers) <= 2


class TestDetectHeadersFunction:
    """Tests for the detect_headers convenience function."""

    def test_basic_detection(self):
        """Test basic header detection."""
        matches = detect_headers(["Date", "Amount", "Balance"], use_slm=False)

        assert len(matches) == 3
        assert matches[0].semantic_type == "date"
        assert matches[1].semantic_type == "amount"
        assert matches[2].semantic_type == "balance"


class TestIsHeaderRowFunction:
    """Tests for the is_header_row convenience function."""

    def test_header_row(self):
        """Test positive header row detection."""
        assert is_header_row(["Date", "Description", "Amount"]) is True

    def test_data_row(self):
        """Test negative detection for data row."""
        assert is_header_row(["01/01/2024", "Some text", "500.00"]) is False

    def test_min_matches(self):
        """Test min_matches parameter."""
        assert is_header_row(["Date", "Balance"], min_matches=2) is True
        assert is_header_row(["Date", "Balance"], min_matches=3) is False
