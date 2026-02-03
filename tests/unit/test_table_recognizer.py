"""
Unit tests for table recognizer and table merger.

Tests the main TableRecognizer orchestrator and cross-page merging.
"""

import pytest
from datetime import date
from decimal import Decimal
from unittest.mock import Mock, MagicMock, patch

from tables.recognizer.table_recognizer import (
    TableRecognizer,
    recognize_tables,
    extract_transactions,
)
from tables.recognizer.table_merger import (
    TableMerger,
    SemanticColumnMapper,
    PageBoundaryInfo,
    merge_table_pages,
)
from tables.recognizer.cell_extractor import Cell, ExtractedRow
from tables.models.table import (
    TableDefinition,
    ColumnDefinition,
    TableDetectionResult,
    DataType,
    ContentType,
)
from tables.utils.geometry import BoundingBox


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_columns():
    """Create sample column definitions."""
    return [
        ColumnDefinition(
            column_id=0,
            x0=50.0,
            x1=150.0,
            header_text="Date",
            semantic_type="date",
            data_type=DataType.DATE,
        ),
        ColumnDefinition(
            column_id=1,
            x0=150.0,
            x1=350.0,
            header_text="Description",
            semantic_type="description",
            data_type=DataType.TEXT,
        ),
        ColumnDefinition(
            column_id=2,
            x0=350.0,
            x1=450.0,
            header_text="Debit",
            semantic_type="debit",
            data_type=DataType.NUMERIC,
        ),
        ColumnDefinition(
            column_id=3,
            x0=450.0,
            x1=550.0,
            header_text="Balance",
            semantic_type="balance",
            data_type=DataType.NUMERIC,
        ),
    ]


@pytest.fixture
def single_page_table(sample_columns):
    """Create a single-page table definition."""
    return TableDefinition(
        table_id="single_page_table",
        page_numbers=[0],
        columns=sample_columns,
        bounds_per_page={0: BoundingBox(x0=50, y0=100, x1=550, y1=500)},
        content_type=ContentType.TRANSACTION,
        detection_confidence=0.9,
    )


@pytest.fixture
def multi_page_table(sample_columns):
    """Create a multi-page table definition."""
    return TableDefinition(
        table_id="multi_page_table",
        page_numbers=[0, 1, 2],
        columns=sample_columns,
        bounds_per_page={
            0: BoundingBox(x0=50, y0=100, x1=550, y1=700),
            1: BoundingBox(x0=50, y0=50, x1=550, y1=700),
            2: BoundingBox(x0=50, y0=50, x1=550, y1=500),
        },
        is_multi_page=True,
        header_repeats_on_pages=True,
        content_type=ContentType.TRANSACTION,
        detection_confidence=0.9,
    )


@pytest.fixture
def sample_detection_result(single_page_table):
    """Create a sample detection result."""
    return TableDetectionResult(
        tables=[single_page_table],
        pages_processed=[0],
        pages_with_tables=[0],
    )


def create_mock_pdf(pages_data):
    """Create a mock PDF document with specified page data."""
    mock = Mock()

    def get_page_words(page_num):
        return pages_data.get(page_num, [])

    def get_page_dimensions(page_num):
        return (612, 792)  # Standard letter size

    mock.get_page_words = get_page_words
    mock.get_page_dimensions = get_page_dimensions
    return mock


# =============================================================================
# SemanticColumnMapper Tests
# =============================================================================

class TestSemanticColumnMapper:
    """Tests for SemanticColumnMapper class."""

    def test_map_by_semantic_type(self, sample_columns):
        """Test mapping columns by semantic type."""
        mapper = SemanticColumnMapper(sample_columns)

        # Create a source column with same semantic type
        source_col = ColumnDefinition(
            column_id=99,  # Different ID
            x0=100,
            x1=200,
            semantic_type="date",
        )

        result = mapper.map_cell_to_reference(source_col)
        assert result is not None
        assert result.column_id == 0  # Should map to reference date column

    def test_map_unknown_type_returns_none(self, sample_columns):
        """Test that unknown type returns None."""
        mapper = SemanticColumnMapper(sample_columns)

        source_col = ColumnDefinition(
            column_id=99,
            x0=100,
            x1=200,
            semantic_type="unknown_type",
        )

        result = mapper.map_cell_to_reference(source_col)
        assert result is None


# =============================================================================
# PageBoundaryInfo Tests
# =============================================================================

class TestPageBoundaryInfo:
    """Tests for PageBoundaryInfo dataclass."""

    def test_initialization(self):
        """Test PageBoundaryInfo creation."""
        info = PageBoundaryInfo(
            page_number=1,
            page_height=792,
            last_row_y_bottom=750,
            first_row_y_top=100,
        )
        assert info.page_number == 1
        assert info.page_height == 792


# =============================================================================
# TableMerger Tests
# =============================================================================

class TestTableMerger:
    """Tests for TableMerger class."""

    def test_initialization(self):
        """Test default initialization."""
        merger = TableMerger()
        assert merger.cell_extractor is not None

    def test_single_page_extraction(self, single_page_table, sample_columns):
        """Test extracting from a single page."""
        pages_data = {
            0: [
                {"text": "15/01/2024", "x0": 60, "x1": 130, "top": 150, "bottom": 162},
                {"text": "Test", "x0": 160, "x1": 200, "top": 150, "bottom": 162},
                {"text": "1000", "x0": 360, "x1": 410, "top": 150, "bottom": 162},
            ]
        }
        mock_pdf = create_mock_pdf(pages_data)

        merger = TableMerger()
        rows = merger.extract_single_page_rows(mock_pdf, single_page_table, 0)

        assert len(rows) >= 1

    def test_multi_page_merge(self, multi_page_table):
        """Test merging rows from multiple pages."""
        pages_data = {
            0: [
                # Header row
                {"text": "Date", "x0": 60, "x1": 100, "top": 100, "bottom": 112},
                # Data row
                {"text": "15/01/2024", "x0": 60, "x1": 130, "top": 150, "bottom": 162},
                {"text": "Page1 Txn", "x0": 160, "x1": 250, "top": 150, "bottom": 162},
            ],
            1: [
                # Repeated header
                {"text": "Date", "x0": 60, "x1": 100, "top": 50, "bottom": 62},
                # Data row
                {"text": "16/01/2024", "x0": 60, "x1": 130, "top": 100, "bottom": 112},
                {"text": "Page2 Txn", "x0": 160, "x1": 250, "top": 100, "bottom": 112},
            ],
            2: [
                # Repeated header
                {"text": "Date", "x0": 60, "x1": 100, "top": 50, "bottom": 62},
                # Data row
                {"text": "17/01/2024", "x0": 60, "x1": 130, "top": 100, "bottom": 112},
                {"text": "Page3 Txn", "x0": 160, "x1": 250, "top": 100, "bottom": 112},
            ],
        }
        mock_pdf = create_mock_pdf(pages_data)

        merger = TableMerger()
        rows = merger.merge_multi_page_table(mock_pdf, multi_page_table)

        # Should have rows from all pages (headers skipped on pages 2 and 3)
        assert len(rows) >= 3


# =============================================================================
# TableRecognizer Tests
# =============================================================================

class TestTableRecognizer:
    """Tests for TableRecognizer class."""

    def test_initialization(self):
        """Test default initialization."""
        recognizer = TableRecognizer()
        assert recognizer.enable_row_merge is True
        assert recognizer.min_rows == 1

    def test_custom_initialization(self):
        """Test custom initialization."""
        recognizer = TableRecognizer(
            enable_row_merge=False,
            min_rows=5,
        )
        assert recognizer.enable_row_merge is False
        assert recognizer.min_rows == 5

    def test_recognize_single_table(self, single_page_table, sample_columns):
        """Test recognizing a single table."""
        pages_data = {
            0: [
                {"text": "15/01/2024", "x0": 60, "x1": 130, "top": 150, "bottom": 162},
                {"text": "Transaction", "x0": 160, "x1": 260, "top": 150, "bottom": 162},
                {"text": "1000", "x0": 360, "x1": 410, "top": 150, "bottom": 162},
                {"text": "49000", "x0": 460, "x1": 530, "top": 150, "bottom": 162},
            ]
        }
        mock_pdf = create_mock_pdf(pages_data)

        detection_result = TableDetectionResult(
            tables=[single_page_table],
            pages_processed=[0],
        )

        recognizer = TableRecognizer()
        result = recognizer.recognize(mock_pdf, detection_result)

        assert result.success is True
        assert result.table_count == 1
        assert result.total_transactions >= 1

    def test_recognize_filters_non_transaction_tables(self, sample_columns):
        """Test that non-transaction tables are filtered out."""
        summary_table = TableDefinition(
            table_id="summary_table",
            page_numbers=[0],
            columns=sample_columns,
            bounds_per_page={0: BoundingBox(x0=50, y0=100, x1=550, y1=200)},
            content_type=ContentType.SUMMARY,  # Not transaction
        )

        detection_result = TableDetectionResult(
            tables=[summary_table],
            pages_processed=[0],
        )

        mock_pdf = create_mock_pdf({})
        recognizer = TableRecognizer()
        result = recognizer.recognize(mock_pdf, detection_result)

        # Should not process summary tables
        assert result.table_count == 0

    def test_recognize_specific_table_ids(self, single_page_table, sample_columns):
        """Test recognizing only specific table IDs."""
        other_table = TableDefinition(
            table_id="other_table",
            page_numbers=[1],
            columns=sample_columns,
            bounds_per_page={1: BoundingBox(x0=50, y0=100, x1=550, y1=500)},
            content_type=ContentType.TRANSACTION,
        )

        pages_data = {
            0: [
                {"text": "15/01/2024", "x0": 60, "x1": 130, "top": 150, "bottom": 162},
                {"text": "Test", "x0": 160, "x1": 200, "top": 150, "bottom": 162},
            ]
        }
        mock_pdf = create_mock_pdf(pages_data)

        detection_result = TableDetectionResult(
            tables=[single_page_table, other_table],
            pages_processed=[0, 1],
        )

        recognizer = TableRecognizer()
        result = recognizer.recognize(
            mock_pdf,
            detection_result,
            table_ids=[single_page_table.table_id],
        )

        # Should only process specified table
        assert result.table_count == 1
        assert result.source_table_ids == [single_page_table.table_id]

    def test_recognize_table_method(self, single_page_table, sample_columns):
        """Test recognize_table method for single table."""
        pages_data = {
            0: [
                {"text": "15/01/2024", "x0": 60, "x1": 130, "top": 150, "bottom": 162},
                {"text": "Test", "x0": 160, "x1": 200, "top": 150, "bottom": 162},
                {"text": "1000", "x0": 360, "x1": 410, "top": 150, "bottom": 162},
            ]
        }
        mock_pdf = create_mock_pdf(pages_data)

        recognizer = TableRecognizer()
        table = recognizer.recognize_table(mock_pdf, single_page_table)

        assert table is not None
        assert table.table_id == single_page_table.table_id

    def test_empty_table_returns_none(self, single_page_table):
        """Test that empty table returns None."""
        mock_pdf = create_mock_pdf({0: []})

        recognizer = TableRecognizer()
        table = recognizer.recognize_table(mock_pdf, single_page_table)

        assert table is None


# =============================================================================
# Convenience Function Tests
# =============================================================================

class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_recognize_tables_function(self, single_page_table, sample_columns):
        """Test recognize_tables convenience function."""
        pages_data = {
            0: [
                {"text": "15/01/2024", "x0": 60, "x1": 130, "top": 150, "bottom": 162},
                {"text": "Test", "x0": 160, "x1": 200, "top": 150, "bottom": 162},
            ]
        }
        mock_pdf = create_mock_pdf(pages_data)

        detection_result = TableDetectionResult(
            tables=[single_page_table],
            pages_processed=[0],
        )

        result = recognize_tables(mock_pdf, detection_result)

        assert result is not None
        assert isinstance(result.tables, list)

    def test_extract_transactions_function(self, single_page_table, sample_columns):
        """Test extract_transactions convenience function."""
        pages_data = {
            0: [
                {"text": "15/01/2024", "x0": 60, "x1": 130, "top": 150, "bottom": 162},
                {"text": "Test", "x0": 160, "x1": 200, "top": 150, "bottom": 162},
            ]
        }
        mock_pdf = create_mock_pdf(pages_data)

        table = extract_transactions(mock_pdf, single_page_table)

        # May return None if no valid transactions
        # but should not raise an error
        assert table is None or hasattr(table, 'rows')

    def test_merge_table_pages_function(self, multi_page_table):
        """Test merge_table_pages convenience function."""
        pages_data = {
            0: [{"text": "Test", "x0": 160, "x1": 200, "top": 150, "bottom": 162}],
            1: [{"text": "Test2", "x0": 160, "x1": 200, "top": 100, "bottom": 112}],
            2: [{"text": "Test3", "x0": 160, "x1": 200, "top": 100, "bottom": 112}],
        }
        mock_pdf = create_mock_pdf(pages_data)

        rows = merge_table_pages(mock_pdf, multi_page_table)

        assert isinstance(rows, list)


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests for the recognition pipeline."""

    def test_full_pipeline_single_page(self, sample_columns):
        """Test full pipeline for single page table."""
        table_def = TableDefinition(
            table_id="integration_test",
            page_numbers=[0],
            columns=sample_columns,
            bounds_per_page={0: BoundingBox(x0=50, y0=100, x1=550, y1=500)},
            content_type=ContentType.TRANSACTION,
            detection_confidence=0.95,
        )

        pages_data = {
            0: [
                # Row 1
                {"text": "15/01/2024", "x0": 60, "x1": 130, "top": 150, "bottom": 162},
                {"text": "ATM", "x0": 160, "x1": 190, "top": 150, "bottom": 162},
                {"text": "Withdrawal", "x0": 195, "x1": 270, "top": 150, "bottom": 162},
                {"text": "5000.00", "x0": 360, "x1": 420, "top": 150, "bottom": 162},
                {"text": "45000.00", "x0": 460, "x1": 530, "top": 150, "bottom": 162},
                # Row 2
                {"text": "16/01/2024", "x0": 60, "x1": 130, "top": 180, "bottom": 192},
                {"text": "Salary", "x0": 160, "x1": 200, "top": 180, "bottom": 192},
                {"text": "Credit", "x0": 205, "x1": 250, "top": 180, "bottom": 192},
                {"text": "50000.00", "x0": 360, "x1": 420, "top": 180, "bottom": 192},
                {"text": "95000.00", "x0": 460, "x1": 530, "top": 180, "bottom": 192},
            ]
        }
        mock_pdf = create_mock_pdf(pages_data)

        detection_result = TableDetectionResult(
            tables=[table_def],
            pages_processed=[0],
        )

        result = recognize_tables(mock_pdf, detection_result)

        assert result.success is True
        assert result.table_count == 1
        assert result.total_transactions == 2

        table = result.tables[0]
        assert table.row_count == 2

        # Check first transaction
        row1 = table.rows[0]
        assert row1.transaction_date == date(2024, 1, 15)
        assert "ATM" in row1.description
        assert row1.debit_amount == Decimal("5000.00")

        # Check second transaction
        row2 = table.rows[1]
        assert row2.transaction_date == date(2024, 1, 16)


# =============================================================================
# Error Handling Tests
# =============================================================================

class TestErrorHandling:
    """Tests for error handling."""

    def test_handles_extraction_error_gracefully(self, single_page_table):
        """Test that extraction errors are handled gracefully."""
        mock_pdf = Mock()
        mock_pdf.get_page_words.side_effect = Exception("PDF read error")

        detection_result = TableDetectionResult(
            tables=[single_page_table],
            pages_processed=[0],
        )

        recognizer = TableRecognizer()
        result = recognizer.recognize(mock_pdf, detection_result)

        # Should handle error gracefully
        assert result.has_errors is True
        assert len(result.errors) > 0

    def test_no_tables_returns_empty_result(self):
        """Test that no tables returns empty result."""
        mock_pdf = Mock()

        detection_result = TableDetectionResult(
            tables=[],
            pages_processed=[0],
        )

        recognizer = TableRecognizer()
        result = recognizer.recognize(mock_pdf, detection_result)

        assert result.success is True
        assert result.table_count == 0
