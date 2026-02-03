"""
Unit tests for cell extraction.

Tests the CellExtractor class and related functionality.
"""

import pytest
from unittest.mock import Mock, MagicMock

from tables.recognizer.cell_extractor import (
    CellExtractor,
    Cell,
    ExtractedRow,
    extract_table_rows,
    get_cell_text_by_column,
)
from tables.models.table import TableDefinition, ColumnDefinition, DataType
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
            header_text="Credit",
            semantic_type="credit",
            data_type=DataType.NUMERIC,
        ),
    ]


@pytest.fixture
def sample_table_definition(sample_columns):
    """Create a sample table definition."""
    return TableDefinition(
        table_id="test_table",
        page_numbers=[0],
        columns=sample_columns,
        bounds_per_page={
            0: BoundingBox(x0=50, y0=100, x1=550, y1=500)
        },
        header_row_indices=[0],
    )


@pytest.fixture
def sample_words():
    """Create sample word dictionaries."""
    return [
        # Row 1 (header)
        {"text": "Date", "x0": 60, "x1": 90, "top": 100, "bottom": 112},
        {"text": "Description", "x0": 160, "x1": 240, "top": 100, "bottom": 112},
        {"text": "Debit", "x0": 360, "x1": 400, "top": 100, "bottom": 112},
        {"text": "Credit", "x0": 460, "x1": 500, "top": 100, "bottom": 112},
        # Row 2 (data)
        {"text": "15/01/2024", "x0": 55, "x1": 120, "top": 130, "bottom": 142},
        {"text": "ATM", "x0": 155, "x1": 180, "top": 130, "bottom": 142},
        {"text": "Withdrawal", "x0": 185, "x1": 260, "top": 130, "bottom": 142},
        {"text": "5000.00", "x0": 355, "x1": 420, "top": 130, "bottom": 142},
        # Row 3 (data)
        {"text": "16/01/2024", "x0": 55, "x1": 120, "top": 160, "bottom": 172},
        {"text": "Salary", "x0": 155, "x1": 200, "top": 160, "bottom": 172},
        {"text": "Credit", "x0": 205, "x1": 250, "top": 160, "bottom": 172},
        {"text": "50000.00", "x0": 455, "x1": 530, "top": 160, "bottom": 172},
    ]


@pytest.fixture
def mock_pdf_document(sample_words):
    """Create a mock PDF document."""
    mock = Mock()
    mock.get_page_words.return_value = sample_words
    return mock


# =============================================================================
# Cell Tests
# =============================================================================

class TestCell:
    """Tests for Cell dataclass."""

    def test_empty_cell(self):
        """Test default empty cell."""
        col = ColumnDefinition(column_id=0, x0=0, x1=100, header_text="Test")
        cell = Cell(column_index=0, column=col)
        assert cell.is_empty is True
        assert cell.text == ""
        assert cell.words == []
        assert cell.bbox is None

    def test_add_word(self):
        """Test adding a word to a cell."""
        col = ColumnDefinition(column_id=0, x0=0, x1=100, header_text="Test")
        cell = Cell(column_index=0, column=col)

        word = {"text": "Hello", "x0": 10, "x1": 50, "top": 100, "bottom": 112}
        cell.add_word(word)

        assert cell.is_empty is False
        assert cell.text == "Hello"
        assert len(cell.words) == 1
        assert cell.bbox is not None
        assert cell.bbox.x0 == 10

    def test_add_multiple_words(self):
        """Test adding multiple words to a cell."""
        col = ColumnDefinition(column_id=0, x0=0, x1=100, header_text="Test")
        cell = Cell(column_index=0, column=col)

        cell.add_word({"text": "Hello", "x0": 10, "x1": 50, "top": 100, "bottom": 112})
        cell.add_word({"text": "World", "x0": 55, "x1": 90, "top": 100, "bottom": 112})

        assert cell.text == "Hello World"
        assert len(cell.words) == 2
        # Bounding box should span both words
        assert cell.bbox.x0 == 10
        assert cell.bbox.x1 == 90


# =============================================================================
# ExtractedRow Tests
# =============================================================================

class TestExtractedRow:
    """Tests for ExtractedRow dataclass."""

    @pytest.fixture
    def sample_row(self, sample_columns):
        """Create a sample extracted row."""
        cells = {}
        for i, col in enumerate(sample_columns):
            cell = Cell(column_index=i, column=col)
            if i == 0:
                cell.add_word({"text": "15/01/2024", "x0": 55, "x1": 120, "top": 100, "bottom": 112})
            elif i == 1:
                cell.add_word({"text": "ATM Withdrawal", "x0": 155, "x1": 260, "top": 100, "bottom": 112})
            elif i == 2:
                cell.add_word({"text": "5000.00", "x0": 355, "x1": 420, "top": 100, "bottom": 112})
            cells[col.column_id] = cell

        return ExtractedRow(
            row_index=1,
            page_number=0,
            cells=cells,
            y_position=100,
            height=12,
        )

    def test_get_cell(self, sample_row):
        """Test get_cell method."""
        cell = sample_row.get_cell(0)
        assert cell is not None
        assert cell.text == "15/01/2024"

        cell = sample_row.get_cell(99)
        assert cell is None

    def test_get_cell_text(self, sample_row):
        """Test get_cell_text method."""
        assert sample_row.get_cell_text(0) == "15/01/2024"
        assert sample_row.get_cell_text(1) == "ATM Withdrawal"
        assert sample_row.get_cell_text(99) == ""

    def test_get_cell_by_semantic_type(self, sample_row, sample_columns):
        """Test get_cell_by_semantic_type method."""
        cell = sample_row.get_cell_by_semantic_type("date", sample_columns)
        assert cell is not None
        assert cell.text == "15/01/2024"

        cell = sample_row.get_cell_by_semantic_type("debit", sample_columns)
        assert cell is not None
        assert cell.text == "5000.00"

    def test_all_cells_empty(self, sample_columns):
        """Test all_cells_empty property."""
        cells = {col.column_id: Cell(column_index=col.column_id, column=col)
                 for col in sample_columns}
        row = ExtractedRow(row_index=0, page_number=0, cells=cells)
        assert row.all_cells_empty is True

    def test_word_count(self, sample_row):
        """Test word_count property."""
        assert sample_row.word_count == 3  # 3 cells with words


# =============================================================================
# CellExtractor Tests
# =============================================================================

class TestCellExtractor:
    """Tests for CellExtractor class."""

    def test_initialization(self):
        """Test default initialization."""
        extractor = CellExtractor()
        assert extractor.column_tolerance == 3.0
        assert extractor.row_tolerance == 3.0

    def test_custom_initialization(self):
        """Test custom initialization."""
        extractor = CellExtractor(
            column_tolerance=5.0,
            row_tolerance=4.0,
            min_overlap=0.5,
        )
        assert extractor.column_tolerance == 5.0
        assert extractor.row_tolerance == 4.0

    def test_extract_cells_from_row(self, sample_columns):
        """Test extracting cells from a row of words."""
        extractor = CellExtractor()
        words = [
            {"text": "15/01/2024", "x0": 55, "x1": 120, "top": 100, "bottom": 112},
            {"text": "ATM", "x0": 155, "x1": 180, "top": 100, "bottom": 112},
            {"text": "5000.00", "x0": 355, "x1": 420, "top": 100, "bottom": 112},
        ]

        cells = extractor.extract_cells_from_row(words, sample_columns)

        assert len(cells) == 4
        assert cells[0].text == "15/01/2024"
        assert cells[1].text == "ATM"
        assert cells[2].text == "5000.00"
        assert cells[3].is_empty is True  # Credit column

    def test_word_at_column_boundary(self, sample_columns):
        """Test word that touches column boundary is correctly assigned."""
        extractor = CellExtractor()
        # Word at exact boundary between columns
        words = [
            {"text": "Boundary", "x0": 148, "x1": 152, "top": 100, "bottom": 112},
        ]

        cells = extractor.extract_cells_from_row(words, sample_columns)
        # Should be assigned to one of the adjacent columns
        non_empty_cells = [c for c in cells.values() if not c.is_empty]
        assert len(non_empty_cells) == 1

    def test_word_spanning_columns(self, sample_columns):
        """Test word that spans across columns."""
        extractor = CellExtractor()
        # Word spans from description into debit column
        words = [
            {"text": "Long description text", "x0": 200, "x1": 380, "top": 100, "bottom": 112},
        ]

        cells = extractor.extract_cells_from_row(words, sample_columns)
        # Should be assigned to column with majority overlap
        non_empty_cells = [c for c in cells.values() if not c.is_empty]
        assert len(non_empty_cells) == 1

    def test_extract_rows_skips_headers(
        self,
        mock_pdf_document,
        sample_table_definition
    ):
        """Test that header rows are skipped by default."""
        extractor = CellExtractor()
        rows = extractor.extract_rows(
            mock_pdf_document,
            0,
            sample_table_definition,
            skip_header_rows=True,
        )
        # Should have 2 data rows (header skipped)
        assert len(rows) == 2

    def test_extract_rows_includes_headers(
        self,
        mock_pdf_document,
        sample_table_definition
    ):
        """Test that headers can be included."""
        extractor = CellExtractor()
        rows = extractor.extract_rows(
            mock_pdf_document,
            0,
            sample_table_definition,
            skip_header_rows=False,
        )
        # Should have 3 rows (including header)
        assert len(rows) == 3
        assert rows[0].is_header is True

    def test_empty_page(self, sample_table_definition):
        """Test handling of empty page."""
        mock = Mock()
        mock.get_page_words.return_value = []

        extractor = CellExtractor()
        rows = extractor.extract_rows(mock, 0, sample_table_definition)
        assert rows == []


# =============================================================================
# Convenience Function Tests
# =============================================================================

class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_extract_table_rows(
        self,
        mock_pdf_document,
        sample_table_definition
    ):
        """Test extract_table_rows convenience function."""
        rows = extract_table_rows(
            mock_pdf_document,
            0,
            sample_table_definition,
        )
        assert len(rows) == 2

    def test_get_cell_text_by_column(self, sample_columns):
        """Test get_cell_text_by_column convenience function."""
        cells = {
            0: Cell(column_index=0, column=sample_columns[0]),
            1: Cell(column_index=1, column=sample_columns[1]),
        }
        cells[0].add_word({"text": "15/01/2024", "x0": 55, "x1": 120, "top": 100, "bottom": 112})

        row = ExtractedRow(row_index=0, page_number=0, cells=cells)

        result = get_cell_text_by_column(row, sample_columns, "date")
        assert result == "15/01/2024"

        result = get_cell_text_by_column(row, sample_columns, "balance")
        assert result == ""


# =============================================================================
# Edge Cases
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases."""

    def test_no_bounds_for_page(self, mock_pdf_document, sample_columns):
        """Test handling when page has no bounds defined."""
        table_def = TableDefinition(
            table_id="test",
            page_numbers=[0],
            columns=sample_columns,
            bounds_per_page={},  # No bounds
        )

        extractor = CellExtractor()
        rows = extractor.extract_rows(mock_pdf_document, 0, table_def)
        assert rows == []

    def test_words_outside_bounds(self, sample_columns):
        """Test that words outside table bounds are excluded."""
        mock = Mock()
        mock.get_page_words.return_value = [
            # Word outside bounds
            {"text": "Outside", "x0": 10, "x1": 40, "top": 50, "bottom": 62},
            # Word inside bounds
            {"text": "Inside", "x0": 100, "x1": 140, "top": 150, "bottom": 162},
        ]

        table_def = TableDefinition(
            table_id="test",
            page_numbers=[0],
            columns=sample_columns,
            bounds_per_page={
                0: BoundingBox(x0=50, y0=100, x1=550, y1=500)
            },
        )

        extractor = CellExtractor()
        rows = extractor.extract_rows(mock, 0, table_def)
        # Only "Inside" word should be captured
        assert len(rows) == 1
        all_text = " ".join(
            cell.text for cell in rows[0].cells.values() if not cell.is_empty
        )
        assert "Outside" not in all_text
        assert "Inside" in all_text
