"""
Unit tests for cross-page table merging.

Tests the TableMerger class, mid-row page break detection,
and semantic column mapping across pages.
"""

import pytest
from unittest.mock import Mock, MagicMock
from dataclasses import dataclass

from tables.recognizer.table_merger import (
    TableMerger,
    SemanticColumnMapper,
    PageBoundaryInfo,
    merge_table_pages,
)
from tables.recognizer.cell_extractor import Cell, ExtractedRow
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
    ]


@pytest.fixture
def table_definition(sample_columns):
    """Create a multi-page table definition."""
    return TableDefinition(
        table_id="test_table",
        page_numbers=[0, 1],  # Spans two pages
        columns=sample_columns,
        bounds_per_page={
            0: BoundingBox(x0=50, y0=100, x1=450, y1=700),
            1: BoundingBox(x0=50, y0=50, x1=450, y1=700),
        },
        header_row_indices=[0],
        header_repeats_on_pages=True,
    )


def create_cell(column: ColumnDefinition, text: str) -> Cell:
    """Helper to create a cell with text."""
    cell = Cell(column_index=column.column_id, column=column)
    if text:
        cell.add_word({
            "text": text,
            "x0": column.x0,
            "x1": column.x1,
            "top": 0,
            "bottom": 12,
        })
    return cell


def create_row(
    row_index: int,
    page_number: int,
    columns: list[ColumnDefinition],
    cell_texts: list[str],
    y_position: float = 100.0,
) -> ExtractedRow:
    """Helper to create a row with specified cell texts."""
    cells = {}
    for col, text in zip(columns, cell_texts):
        cells[col.column_id] = create_cell(col, text)

    return ExtractedRow(
        row_index=row_index,
        page_number=page_number,
        cells=cells,
        y_position=y_position,
        height=12.0,
    )


# =============================================================================
# SemanticColumnMapper Tests
# =============================================================================

class TestSemanticColumnMapper:
    """Tests for SemanticColumnMapper class."""

    def test_map_by_semantic_type(self, sample_columns):
        """Test mapping columns by semantic type."""
        mapper = SemanticColumnMapper(sample_columns)

        # Create a source column with same semantic type but different ID
        source_col = ColumnDefinition(
            column_id=99,  # Different ID
            x0=60.0,
            x1=140.0,
            semantic_type="date",
        )

        result = mapper.map_cell_to_reference(source_col)

        assert result is not None
        assert result.column_id == 0
        assert result.semantic_type == "date"

    def test_fallback_to_position_when_no_semantic_type(self, sample_columns):
        """Test fallback to position-based matching."""
        mapper = SemanticColumnMapper(sample_columns)

        # Column with no semantic type but same ID
        source_col = ColumnDefinition(
            column_id=1,
            x0=160.0,
            x1=340.0,
            semantic_type=None,
        )

        result = mapper.map_cell_to_reference(source_col)

        assert result is not None
        assert result.column_id == 1

    def test_remap_row(self, sample_columns):
        """Test remapping a row's cells."""
        mapper = SemanticColumnMapper(sample_columns)

        # Create a row with cells mapped to different column IDs
        other_columns = [
            ColumnDefinition(column_id=10, x0=0, x1=100, semantic_type="date"),
            ColumnDefinition(column_id=11, x0=100, x1=300, semantic_type="description"),
        ]

        cells = {
            10: create_cell(other_columns[0], "01/01/24"),
            11: create_cell(other_columns[1], "Test"),
        }
        row = ExtractedRow(row_index=0, page_number=1, cells=cells)

        remapped = mapper.remap_row(row, other_columns)

        # Should have cells keyed by reference column IDs
        assert 0 in remapped.cells  # date column
        assert 1 in remapped.cells  # description column
        assert remapped.cells[0].text == "01/01/24"


# =============================================================================
# PageBoundaryInfo Tests
# =============================================================================

class TestPageBoundaryInfo:
    """Tests for PageBoundaryInfo dataclass."""

    def test_basic_initialization(self):
        """Test basic initialization."""
        info = PageBoundaryInfo(
            page_number=0,
            page_height=800.0,
            first_row_y_top=100.0,
            last_row_y_bottom=750.0,
        )

        assert info.page_number == 0
        assert info.page_height == 800.0


# =============================================================================
# Mid-Row Page Break Detection Tests
# =============================================================================

class TestMidRowPageBreak:
    """Tests for mid-row page break detection."""

    @pytest.fixture
    def merger(self):
        """Create a TableMerger instance."""
        return TableMerger()

    def test_detect_mid_row_break_position_signals(self, merger, sample_columns):
        """Test detection based on position signals."""
        prev_info = PageBoundaryInfo(
            page_number=0,
            page_height=800.0,
            last_row_y_bottom=780.0,  # Near bottom
        )
        current_info = PageBoundaryInfo(
            page_number=1,
            page_height=800.0,
            first_row_y_top=30.0,  # Near top
        )

        # Create a row with no date (continuation signal)
        row = create_row(0, 1, sample_columns, ["", "Continuation text", ""])

        result = merger._detect_mid_row_break(prev_info, current_info, [row])

        assert result is True

    def test_no_break_when_row_has_date(self, merger, sample_columns):
        """Test no break detected when row has date."""
        prev_info = PageBoundaryInfo(
            page_number=0,
            page_height=800.0,
            last_row_y_bottom=780.0,
        )
        current_info = PageBoundaryInfo(
            page_number=1,
            page_height=800.0,
            first_row_y_top=30.0,
        )

        # Row has a date - not a continuation
        row = create_row(0, 1, sample_columns, ["15/01/24", "New transaction", ""])

        result = merger._detect_mid_row_break(prev_info, current_info, [row])

        assert result is False

    def test_no_break_when_not_near_boundaries(self, merger, sample_columns):
        """Test no break when rows are not near page boundaries."""
        prev_info = PageBoundaryInfo(
            page_number=0,
            page_height=800.0,
            last_row_y_bottom=400.0,  # Middle of page
        )
        current_info = PageBoundaryInfo(
            page_number=1,
            page_height=800.0,
            first_row_y_top=200.0,  # Not at top
        )

        row = create_row(0, 1, sample_columns, ["", "Text", ""])

        result = merger._detect_mid_row_break(prev_info, current_info, [row])

        assert result is False


# =============================================================================
# Row Merging Tests
# =============================================================================

class TestRowMerging:
    """Tests for merging split rows."""

    @pytest.fixture
    def merger(self):
        """Create a TableMerger instance."""
        return TableMerger()

    def test_merge_split_row_concatenates_text(self, merger, sample_columns):
        """Test that split row text is concatenated."""
        prev_row = create_row(0, 0, sample_columns, ["15/01/24", "Start of", "1000"])
        cont_row = create_row(1, 1, sample_columns, ["", "description", ""])

        merged = merger._merge_split_row(prev_row, cont_row)

        assert merged.cells[0].text == "15/01/24"  # Date unchanged
        assert "Start of" in merged.cells[1].text
        assert "description" in merged.cells[1].text
        assert merged.cells[2].text == "1000"  # Debit unchanged

    def test_merge_preserves_original_row_info(self, merger, sample_columns):
        """Test that merged row preserves original row info."""
        prev_row = create_row(5, 0, sample_columns, ["15/01/24", "Test", ""])
        prev_row.y_position = 500.0
        prev_row.height = 12.0

        cont_row = create_row(0, 1, sample_columns, ["", "More", ""])
        cont_row.height = 15.0

        merged = merger._merge_split_row(prev_row, cont_row)

        assert merged.row_index == 5  # Original index
        assert merged.page_number == 0  # Original page
        assert merged.height == 27.0  # Combined height


# =============================================================================
# Multi-Page Table Merging Tests
# =============================================================================

class TestMultiPageMerging:
    """Tests for full multi-page table merging."""

    def test_merge_two_page_table(self, table_definition, sample_columns):
        """Test merging a two-page table."""
        # Mock PDF document
        mock_pdf = Mock()
        mock_pdf.get_page_dimensions.return_value = (612, 792)

        # Mock rows for each page
        page0_rows = [
            create_row(0, 0, sample_columns, ["15/01/24", "Transaction 1", "1000"], y_position=100),
            create_row(1, 0, sample_columns, ["16/01/24", "Transaction 2", "500"], y_position=120),
        ]
        page1_rows = [
            create_row(0, 1, sample_columns, ["17/01/24", "Transaction 3", "200"], y_position=50),
        ]

        # Create merger and mock cell_extractor
        merger = TableMerger()
        merger.cell_extractor = Mock()
        merger.cell_extractor.extract_rows.side_effect = [page0_rows, page1_rows]

        result = merger.merge_multi_page_table(mock_pdf, table_definition)

        assert len(result) == 3
        assert result[0].cells[1].text == "Transaction 1"
        assert result[2].cells[1].text == "Transaction 3"

        # Check row indices are renumbered
        assert result[0].row_index == 0
        assert result[1].row_index == 1
        assert result[2].row_index == 2

    def test_merge_with_repeated_headers(self, table_definition, sample_columns):
        """Test that repeated headers are skipped."""
        mock_pdf = Mock()
        mock_pdf.get_page_dimensions.return_value = (612, 792)

        page0_rows = [
            create_row(0, 0, sample_columns, ["15/01/24", "Transaction 1", "1000"]),
        ]
        page1_rows = [
            create_row(0, 1, sample_columns, ["16/01/24", "Transaction 2", "500"]),
        ]

        merger = TableMerger()
        merger.cell_extractor = Mock()

        # Simulate that second page call skips headers
        merger.cell_extractor.extract_rows.side_effect = [page0_rows, page1_rows]

        table_definition.header_repeats_on_pages = True
        result = merger.merge_multi_page_table(mock_pdf, table_definition)

        # Both transactions should be included
        assert len(result) == 2


# =============================================================================
# Convenience Function Tests
# =============================================================================

class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_merge_table_pages(self, table_definition):
        """Test merge_table_pages convenience function."""
        mock_pdf = Mock()
        mock_pdf.get_page_dimensions.return_value = (612, 792)
        mock_pdf.get_page_words.return_value = []

        # Should not raise and return empty list when no words
        result = merge_table_pages(mock_pdf, table_definition)

        assert isinstance(result, list)


# =============================================================================
# Edge Cases
# =============================================================================

class TestEdgeCases:
    """Edge case tests."""

    def test_empty_page_numbers(self, sample_columns):
        """Test with empty page numbers."""
        table_def = TableDefinition(
            table_id="test",
            page_numbers=[],
            columns=sample_columns,
        )

        merger = TableMerger()
        result = merger.merge_multi_page_table(Mock(), table_def)

        assert result == []

    def test_single_page_table(self, sample_columns):
        """Test single page table (no actual merging needed)."""
        table_def = TableDefinition(
            table_id="test",
            page_numbers=[0],
            columns=sample_columns,
            bounds_per_page={0: BoundingBox(x0=50, y0=100, x1=450, y1=700)},
        )

        mock_pdf = Mock()
        mock_pdf.get_page_dimensions.return_value = (612, 792)

        rows = [
            create_row(0, 0, sample_columns, ["15/01/24", "Test", "1000"]),
        ]

        merger = TableMerger()
        merger.cell_extractor = Mock()
        merger.cell_extractor.extract_rows.return_value = rows

        result = merger.merge_multi_page_table(mock_pdf, table_def)

        assert len(result) == 1

    def test_column_mapping_with_missing_semantic_types(self, sample_columns):
        """Test column mapping when some columns lack semantic types."""
        # Create columns without semantic types
        minimal_columns = [
            ColumnDefinition(column_id=0, x0=50, x1=150, semantic_type=None),
            ColumnDefinition(column_id=1, x0=150, x1=350, semantic_type="description"),
        ]

        mapper = SemanticColumnMapper(minimal_columns)

        # Should still map by position for column without semantic type
        source = ColumnDefinition(column_id=0, x0=55, x1=145, semantic_type=None)
        result = mapper.map_cell_to_reference(source)

        assert result is not None
        assert result.column_id == 0
