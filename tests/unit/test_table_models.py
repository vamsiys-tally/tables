"""
Unit tests for table models.

Tests cover:
- ColumnDefinition creation and operations
- TableDefinition creation and operations
- TableDetectionResult container
- Serialization (to_dict) and deserialization (from_dict)
"""

import pytest

from tables.models.table import (
    ColumnDefinition,
    TableDefinition,
    TableDetectionResult,
    StructureType,
    ContentType,
    DataType,
)
from tables.utils.geometry import BoundingBox


class TestColumnDefinition:
    """Tests for the ColumnDefinition class."""

    def test_basic_creation(self):
        """Test basic column creation."""
        col = ColumnDefinition(
            column_id=0,
            x0=50.0,
            x1=150.0,
            header_text="Date",
            semantic_type="date",
            data_type=DataType.DATE,
        )

        assert col.column_id == 0
        assert col.x0 == 50.0
        assert col.x1 == 150.0
        assert col.header_text == "Date"
        assert col.semantic_type == "date"
        assert col.data_type == DataType.DATE

    def test_width(self):
        """Test width calculation."""
        col = ColumnDefinition(column_id=0, x0=50.0, x1=150.0)
        assert col.width == 100.0

    def test_center_x(self):
        """Test horizontal center calculation."""
        col = ColumnDefinition(column_id=0, x0=50.0, x1=150.0)
        assert col.center_x == 100.0

    def test_contains_x(self):
        """Test x-coordinate containment."""
        col = ColumnDefinition(column_id=0, x0=50.0, x1=150.0)

        assert col.contains_x(100.0) is True
        assert col.contains_x(50.0) is True  # Edge
        assert col.contains_x(150.0) is True  # Edge
        assert col.contains_x(200.0) is False  # Outside

    def test_contains_x_with_tolerance(self):
        """Test x-coordinate containment with tolerance."""
        col = ColumnDefinition(column_id=0, x0=50.0, x1=150.0)

        # Just outside but within tolerance
        assert col.contains_x(48.0, tolerance=5.0) is True
        assert col.contains_x(152.0, tolerance=5.0) is True

    def test_overlaps_x_range(self):
        """Test x-range overlap detection."""
        col = ColumnDefinition(column_id=0, x0=50.0, x1=150.0)

        assert col.overlaps_x_range(100.0, 200.0) is True  # Partial overlap
        assert col.overlaps_x_range(0.0, 100.0) is True  # Partial overlap
        assert col.overlaps_x_range(200.0, 300.0) is False  # No overlap

    def test_to_dict(self):
        """Test serialization to dictionary."""
        col = ColumnDefinition(
            column_id=0,
            x0=50.0,
            x1=150.0,
            header_text="Debit",
            semantic_type="debit",
            data_type=DataType.NUMERIC,
            confidence=0.95,
        )

        d = col.to_dict()
        assert d["column_id"] == 0
        assert d["x0"] == 50.0
        assert d["x1"] == 150.0
        assert d["header_text"] == "Debit"
        assert d["semantic_type"] == "debit"
        assert d["data_type"] == "numeric"
        assert d["confidence"] == 0.95


class TestTableDefinition:
    """Tests for the TableDefinition class."""

    @pytest.fixture
    def sample_columns(self):
        """Create sample columns for testing."""
        return [
            ColumnDefinition(column_id=0, x0=0, x1=50, header_text="Date", semantic_type="date"),
            ColumnDefinition(column_id=1, x0=50, x1=200, header_text="Description", semantic_type="description"),
            ColumnDefinition(column_id=2, x0=200, x1=280, header_text="Debit", semantic_type="debit"),
            ColumnDefinition(column_id=3, x0=280, x1=360, header_text="Credit", semantic_type="credit"),
            ColumnDefinition(column_id=4, x0=360, x1=450, header_text="Balance", semantic_type="balance"),
        ]

    def test_basic_creation(self, sample_columns):
        """Test basic table creation."""
        table = TableDefinition(
            table_id="test_table",
            page_numbers=[0],
            columns=sample_columns,
            structure_type=StructureType.BORDERED,
            content_type=ContentType.TRANSACTION,
        )

        assert table.table_id == "test_table"
        assert table.page_numbers == [0]
        assert table.column_count == 5
        assert table.structure_type == StructureType.BORDERED
        assert table.content_type == ContentType.TRANSACTION

    def test_auto_generated_table_id(self):
        """Test auto-generation of table ID."""
        table = TableDefinition()
        assert table.table_id.startswith("table_")
        assert len(table.table_id) == 14  # "table_" + 8 chars

    def test_page_properties(self, sample_columns):
        """Test page-related properties."""
        table = TableDefinition(
            page_numbers=[1, 2, 3],
            columns=sample_columns,
        )

        assert table.first_page == 1
        assert table.last_page == 3
        assert table.page_count == 3

    def test_get_column_by_semantic_type(self, sample_columns):
        """Test finding column by semantic type."""
        table = TableDefinition(columns=sample_columns)

        date_col = table.get_column_by_semantic_type("date")
        assert date_col is not None
        assert date_col.header_text == "Date"

        unknown_col = table.get_column_by_semantic_type("unknown")
        assert unknown_col is None

    def test_get_columns_by_semantic_type(self, sample_columns):
        """Test finding all columns with a semantic type."""
        # Add another date column
        columns = sample_columns + [
            ColumnDefinition(column_id=5, x0=450, x1=500, header_text="Value Date", semantic_type="date")
        ]
        table = TableDefinition(columns=columns)

        date_cols = table.get_columns_by_semantic_type("date")
        assert len(date_cols) == 2

    def test_find_column_for_x(self, sample_columns):
        """Test finding column for x-coordinate."""
        table = TableDefinition(columns=sample_columns)

        col = table.find_column_for_x(25.0)
        assert col is not None
        assert col.header_text == "Date"

        col = table.find_column_for_x(320.0)
        assert col is not None
        assert col.header_text == "Credit"

    def test_has_required_transaction_columns(self, sample_columns):
        """Test checking for required transaction columns."""
        table = TableDefinition(columns=sample_columns)
        assert table.has_required_transaction_columns() is True

        # Missing date
        incomplete_cols = [c for c in sample_columns if c.semantic_type != "date"]
        table2 = TableDefinition(columns=incomplete_cols)
        assert table2.has_required_transaction_columns() is False

    def test_bounds_per_page(self, sample_columns):
        """Test bounds per page storage and retrieval."""
        bounds = {
            0: BoundingBox(x0=50, y0=100, x1=500, y1=600),
            1: BoundingBox(x0=50, y0=50, x1=500, y1=650),
        }
        table = TableDefinition(
            page_numbers=[0, 1],
            columns=sample_columns,
            bounds_per_page=bounds,
        )

        assert table.get_bounds(0) == bounds[0]
        assert table.get_bounds(1) == bounds[1]
        assert table.get_bounds(2) is None

    def test_add_warning(self, sample_columns):
        """Test adding warnings."""
        table = TableDefinition(columns=sample_columns)

        table.add_warning("First warning")
        table.add_warning("Second warning")
        table.add_warning("First warning")  # Duplicate

        assert len(table.warnings) == 2

    def test_to_dict(self, sample_columns):
        """Test serialization to dictionary."""
        table = TableDefinition(
            table_id="test_table",
            page_numbers=[0, 1],
            columns=sample_columns,
            structure_type=StructureType.BORDERED,
            content_type=ContentType.TRANSACTION,
            is_multi_page=True,
            detection_confidence=0.9,
        )

        d = table.to_dict()
        assert d["table_id"] == "test_table"
        assert d["page_numbers"] == [0, 1]
        assert len(d["columns"]) == 5
        assert d["structure_type"] == "bordered"
        assert d["content_type"] == "transaction"
        assert d["is_multi_page"] is True
        assert d["detection_confidence"] == 0.9

    def test_from_dict(self, sample_columns):
        """Test deserialization from dictionary."""
        original = TableDefinition(
            table_id="test_table",
            page_numbers=[0, 1],
            columns=sample_columns,
            structure_type=StructureType.BORDERED,
            content_type=ContentType.TRANSACTION,
            bounds_per_page={0: BoundingBox(0, 0, 100, 100)},
        )

        d = original.to_dict()
        restored = TableDefinition.from_dict(d)

        assert restored.table_id == original.table_id
        assert restored.page_numbers == original.page_numbers
        assert restored.column_count == original.column_count
        assert restored.structure_type == original.structure_type
        assert restored.content_type == original.content_type


class TestTableDetectionResult:
    """Tests for the TableDetectionResult container."""

    @pytest.fixture
    def sample_tables(self):
        """Create sample tables for testing."""
        return [
            TableDefinition(
                table_id="table_1",
                page_numbers=[0],
                content_type=ContentType.TRANSACTION,
            ),
            TableDefinition(
                table_id="table_2",
                page_numbers=[1, 2],
                content_type=ContentType.TRANSACTION,
            ),
            TableDefinition(
                table_id="table_3",
                page_numbers=[0],
                content_type=ContentType.SUMMARY,
            ),
        ]

    def test_basic_creation(self, sample_tables):
        """Test basic result creation."""
        result = TableDetectionResult(
            tables=sample_tables,
            pages_processed=[0, 1, 2],
            pages_with_tables=[0, 1, 2],
        )

        assert result.table_count == 3
        assert result.has_tables is True

    def test_empty_result(self):
        """Test empty result."""
        result = TableDetectionResult()

        assert result.table_count == 0
        assert result.has_tables is False

    def test_transaction_tables_filter(self, sample_tables):
        """Test filtering to transaction tables only."""
        result = TableDetectionResult(tables=sample_tables)

        transaction_tables = result.transaction_tables
        assert len(transaction_tables) == 2
        assert all(t.content_type == ContentType.TRANSACTION for t in transaction_tables)

    def test_get_tables_on_page(self, sample_tables):
        """Test getting tables on a specific page."""
        result = TableDetectionResult(tables=sample_tables)

        page_0_tables = result.get_tables_on_page(0)
        assert len(page_0_tables) == 2  # table_1 and table_3

        page_1_tables = result.get_tables_on_page(1)
        assert len(page_1_tables) == 1  # table_2

    def test_to_dict(self, sample_tables):
        """Test serialization to dictionary."""
        result = TableDetectionResult(
            tables=sample_tables,
            pages_processed=[0, 1, 2],
            pages_with_tables=[0, 1, 2],
            processing_time_ms=150,
        )

        d = result.to_dict()
        assert len(d["tables"]) == 3
        assert d["pages_processed"] == [0, 1, 2]
        assert d["processing_time_ms"] == 150
        assert d["summary"]["table_count"] == 3
        assert d["summary"]["transaction_table_count"] == 2


class TestEnumerations:
    """Tests for table-related enumerations."""

    def test_structure_type_values(self):
        """Test StructureType enum values."""
        assert StructureType.BORDERED.value == "bordered"
        assert StructureType.SEMI_BORDERED.value == "semi_bordered"
        assert StructureType.UNBORDERED.value == "unbordered"
        assert StructureType.SHADED.value == "shaded"
        assert StructureType.UNKNOWN.value == "unknown"

    def test_content_type_values(self):
        """Test ContentType enum values."""
        assert ContentType.TRANSACTION.value == "transaction"
        assert ContentType.SUMMARY.value == "summary"
        assert ContentType.ACCOUNT_INFO.value == "account_info"
        assert ContentType.OTHER.value == "other"

    def test_data_type_values(self):
        """Test DataType enum values."""
        assert DataType.DATE.value == "date"
        assert DataType.NUMERIC.value == "numeric"
        assert DataType.TEXT.value == "text"
        assert DataType.MIXED.value == "mixed"
        assert DataType.UNKNOWN.value == "unknown"
