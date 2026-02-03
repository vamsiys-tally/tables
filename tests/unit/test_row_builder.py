"""
Unit tests for row builder.

Tests the RowBuilder class and row merging logic.
"""

import pytest
from datetime import date
from decimal import Decimal
from unittest.mock import Mock

from tables.recognizer.row_builder import (
    RowBuilder,
    MergeContext,
    build_transactions,
)
from tables.recognizer.cell_extractor import Cell, ExtractedRow
from tables.models.table import TableDefinition, ColumnDefinition, DataType
from tables.models.transaction import SemanticType


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
        ColumnDefinition(
            column_id=4,
            x0=550.0,
            x1=650.0,
            header_text="Balance",
            semantic_type="balance",
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
    )


def create_extracted_row(row_index, cells_data, sample_columns, page_number=0):
    """Helper to create an ExtractedRow with specified cell data."""
    cells = {}
    for col in sample_columns:
        cell = Cell(column_index=col.column_id, column=col)
        if col.column_id in cells_data:
            for word in cells_data[col.column_id]:
                cell.add_word({"text": word, "x0": 0, "x1": 100, "top": 0, "bottom": 12})
        cells[col.column_id] = cell

    return ExtractedRow(
        row_index=row_index,
        page_number=page_number,
        cells=cells,
        y_position=row_index * 20,
        height=12,
    )


# =============================================================================
# MergeContext Tests
# =============================================================================

class TestMergeContext:
    """Tests for MergeContext class."""

    def test_from_columns_detects_date(self, sample_columns):
        """Test that date column is detected."""
        context = MergeContext.from_columns(sample_columns)
        assert context.has_date_column is True
        assert context.date_column_index == 0

    def test_from_columns_detects_balance(self, sample_columns):
        """Test that balance column is detected."""
        context = MergeContext.from_columns(sample_columns)
        assert context.has_balance_column is True
        assert context.balance_column_index == 4

    def test_from_columns_no_balance(self):
        """Test when no balance column exists."""
        columns = [
            ColumnDefinition(column_id=0, x0=0, x1=100, semantic_type="date"),
            ColumnDefinition(column_id=1, x0=100, x1=200, semantic_type="description"),
        ]
        context = MergeContext.from_columns(columns)
        assert context.has_balance_column is False
        assert context.balance_column_index is None


# =============================================================================
# RowBuilder Tests
# =============================================================================

class TestRowBuilder:
    """Tests for RowBuilder class."""

    def test_initialization(self, sample_table_definition):
        """Test default initialization."""
        builder = RowBuilder(sample_table_definition)
        assert builder.enable_merge is True
        assert builder.merge_context.has_date_column is True

    def test_initialization_merge_disabled(self, sample_table_definition):
        """Test initialization with merge disabled."""
        builder = RowBuilder(sample_table_definition, enable_merge=False)
        assert builder.enable_merge is False

    def test_build_single_transaction(self, sample_table_definition, sample_columns):
        """Test building a single transaction row."""
        row = create_extracted_row(
            0,
            {
                0: ["15/01/2024"],
                1: ["ATM", "Withdrawal"],
                2: ["5000.00"],
            },
            sample_columns,
        )

        builder = RowBuilder(sample_table_definition)
        transactions = builder.build_transactions([row])

        assert len(transactions) == 1
        txn = transactions[0]
        assert txn.transaction_date == date(2024, 1, 15)
        assert txn.description == "ATM Withdrawal"
        assert txn.debit_amount == Decimal("5000.00")

    def test_build_transaction_with_credit(self, sample_table_definition, sample_columns):
        """Test building a transaction with credit amount."""
        row = create_extracted_row(
            0,
            {
                0: ["16/01/2024"],
                1: ["Salary", "Credit"],
                3: ["50000.00"],
                4: ["55000.00"],
            },
            sample_columns,
        )

        builder = RowBuilder(sample_table_definition)
        transactions = builder.build_transactions([row])

        assert len(transactions) == 1
        txn = transactions[0]
        assert txn.credit_amount == Decimal("50000.00")
        assert txn.balance == Decimal("55000.00")

    def test_empty_rows_skipped(self, sample_table_definition, sample_columns):
        """Test that empty rows are skipped."""
        row = create_extracted_row(0, {}, sample_columns)

        builder = RowBuilder(sample_table_definition)
        transactions = builder.build_transactions([row])

        assert len(transactions) == 0

    def test_multiple_transactions(self, sample_table_definition, sample_columns):
        """Test building multiple transactions."""
        rows = [
            create_extracted_row(
                0,
                {0: ["15/01/2024"], 1: ["Transaction 1"], 2: ["1000"]},
                sample_columns,
            ),
            create_extracted_row(
                1,
                {0: ["16/01/2024"], 1: ["Transaction 2"], 3: ["500"]},
                sample_columns,
            ),
        ]

        builder = RowBuilder(sample_table_definition)
        transactions = builder.build_transactions(rows)

        assert len(transactions) == 2
        assert transactions[0].debit_amount == Decimal("1000")
        assert transactions[1].credit_amount == Decimal("500")


# =============================================================================
# Row Merging Tests
# =============================================================================

class TestRowMerging:
    """Tests for row merging logic."""

    def test_merge_wrapped_rows_by_empty_date(
        self, sample_table_definition, sample_columns
    ):
        """Test merging when continuation row has empty date."""
        rows = [
            create_extracted_row(
                0,
                {
                    0: ["15/01/2024"],
                    1: ["ATM Withdrawal"],
                    2: ["5000"],
                    4: ["45000"],
                },
                sample_columns,
            ),
            create_extracted_row(
                1,
                {
                    # No date - indicates continuation
                    1: ["Location: XYZ Branch"],
                    # No balance - indicates continuation
                },
                sample_columns,
            ),
        ]

        builder = RowBuilder(sample_table_definition, enable_merge=True)
        transactions = builder.build_transactions(rows)

        # Should merge into single transaction
        assert len(transactions) == 1
        txn = transactions[0]
        assert txn.transaction_date == date(2024, 1, 15)
        assert "ATM Withdrawal" in txn.description
        assert "Location: XYZ Branch" in txn.description
        assert txn.is_merged is True

    def test_merge_three_row_transaction(
        self, sample_table_definition, sample_columns
    ):
        """Test merging transaction spanning 3 visual rows."""
        rows = [
            create_extracted_row(
                0,
                {0: ["15/01/2024"], 1: ["Line 1"], 2: ["5000"], 4: ["45000"]},
                sample_columns,
            ),
            create_extracted_row(
                1,
                {1: ["Line 2"]},  # Continuation
                sample_columns,
            ),
            create_extracted_row(
                2,
                {1: ["Line 3"]},  # Continuation
                sample_columns,
            ),
        ]

        builder = RowBuilder(sample_table_definition, enable_merge=True)
        transactions = builder.build_transactions(rows)

        assert len(transactions) == 1
        txn = transactions[0]
        assert "Line 1" in txn.description
        assert "Line 2" in txn.description
        assert "Line 3" in txn.description
        assert len(txn.source_row_indices) == 3

    def test_no_merge_when_disabled(
        self, sample_table_definition, sample_columns
    ):
        """Test that merge is skipped when disabled."""
        rows = [
            create_extracted_row(
                0,
                {0: ["15/01/2024"], 1: ["Transaction"]},
                sample_columns,
            ),
            create_extracted_row(
                1,
                {1: ["Continuation"]},  # Would normally merge
                sample_columns,
            ),
        ]

        builder = RowBuilder(sample_table_definition, enable_merge=False)
        transactions = builder.build_transactions(rows)

        # Should NOT merge
        assert len(transactions) == 2

    def test_separate_transactions_not_merged(
        self, sample_table_definition, sample_columns
    ):
        """Test that separate transactions are not merged."""
        rows = [
            create_extracted_row(
                0,
                {0: ["15/01/2024"], 1: ["Transaction 1"], 2: ["1000"], 4: ["49000"]},
                sample_columns,
            ),
            create_extracted_row(
                1,
                {0: ["16/01/2024"], 1: ["Transaction 2"], 3: ["500"], 4: ["49500"]},
                sample_columns,
            ),
        ]

        builder = RowBuilder(sample_table_definition, enable_merge=True)
        transactions = builder.build_transactions(rows)

        # Should NOT merge - both have dates
        assert len(transactions) == 2


# =============================================================================
# Value Parsing Tests
# =============================================================================

class TestValueParsing:
    """Tests for value parsing during row building."""

    def test_parse_date_formats(self, sample_table_definition, sample_columns):
        """Test various date format parsing."""
        test_cases = [
            ("15/01/2024", date(2024, 1, 15)),
            ("15-Jan-2024", date(2024, 1, 15)),
            ("2024-01-15", date(2024, 1, 15)),
        ]

        builder = RowBuilder(sample_table_definition)

        for date_str, expected in test_cases:
            row = create_extracted_row(
                0, {0: [date_str], 1: ["Test"]}, sample_columns
            )
            transactions = builder.build_transactions([row])
            assert transactions[0].transaction_date == expected

    def test_parse_amount_formats(self, sample_table_definition, sample_columns):
        """Test various amount format parsing."""
        test_cases = [
            ("5000", Decimal("5000")),
            ("5,000.00", Decimal("5000.00")),
            ("1,00,000", Decimal("100000")),  # Indian format
            ("₹ 5000", Decimal("5000")),
        ]

        builder = RowBuilder(sample_table_definition)

        for amount_str, expected in test_cases:
            row = create_extracted_row(
                0, {0: ["15/01/2024"], 1: ["Test"], 2: [amount_str]}, sample_columns
            )
            transactions = builder.build_transactions([row])
            assert transactions[0].debit_amount == expected

    def test_invalid_date_generates_warning(
        self, sample_table_definition, sample_columns
    ):
        """Test that invalid date generates a warning."""
        row = create_extracted_row(
            0, {0: ["invalid_date"], 1: ["Test"], 2: ["1000"]}, sample_columns
        )

        builder = RowBuilder(sample_table_definition)
        transactions = builder.build_transactions([row])
        warnings = builder.get_warnings()

        assert len(transactions) == 1
        assert transactions[0].transaction_date is None
        assert len(warnings) > 0
        assert any("date" in w.column_name.lower() for w in warnings)

    def test_invalid_amount_generates_warning(
        self, sample_table_definition, sample_columns
    ):
        """Test that invalid amount generates a warning."""
        row = create_extracted_row(
            0, {0: ["15/01/2024"], 1: ["Test"], 2: ["not_a_number"]}, sample_columns
        )

        builder = RowBuilder(sample_table_definition)
        transactions = builder.build_transactions([row])
        warnings = builder.get_warnings()

        assert len(transactions) == 1
        assert transactions[0].debit_amount is None
        assert len(warnings) > 0


# =============================================================================
# Column Schema Building Tests
# =============================================================================

class TestColumnSchemaBuilding:
    """Tests for building column schemas."""

    def test_build_column_schemas(self, sample_table_definition):
        """Test building column schemas from table definition."""
        builder = RowBuilder(sample_table_definition)
        schemas = builder.build_column_schemas()

        assert len(schemas) == 5
        assert schemas[0].semantic_type == SemanticType.DATE
        assert schemas[1].semantic_type == SemanticType.DESCRIPTION
        assert schemas[2].semantic_type == SemanticType.DEBIT
        assert schemas[3].semantic_type == SemanticType.CREDIT
        assert schemas[4].semantic_type == SemanticType.BALANCE

    def test_column_schema_data_types(self, sample_table_definition):
        """Test that data types are correctly assigned."""
        builder = RowBuilder(sample_table_definition)
        schemas = builder.build_column_schemas()

        assert schemas[0].data_type == "date"  # Date column
        assert schemas[1].data_type == "str"  # Description
        assert schemas[2].data_type == "Decimal"  # Debit


# =============================================================================
# Convenience Function Tests
# =============================================================================

class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_build_transactions_function(
        self, sample_table_definition, sample_columns
    ):
        """Test build_transactions convenience function."""
        rows = [
            create_extracted_row(
                0, {0: ["15/01/2024"], 1: ["Test"], 2: ["1000"]}, sample_columns
            ),
        ]

        transactions, warnings = build_transactions(
            sample_table_definition, rows
        )

        assert len(transactions) == 1
        assert isinstance(warnings, list)


# =============================================================================
# Edge Cases
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_row_list(self, sample_table_definition):
        """Test building from empty row list."""
        builder = RowBuilder(sample_table_definition)
        transactions = builder.build_transactions([])
        assert transactions == []

    def test_row_with_all_nil_values(self, sample_table_definition, sample_columns):
        """Test row where all values are nil/empty placeholders."""
        row = create_extracted_row(
            0, {0: ["-"], 1: ["nil"], 2: ["--"]}, sample_columns
        )

        builder = RowBuilder(sample_table_definition)
        transactions = builder.build_transactions([row])

        # Should produce transaction but with no parsed values
        assert len(transactions) == 1
        assert transactions[0].transaction_date is None
        assert transactions[0].debit_amount is None
