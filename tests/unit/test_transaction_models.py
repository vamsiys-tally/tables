"""
Unit tests for transaction data models.

Tests TransactionRow, TransactionTable, ColumnSchema, and related models.
"""

import pytest
from datetime import date
from decimal import Decimal

from tables.models.transaction import (
    TransactionRow,
    TransactionTable,
    ColumnSchema,
    ExtractionResult,
    ParseWarning,
    SemanticType,
    ParseStatus,
)


# =============================================================================
# TransactionRow Tests
# =============================================================================

class TestTransactionRow:
    """Tests for TransactionRow dataclass."""

    def test_default_initialization(self):
        """Test default TransactionRow creation."""
        row = TransactionRow()
        assert row.row_id == 0
        assert row.source_page == 0
        assert row.source_row_indices == []
        assert row.transaction_date is None
        assert row.description == ""
        assert row.debit_amount is None
        assert row.credit_amount is None
        assert row.balance is None
        assert row.confidence == 1.0

    def test_full_initialization(self):
        """Test TransactionRow with all fields."""
        row = TransactionRow(
            row_id=5,
            source_page=2,
            source_row_indices=[10, 11],
            transaction_date=date(2024, 1, 15),
            value_date=date(2024, 1, 16),
            description="ATM Withdrawal",
            reference="REF123456",
            debit_amount=Decimal("5000.00"),
            credit_amount=None,
            balance=Decimal("45000.00"),
            confidence=0.95,
            raw_cells={"date": "15/01/2024", "description": "ATM Withdrawal"},
        )
        assert row.row_id == 5
        assert row.source_page == 2
        assert row.transaction_date == date(2024, 1, 15)
        assert row.debit_amount == Decimal("5000.00")

    def test_has_amount_property(self):
        """Test has_amount property."""
        row_no_amount = TransactionRow()
        assert row_no_amount.has_amount is False

        row_debit = TransactionRow(debit_amount=Decimal("100"))
        assert row_debit.has_amount is True

        row_credit = TransactionRow(credit_amount=Decimal("100"))
        assert row_credit.has_amount is True

    def test_amount_property(self):
        """Test amount property calculation."""
        # Debit should be negative
        row_debit = TransactionRow(debit_amount=Decimal("100"))
        assert row_debit.amount == Decimal("-100")

        # Credit should be positive
        row_credit = TransactionRow(credit_amount=Decimal("100"))
        assert row_credit.amount == Decimal("100")

        # Both present: net amount
        row_both = TransactionRow(
            debit_amount=Decimal("50"),
            credit_amount=Decimal("100"),
        )
        assert row_both.amount == Decimal("50")

        # No amount
        row_none = TransactionRow()
        assert row_none.amount is None

    def test_is_debit_and_is_credit(self):
        """Test is_debit and is_credit properties."""
        row_debit = TransactionRow(debit_amount=Decimal("100"))
        assert row_debit.is_debit is True
        assert row_debit.is_credit is False

        row_credit = TransactionRow(credit_amount=Decimal("100"))
        assert row_credit.is_debit is False
        assert row_credit.is_credit is True

    def test_is_merged_property(self):
        """Test is_merged property."""
        row_single = TransactionRow(source_row_indices=[5])
        assert row_single.is_merged is False

        row_merged = TransactionRow(source_row_indices=[5, 6, 7])
        assert row_merged.is_merged is True

    def test_add_warning(self):
        """Test add_warning method."""
        row = TransactionRow()
        row.add_warning("DATE_PARSE_FAILED")
        row.add_warning("AMOUNT_PARSE_FAILED")
        row.add_warning("DATE_PARSE_FAILED")  # Duplicate
        assert len(row.warnings) == 2

    def test_to_dict(self):
        """Test serialization to dictionary."""
        row = TransactionRow(
            row_id=1,
            transaction_date=date(2024, 1, 15),
            debit_amount=Decimal("1000.50"),
            description="Test",
        )
        data = row.to_dict()
        assert data["row_id"] == 1
        assert data["transaction_date"] == "2024-01-15"
        assert data["debit_amount"] == "1000.50"
        assert data["description"] == "Test"

    def test_from_dict(self):
        """Test deserialization from dictionary."""
        data = {
            "row_id": 5,
            "transaction_date": "2024-01-15",
            "debit_amount": "5000.00",
            "description": "ATM",
        }
        row = TransactionRow.from_dict(data)
        assert row.row_id == 5
        assert row.transaction_date == date(2024, 1, 15)
        assert row.debit_amount == Decimal("5000.00")


# =============================================================================
# ColumnSchema Tests
# =============================================================================

class TestColumnSchema:
    """Tests for ColumnSchema dataclass."""

    def test_initialization(self):
        """Test ColumnSchema creation."""
        schema = ColumnSchema(
            name="Transaction Date",
            semantic_type=SemanticType.DATE,
            data_type="date",
            nullable=False,
            source_column_index=0,
        )
        assert schema.name == "Transaction Date"
        assert schema.semantic_type == SemanticType.DATE
        assert schema.data_type == "date"
        assert schema.nullable is False

    def test_to_dict(self):
        """Test serialization."""
        schema = ColumnSchema(
            name="Amount",
            semantic_type=SemanticType.DEBIT,
            data_type="Decimal",
        )
        data = schema.to_dict()
        assert data["name"] == "Amount"
        assert data["semantic_type"] == "debit"


# =============================================================================
# TransactionTable Tests
# =============================================================================

class TestTransactionTable:
    """Tests for TransactionTable dataclass."""

    @pytest.fixture
    def sample_rows(self):
        """Create sample transaction rows."""
        return [
            TransactionRow(
                row_id=0,
                transaction_date=date(2024, 1, 10),
                description="Transaction 1",
                debit_amount=Decimal("1000"),
            ),
            TransactionRow(
                row_id=1,
                transaction_date=date(2024, 1, 15),
                description="Transaction 2",
                credit_amount=Decimal("500"),
            ),
            TransactionRow(
                row_id=2,
                transaction_date=date(2024, 1, 20),
                description="Transaction 3",
                debit_amount=Decimal("200"),
            ),
        ]

    def test_default_initialization(self):
        """Test default TransactionTable creation."""
        table = TransactionTable()
        assert table.rows == []
        assert table.columns == []
        assert table.row_count == 0

    def test_initialization_with_rows(self, sample_rows):
        """Test TransactionTable with rows."""
        table = TransactionTable(
            table_id="test_table",
            source_pages=[1, 2],
            rows=sample_rows,
        )
        assert table.row_count == 3
        assert len(table.rows) == 3

    def test_date_range_property(self, sample_rows):
        """Test date_range property calculation."""
        table = TransactionTable(rows=sample_rows)
        assert table.date_range == (date(2024, 1, 10), date(2024, 1, 20))

    def test_total_debit_property(self, sample_rows):
        """Test total_debit property calculation."""
        table = TransactionTable(rows=sample_rows)
        assert table.total_debit == Decimal("1200")  # 1000 + 200

    def test_total_credit_property(self, sample_rows):
        """Test total_credit property calculation."""
        table = TransactionTable(rows=sample_rows)
        assert table.total_credit == Decimal("500")

    def test_net_amount_property(self, sample_rows):
        """Test net_amount property calculation."""
        table = TransactionTable(rows=sample_rows)
        # 500 (credit) - 1200 (debit) = -700
        assert table.net_amount == Decimal("-700")

    def test_add_row(self):
        """Test add_row method."""
        table = TransactionTable()
        row = TransactionRow(
            row_id=0,
            transaction_date=date(2024, 1, 15),
            debit_amount=Decimal("100"),
        )
        table.add_row(row)
        assert table.row_count == 1
        assert table.total_debit == Decimal("100")

    def test_add_warning(self):
        """Test add_warning method."""
        table = TransactionTable()
        warning = ParseWarning(
            row_id=0,
            column_name="date",
            message="Parse failed",
        )
        table.add_warning(warning)
        assert len(table.parse_warnings) == 1

    def test_get_column_by_type(self):
        """Test get_column_by_type method."""
        table = TransactionTable(columns=[
            ColumnSchema(name="Date", semantic_type=SemanticType.DATE, data_type="date"),
            ColumnSchema(name="Debit", semantic_type=SemanticType.DEBIT, data_type="Decimal"),
        ])
        result = table.get_column_by_type(SemanticType.DATE)
        assert result is not None
        assert result.name == "Date"

        result = table.get_column_by_type(SemanticType.REFERENCE)
        assert result is None

    def test_has_column_type(self):
        """Test has_column_type method."""
        table = TransactionTable(columns=[
            ColumnSchema(name="Date", semantic_type=SemanticType.DATE, data_type="date"),
        ])
        assert table.has_column_type(SemanticType.DATE) is True
        assert table.has_column_type(SemanticType.DEBIT) is False

    def test_to_dict(self, sample_rows):
        """Test serialization."""
        table = TransactionTable(
            table_id="test_table",
            source_pages=[1],
            rows=sample_rows,
        )
        data = table.to_dict()
        assert data["table_id"] == "test_table"
        assert data["source_pages"] == [1]
        assert len(data["rows"]) == 3
        assert data["metadata"]["row_count"] == 3

    def test_from_dict(self):
        """Test deserialization."""
        data = {
            "table_id": "test",
            "source_pages": [1, 2],
            "columns": [
                {
                    "name": "Date",
                    "semantic_type": "date",
                    "data_type": "date",
                    "nullable": False,
                    "source_column_index": 0,
                    "source_header_text": "Date",
                }
            ],
            "rows": [
                {
                    "row_id": 0,
                    "transaction_date": "2024-01-15",
                    "description": "Test",
                }
            ],
            "confidence_score": 0.9,
            "parse_warnings": [],
        }
        table = TransactionTable.from_dict(data)
        assert table.table_id == "test"
        assert table.row_count == 1
        assert len(table.columns) == 1


# =============================================================================
# ExtractionResult Tests
# =============================================================================

class TestExtractionResult:
    """Tests for ExtractionResult dataclass."""

    def test_default_initialization(self):
        """Test default ExtractionResult creation."""
        result = ExtractionResult()
        assert result.tables == []
        assert result.table_count == 0
        assert result.total_transactions == 0
        assert result.has_errors is False
        assert result.success is True

    def test_total_transactions(self):
        """Test total_transactions across multiple tables."""
        rows1 = [TransactionRow(row_id=i) for i in range(5)]
        rows2 = [TransactionRow(row_id=i) for i in range(3)]

        result = ExtractionResult(tables=[
            TransactionTable(rows=rows1),
            TransactionTable(rows=rows2),
        ])
        assert result.total_transactions == 8

    def test_has_errors(self):
        """Test has_errors property."""
        result = ExtractionResult()
        assert result.has_errors is False
        assert result.success is True

        result.add_error("Something failed")
        assert result.has_errors is True
        assert result.success is False

    def test_add_table(self):
        """Test add_table method."""
        result = ExtractionResult()
        table = TransactionTable(table_id="test")
        result.add_table(table)
        assert result.table_count == 1

    def test_to_dict(self):
        """Test serialization."""
        result = ExtractionResult(
            tables=[TransactionTable(table_id="test")],
            processing_time_ms=100,
        )
        data = result.to_dict()
        assert data["success"] is True
        assert len(data["tables"]) == 1
        assert data["processing_time_ms"] == 100


# =============================================================================
# ParseWarning Tests
# =============================================================================

class TestParseWarning:
    """Tests for ParseWarning dataclass."""

    def test_initialization(self):
        """Test ParseWarning creation."""
        warning = ParseWarning(
            row_id=5,
            column_name="date",
            message="Could not parse date",
            raw_value="invalid",
            code="DATE_PARSE_FAILED",
        )
        assert warning.row_id == 5
        assert warning.column_name == "date"
        assert warning.message == "Could not parse date"

    def test_to_dict(self):
        """Test serialization."""
        warning = ParseWarning(
            row_id=0,
            column_name="amount",
            message="Parse failed",
        )
        data = warning.to_dict()
        assert data["row_id"] == 0
        assert data["column_name"] == "amount"


# =============================================================================
# SemanticType Tests
# =============================================================================

class TestSemanticType:
    """Tests for SemanticType enum."""

    def test_enum_values(self):
        """Test enum value strings."""
        assert SemanticType.DATE.value == "date"
        assert SemanticType.DEBIT.value == "debit"
        assert SemanticType.CREDIT.value == "credit"
        assert SemanticType.BALANCE.value == "balance"
        assert SemanticType.DESCRIPTION.value == "description"

    def test_enum_from_string(self):
        """Test creating enum from string."""
        assert SemanticType("date") == SemanticType.DATE
        assert SemanticType("debit") == SemanticType.DEBIT
