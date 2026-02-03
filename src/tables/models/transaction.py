"""
Transaction Data Models.

This module defines data structures for representing extracted transaction data
from bank statement tables. These models are the output of Module 3 (Structure
Recognition) and contain the final parsed transaction data.

Model Hierarchy:
    TransactionTable
    ├── ColumnSchema (multiple)
    ├── TransactionRow (multiple)
    │   ├── Parsed values (date, amounts, etc.)
    │   └── RawCells (original text)
    └── ExtractionMetadata

Key Concepts:
    - TransactionRow: Single transaction with parsed fields
    - TransactionTable: Complete extracted table with all transactions
    - ColumnSchema: Describes column type and parsing rules
    - ParseWarning: Warning about parsing issues in specific cells

Example Usage:
    >>> from tables.models.transaction import TransactionTable, TransactionRow
    >>> from decimal import Decimal
    >>> from datetime import date
    >>>
    >>> row = TransactionRow(
    ...     row_id=0,
    ...     transaction_date=date(2024, 1, 15),
    ...     description="ATM Withdrawal",
    ...     debit_amount=Decimal("5000.00"),
    ... )
    >>> table = TransactionTable(rows=[row])
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from enum import Enum
from typing import Optional, Any, Tuple
import uuid


# =============================================================================
# Enumerations
# =============================================================================

class SemanticType(str, Enum):
    """
    Semantic type of a column in a transaction table.

    Maps to the semantic types detected in Module 2 but provides
    a strict enum for type safety in Module 3.

    Values:
        DATE: Transaction or posting date
        VALUE_DATE: Value date (when transaction takes effect)
        DESCRIPTION: Transaction description/narration
        REFERENCE: Reference number, UTR, cheque number
        DEBIT: Debit/withdrawal amount
        CREDIT: Credit/deposit amount
        BALANCE: Running balance
        AMOUNT: Generic amount (when debit/credit not separated)
        OTHER: Column type not recognized
    """

    DATE = "date"
    VALUE_DATE = "value_date"
    DESCRIPTION = "description"
    REFERENCE = "reference"
    DEBIT = "debit"
    CREDIT = "credit"
    BALANCE = "balance"
    AMOUNT = "amount"
    OTHER = "other"


class ParseStatus(str, Enum):
    """
    Status of value parsing for a cell.

    Values:
        SUCCESS: Value parsed successfully
        EMPTY: Cell was empty (valid for optional fields)
        FAILED: Parsing failed, raw value preserved
        PARTIAL: Partial parsing (some data extracted)
    """

    SUCCESS = "success"
    EMPTY = "empty"
    FAILED = "failed"
    PARTIAL = "partial"


# =============================================================================
# Parse Warning
# =============================================================================

@dataclass
class ParseWarning:
    """
    Warning about a parsing issue in a specific location.

    Captures details about parsing failures or anomalies for
    debugging and quality assessment.

    Attributes:
        row_id: Row where warning occurred
        column_name: Column name (or semantic type)
        message: Human-readable warning message
        raw_value: Original cell value that caused the warning
        code: Machine-readable warning code
    """

    row_id: int
    column_name: str
    message: str
    raw_value: str = ""
    code: str = "PARSE_WARNING"

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "row_id": self.row_id,
            "column_name": self.column_name,
            "message": self.message,
            "raw_value": self.raw_value,
            "code": self.code,
        }


# =============================================================================
# Column Schema
# =============================================================================

@dataclass
class ColumnSchema:
    """
    Schema definition for a column in the extracted table.

    Maps from the detected column structure to the standardized
    output schema.

    Attributes:
        name: Display name for the column
        semantic_type: Semantic type (date, debit, etc.)
        data_type: Python data type name (date, Decimal, str)
        nullable: Whether column allows null values
        source_column_index: Original column index from detection
        source_header_text: Original header text from PDF

    Example:
        >>> schema = ColumnSchema(
        ...     name="Transaction Date",
        ...     semantic_type=SemanticType.DATE,
        ...     data_type="date",
        ...     nullable=False,
        ...     source_column_index=0,
        ... )
    """

    name: str
    semantic_type: SemanticType
    data_type: str  # "date", "Decimal", "str"
    nullable: bool = True
    source_column_index: int = 0
    source_header_text: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "name": self.name,
            "semantic_type": self.semantic_type.value,
            "data_type": self.data_type,
            "nullable": self.nullable,
            "source_column_index": self.source_column_index,
            "source_header_text": self.source_header_text,
        }


# =============================================================================
# Transaction Row
# =============================================================================

@dataclass
class TransactionRow:
    """
    A single transaction row extracted from a bank statement.

    Contains both the parsed values and metadata about the extraction.
    All amount fields use Decimal for financial precision.

    Attributes:
        row_id: Sequential row identifier (0-indexed)
        source_page: Page number where row was found
        source_row_indices: Original visual row indices (for merged rows)

        transaction_date: Primary transaction date
        value_date: Value date (when transaction takes effect)
        description: Transaction description/narration
        reference: Reference number (UTR, cheque number, etc.)
        debit_amount: Debit/withdrawal amount (positive number)
        credit_amount: Credit/deposit amount (positive number)
        balance: Running balance after transaction

        confidence: Extraction confidence score (0.0-1.0)
        warnings: List of warning codes for this row
        raw_cells: Original cell values keyed by semantic type

    Example:
        >>> from decimal import Decimal
        >>> from datetime import date
        >>> row = TransactionRow(
        ...     row_id=0,
        ...     source_page=1,
        ...     transaction_date=date(2024, 1, 15),
        ...     description="ATM Withdrawal - HDFC",
        ...     debit_amount=Decimal("5000.00"),
        ...     balance=Decimal("45000.00"),
        ...     confidence=0.95,
        ... )
    """

    # Identity
    row_id: int = 0
    source_page: int = 0
    source_row_indices: list[int] = field(default_factory=list)

    # Standard transaction fields
    transaction_date: Optional[date] = None
    value_date: Optional[date] = None
    description: str = ""
    reference: Optional[str] = None
    debit_amount: Optional[Decimal] = None
    credit_amount: Optional[Decimal] = None
    balance: Optional[Decimal] = None

    # Quality metadata
    confidence: float = 1.0
    warnings: list[str] = field(default_factory=list)

    # Raw values for debugging/audit
    raw_cells: dict[str, str] = field(default_factory=dict)

    @property
    def has_amount(self) -> bool:
        """Check if row has any amount (debit or credit)."""
        return self.debit_amount is not None or self.credit_amount is not None

    @property
    def amount(self) -> Optional[Decimal]:
        """
        Get the transaction amount (positive for credit, negative for debit).

        Returns net amount - if both debit and credit exist, returns difference.
        """
        if self.credit_amount is not None and self.debit_amount is not None:
            return self.credit_amount - self.debit_amount
        elif self.credit_amount is not None:
            return self.credit_amount
        elif self.debit_amount is not None:
            return -self.debit_amount
        return None

    @property
    def is_debit(self) -> bool:
        """Check if this is a debit transaction."""
        return self.debit_amount is not None and self.debit_amount > 0

    @property
    def is_credit(self) -> bool:
        """Check if this is a credit transaction."""
        return self.credit_amount is not None and self.credit_amount > 0

    @property
    def is_merged(self) -> bool:
        """Check if this row was merged from multiple visual rows."""
        return len(self.source_row_indices) > 1

    def add_warning(self, warning: str) -> None:
        """Add a warning code to this row."""
        if warning not in self.warnings:
            self.warnings.append(warning)

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize to dictionary.

        Converts date and Decimal values to JSON-compatible formats.

        Returns:
            Dictionary representation suitable for JSON serialization
        """
        return {
            "row_id": self.row_id,
            "source_page": self.source_page,
            "source_row_indices": self.source_row_indices,
            "transaction_date": self.transaction_date.isoformat() if self.transaction_date else None,
            "value_date": self.value_date.isoformat() if self.value_date else None,
            "description": self.description,
            "reference": self.reference,
            "debit_amount": str(self.debit_amount) if self.debit_amount is not None else None,
            "credit_amount": str(self.credit_amount) if self.credit_amount is not None else None,
            "balance": str(self.balance) if self.balance is not None else None,
            "confidence": self.confidence,
            "warnings": self.warnings,
            "raw_cells": self.raw_cells,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TransactionRow:
        """
        Create a TransactionRow from a dictionary.

        Args:
            data: Dictionary representation

        Returns:
            TransactionRow instance
        """
        def parse_date(val: Optional[str]) -> Optional[date]:
            if val is None:
                return None
            return date.fromisoformat(val)

        def parse_decimal(val: Optional[str]) -> Optional[Decimal]:
            if val is None:
                return None
            return Decimal(val)

        return cls(
            row_id=data.get("row_id", 0),
            source_page=data.get("source_page", 0),
            source_row_indices=data.get("source_row_indices", []),
            transaction_date=parse_date(data.get("transaction_date")),
            value_date=parse_date(data.get("value_date")),
            description=data.get("description", ""),
            reference=data.get("reference"),
            debit_amount=parse_decimal(data.get("debit_amount")),
            credit_amount=parse_decimal(data.get("credit_amount")),
            balance=parse_decimal(data.get("balance")),
            confidence=data.get("confidence", 1.0),
            warnings=data.get("warnings", []),
            raw_cells=data.get("raw_cells", {}),
        )


# =============================================================================
# Transaction Table
# =============================================================================

@dataclass
class TransactionTable:
    """
    Complete extracted transaction table from a bank statement.

    Contains all parsed transaction rows along with schema information
    and extraction metadata.

    Attributes:
        table_id: Unique identifier (matches TableDefinition.table_id)
        source_pages: List of page numbers this table came from
        columns: Schema for each column
        rows: List of extracted transaction rows

        row_count: Number of rows extracted
        date_range: Tuple of (earliest_date, latest_date)
        total_debit: Sum of all debit amounts
        total_credit: Sum of all credit amounts

        confidence_score: Overall extraction confidence (0.0-1.0)
        parse_warnings: List of parsing warnings

    Example:
        >>> table = TransactionTable(
        ...     table_id="table_abc123",
        ...     source_pages=[1, 2],
        ...     rows=[row1, row2, row3],
        ... )
        >>> table.row_count
        3
    """

    # Identity
    table_id: str = field(default_factory=lambda: f"txn_{uuid.uuid4().hex[:8]}")
    source_pages: list[int] = field(default_factory=list)

    # Schema
    columns: list[ColumnSchema] = field(default_factory=list)

    # Data
    rows: list[TransactionRow] = field(default_factory=list)

    # Computed metadata (updated via compute_metadata)
    _row_count: int = 0
    _date_range: Optional[Tuple[date, date]] = None
    _total_debit: Optional[Decimal] = None
    _total_credit: Optional[Decimal] = None

    # Quality metrics
    confidence_score: float = 1.0
    parse_warnings: list[ParseWarning] = field(default_factory=list)

    def __post_init__(self):
        """Compute metadata after initialization."""
        self.compute_metadata()

    @property
    def row_count(self) -> int:
        """Get the number of rows."""
        return len(self.rows)

    @property
    def date_range(self) -> Optional[Tuple[date, date]]:
        """Get the date range (earliest, latest) of transactions."""
        return self._date_range

    @property
    def total_debit(self) -> Optional[Decimal]:
        """Get the sum of all debit amounts."""
        return self._total_debit

    @property
    def total_credit(self) -> Optional[Decimal]:
        """Get the sum of all credit amounts."""
        return self._total_credit

    @property
    def net_amount(self) -> Optional[Decimal]:
        """Get net amount (total_credit - total_debit)."""
        if self._total_credit is not None and self._total_debit is not None:
            return self._total_credit - self._total_debit
        return None

    def compute_metadata(self) -> None:
        """
        Compute aggregated metadata from rows.

        Updates date_range, total_debit, total_credit based on row data.
        """
        if not self.rows:
            self._date_range = None
            self._total_debit = None
            self._total_credit = None
            return

        # Compute date range
        dates = [r.transaction_date for r in self.rows if r.transaction_date]
        if dates:
            self._date_range = (min(dates), max(dates))
        else:
            self._date_range = None

        # Compute totals
        debits = [r.debit_amount for r in self.rows if r.debit_amount is not None]
        credits = [r.credit_amount for r in self.rows if r.credit_amount is not None]

        self._total_debit = sum(debits, Decimal("0")) if debits else None
        self._total_credit = sum(credits, Decimal("0")) if credits else None

    def add_row(self, row: TransactionRow) -> None:
        """
        Add a row and update metadata.

        Args:
            row: TransactionRow to add
        """
        self.rows.append(row)
        self.compute_metadata()

    def add_warning(self, warning: ParseWarning) -> None:
        """
        Add a parse warning.

        Args:
            warning: ParseWarning to add
        """
        self.parse_warnings.append(warning)

    def get_column_by_type(self, semantic_type: SemanticType) -> Optional[ColumnSchema]:
        """
        Find a column by semantic type.

        Args:
            semantic_type: Type to search for

        Returns:
            First matching ColumnSchema or None
        """
        for col in self.columns:
            if col.semantic_type == semantic_type:
                return col
        return None

    def has_column_type(self, semantic_type: SemanticType) -> bool:
        """Check if table has a column of the specified type."""
        return self.get_column_by_type(semantic_type) is not None

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize to dictionary.

        Returns:
            Dictionary representation suitable for JSON serialization
        """
        return {
            "table_id": self.table_id,
            "source_pages": self.source_pages,
            "columns": [col.to_dict() for col in self.columns],
            "rows": [row.to_dict() for row in self.rows],
            "metadata": {
                "row_count": self.row_count,
                "date_range": (
                    [self._date_range[0].isoformat(), self._date_range[1].isoformat()]
                    if self._date_range else None
                ),
                "total_debit": str(self._total_debit) if self._total_debit is not None else None,
                "total_credit": str(self._total_credit) if self._total_credit is not None else None,
                "net_amount": str(self.net_amount) if self.net_amount is not None else None,
            },
            "confidence_score": self.confidence_score,
            "parse_warnings": [w.to_dict() for w in self.parse_warnings],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TransactionTable:
        """
        Create a TransactionTable from a dictionary.

        Args:
            data: Dictionary representation

        Returns:
            TransactionTable instance
        """
        columns = [
            ColumnSchema(
                name=col["name"],
                semantic_type=SemanticType(col["semantic_type"]),
                data_type=col["data_type"],
                nullable=col.get("nullable", True),
                source_column_index=col.get("source_column_index", 0),
                source_header_text=col.get("source_header_text", ""),
            )
            for col in data.get("columns", [])
        ]

        rows = [
            TransactionRow.from_dict(row)
            for row in data.get("rows", [])
        ]

        warnings = [
            ParseWarning(
                row_id=w["row_id"],
                column_name=w["column_name"],
                message=w["message"],
                raw_value=w.get("raw_value", ""),
                code=w.get("code", "PARSE_WARNING"),
            )
            for w in data.get("parse_warnings", [])
        ]

        return cls(
            table_id=data.get("table_id", f"txn_{uuid.uuid4().hex[:8]}"),
            source_pages=data.get("source_pages", []),
            columns=columns,
            rows=rows,
            confidence_score=data.get("confidence_score", 1.0),
            parse_warnings=warnings,
        )


# =============================================================================
# Extraction Result
# =============================================================================

@dataclass
class ExtractionResult:
    """
    Complete result of transaction extraction from a document.

    Contains all extracted tables and overall processing metadata.

    Attributes:
        tables: List of extracted TransactionTable objects
        source_table_ids: IDs of TableDefinitions that were processed
        pages_processed: List of page numbers processed
        processing_time_ms: Time taken for extraction
        overall_confidence: Weighted average confidence across tables
        warnings: Global warnings (not table-specific)
        errors: List of error messages if any tables failed

    Example:
        >>> result = ExtractionResult(tables=[table1, table2])
        >>> result.total_transactions
        150
    """

    tables: list[TransactionTable] = field(default_factory=list)
    source_table_ids: list[str] = field(default_factory=list)
    pages_processed: list[int] = field(default_factory=list)
    processing_time_ms: int = 0
    overall_confidence: float = 1.0
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def table_count(self) -> int:
        """Get the number of extracted tables."""
        return len(self.tables)

    @property
    def total_transactions(self) -> int:
        """Get total number of transactions across all tables."""
        return sum(t.row_count for t in self.tables)

    @property
    def has_errors(self) -> bool:
        """Check if any errors occurred during extraction."""
        return len(self.errors) > 0

    @property
    def success(self) -> bool:
        """Check if extraction was successful (no errors)."""
        return not self.has_errors

    def add_table(self, table: TransactionTable) -> None:
        """Add an extracted table."""
        self.tables.append(table)

    def add_warning(self, warning: str) -> None:
        """Add a global warning."""
        if warning not in self.warnings:
            self.warnings.append(warning)

    def add_error(self, error: str) -> None:
        """Add an error message."""
        if error not in self.errors:
            self.errors.append(error)

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize to dictionary.

        Returns:
            Dictionary representation suitable for JSON serialization
        """
        return {
            "success": self.success,
            "tables": [t.to_dict() for t in self.tables],
            "source_table_ids": self.source_table_ids,
            "pages_processed": self.pages_processed,
            "processing_time_ms": self.processing_time_ms,
            "overall_confidence": self.overall_confidence,
            "warnings": self.warnings,
            "errors": self.errors,
            "summary": {
                "table_count": self.table_count,
                "total_transactions": self.total_transactions,
            },
        }
