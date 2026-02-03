"""
Table Definition Models.

This module defines data structures for representing detected tables
and their structural components. These models are the output of
Module 2 (Table Detection) and the input to Module 3 (Structure Recognition).

Model Hierarchy:
    TableDefinition
    ├── ColumnDefinition (multiple)
    ├── BoundingBox (per page)
    └── TableMetadata

Key Concepts:
    - TableDefinition: Complete table structure with columns and boundaries
    - ColumnDefinition: Individual column with position and semantic type
    - StructureType: Visual table type (bordered, semi-bordered, unbordered)
    - ContentType: Semantic table type (transaction, summary, account_info)

Example Usage:
    >>> from tables.models.table import TableDefinition, ColumnDefinition
    >>> from tables.utils.geometry import BoundingBox
    >>>
    >>> column = ColumnDefinition(
    ...     column_id=0,
    ...     x0=50.0,
    ...     x1=150.0,
    ...     header_text="Date",
    ...     semantic_type="date",
    ... )
    >>> table = TableDefinition(
    ...     table_id="table_0",
    ...     page_numbers=[1],
    ...     columns=[column],
    ... )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Any
import uuid

from tables.utils.geometry import BoundingBox


# =============================================================================
# Enumerations
# =============================================================================

class StructureType(str, Enum):
    """
    Visual structure type of a table.

    Determines how the table boundaries are rendered visually,
    which affects the extraction strategy.

    Values:
        BORDERED: Full grid with horizontal and vertical lines
        SEMI_BORDERED: Partial lines (often just horizontal row separators)
        UNBORDERED: No lines, relies on whitespace alignment
        SHADED: Uses background shading to differentiate rows/sections
        UNKNOWN: Structure type could not be determined
    """

    BORDERED = "bordered"
    SEMI_BORDERED = "semi_bordered"
    UNBORDERED = "unbordered"
    SHADED = "shaded"
    UNKNOWN = "unknown"


class ContentType(str, Enum):
    """
    Semantic content type of a table.

    Indicates what kind of data the table contains, which affects
    how rows should be interpreted and validated.

    Values:
        TRANSACTION: Transaction table with date, description, amounts
        SUMMARY: Summary table (totals, period statistics)
        ACCOUNT_INFO: Account information (account number, holder name)
        INTEREST_CHARGES: Interest and charges table
        OTHER: Unrecognized table type
    """

    TRANSACTION = "transaction"
    SUMMARY = "summary"
    ACCOUNT_INFO = "account_info"
    INTEREST_CHARGES = "interest_charges"
    OTHER = "other"


class DataType(str, Enum):
    """
    Data type of column values.

    Used to guide parsing and validation in Module 3.

    Values:
        DATE: Date values (various formats)
        NUMERIC: Numeric values (amounts, balances)
        TEXT: Free-form text
        MIXED: Column contains multiple data types
        UNKNOWN: Data type not determined
    """

    DATE = "date"
    NUMERIC = "numeric"
    TEXT = "text"
    MIXED = "mixed"
    UNKNOWN = "unknown"


# =============================================================================
# Column Definition
# =============================================================================

@dataclass
class ColumnDefinition:
    """
    Definition of a single table column.

    Captures the position, header text, and semantic meaning of a column.
    Used to guide cell extraction in Module 3.

    Attributes:
        column_id: Unique identifier for this column within the table
        x0: Left edge x-coordinate
        x1: Right edge x-coordinate
        header_text: Original header text (may be empty)
        semantic_type: Detected semantic type (date, debit, credit, etc.)
        data_type: Detected data type of column values
        confidence: Confidence score for semantic type detection (0.0-1.0)
        source_header_row: Row index where header was found (0-indexed)

    Example:
        >>> col = ColumnDefinition(
        ...     column_id=0,
        ...     x0=50.0,
        ...     x1=150.0,
        ...     header_text="Transaction Date",
        ...     semantic_type="date",
        ...     data_type=DataType.DATE,
        ...     confidence=0.95,
        ... )
    """

    column_id: int
    x0: float
    x1: float
    header_text: str = ""
    semantic_type: Optional[str] = None
    data_type: DataType = DataType.UNKNOWN
    confidence: float = 0.0
    source_header_row: Optional[int] = None

    @property
    def width(self) -> float:
        """Get the width of the column."""
        return self.x1 - self.x0

    @property
    def center_x(self) -> float:
        """Get the horizontal center of the column."""
        return (self.x0 + self.x1) / 2

    def contains_x(self, x: float, tolerance: float = 2.0) -> bool:
        """
        Check if an x-coordinate falls within this column.

        Args:
            x: X-coordinate to check
            tolerance: Extend column boundaries by this amount

        Returns:
            True if x is within the column bounds
        """
        return self.x0 - tolerance <= x <= self.x1 + tolerance

    def overlaps_x_range(self, x0: float, x1: float) -> bool:
        """
        Check if this column overlaps with an x-range.

        Args:
            x0: Left edge of range
            x1: Right edge of range

        Returns:
            True if column overlaps the range
        """
        return self.x0 < x1 and x0 < self.x1

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize to dictionary.

        Returns:
            Dictionary representation suitable for JSON serialization
        """
        return {
            "column_id": self.column_id,
            "x0": self.x0,
            "x1": self.x1,
            "header_text": self.header_text,
            "semantic_type": self.semantic_type,
            "data_type": self.data_type.value if isinstance(self.data_type, DataType) else self.data_type,
            "confidence": self.confidence,
            "source_header_row": self.source_header_row,
        }


# =============================================================================
# Table Definition
# =============================================================================

@dataclass
class TableDefinition:
    """
    Complete definition of a detected table.

    Captures all structural information about a table including:
    - Location (bounding boxes per page)
    - Column definitions
    - Structure type and content type
    - Cross-page spanning information
    - Detection metadata

    This is the primary output of Module 2 (Table Detection) and
    the primary input to Module 3 (Structure Recognition).

    Attributes:
        table_id: Unique identifier for this table
        page_numbers: List of page numbers where this table appears
        columns: List of column definitions
        structure_type: Visual structure type (bordered, etc.)
        content_type: Semantic content type (transaction, etc.)
        bounds_per_page: Bounding box for each page
        header_row_indices: Row indices identified as headers
        header_repeats_on_pages: Whether headers repeat on each page
        is_multi_page: Whether table spans multiple pages
        continuation_confidence: Confidence that pages are connected (0.0-1.0)
        detection_confidence: Overall detection confidence (0.0-1.0)
        warnings: List of warnings encountered during detection

    Example:
        >>> table = TableDefinition(
        ...     table_id="table_0",
        ...     page_numbers=[1, 2],
        ...     structure_type=StructureType.BORDERED,
        ...     content_type=ContentType.TRANSACTION,
        ...     is_multi_page=True,
        ... )
    """

    # Identity
    table_id: str = field(default_factory=lambda: f"table_{uuid.uuid4().hex[:8]}")
    page_numbers: list[int] = field(default_factory=list)

    # Column structure
    columns: list[ColumnDefinition] = field(default_factory=list)

    # Classification
    structure_type: StructureType = StructureType.UNKNOWN
    content_type: ContentType = ContentType.OTHER

    # Spatial information
    bounds_per_page: dict[int, BoundingBox] = field(default_factory=dict)

    # Header information
    header_row_indices: list[int] = field(default_factory=list)
    header_repeats_on_pages: bool = False

    # Cross-page information
    is_multi_page: bool = False
    continuation_confidence: float = 0.0

    # Detection metadata
    detection_confidence: float = 0.0
    warnings: list[str] = field(default_factory=list)

    @property
    def column_count(self) -> int:
        """Get the number of columns in the table."""
        return len(self.columns)

    @property
    def first_page(self) -> Optional[int]:
        """Get the first page number where this table appears."""
        return self.page_numbers[0] if self.page_numbers else None

    @property
    def last_page(self) -> Optional[int]:
        """Get the last page number where this table appears."""
        return self.page_numbers[-1] if self.page_numbers else None

    @property
    def page_count(self) -> int:
        """Get the number of pages this table spans."""
        return len(self.page_numbers)

    def get_bounds(self, page_number: int) -> Optional[BoundingBox]:
        """
        Get the bounding box for a specific page.

        Args:
            page_number: Page number to get bounds for

        Returns:
            BoundingBox for the page, or None if page not in table
        """
        return self.bounds_per_page.get(page_number)

    def get_column_by_semantic_type(self, semantic_type: str) -> Optional[ColumnDefinition]:
        """
        Find a column by its semantic type.

        Args:
            semantic_type: Semantic type to search for (e.g., "date", "debit")

        Returns:
            First matching ColumnDefinition, or None if not found
        """
        for column in self.columns:
            if column.semantic_type == semantic_type:
                return column
        return None

    def get_columns_by_semantic_type(self, semantic_type: str) -> list[ColumnDefinition]:
        """
        Find all columns with a specific semantic type.

        Args:
            semantic_type: Semantic type to search for

        Returns:
            List of matching ColumnDefinitions
        """
        return [col for col in self.columns if col.semantic_type == semantic_type]

    def find_column_for_x(self, x: float, tolerance: float = 2.0) -> Optional[ColumnDefinition]:
        """
        Find which column contains a given x-coordinate.

        Args:
            x: X-coordinate to look up
            tolerance: Extend column boundaries by this amount

        Returns:
            ColumnDefinition containing x, or None if not found
        """
        for column in self.columns:
            if column.contains_x(x, tolerance):
                return column
        return None

    def has_required_transaction_columns(self) -> bool:
        """
        Check if table has minimum required columns for transaction extraction.

        A transaction table should have at least:
        - A date column
        - A description column
        - At least one amount column (debit, credit, or generic amount)

        Returns:
            True if minimum required columns are present
        """
        semantic_types = {col.semantic_type for col in self.columns if col.semantic_type}

        has_date = "date" in semantic_types
        has_description = "description" in semantic_types
        has_amount = bool(semantic_types & {"debit", "credit", "amount", "balance"})

        return has_date and has_description and has_amount

    def add_warning(self, warning: str) -> None:
        """
        Add a warning message.

        Args:
            warning: Warning message to add
        """
        if warning not in self.warnings:
            self.warnings.append(warning)

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize to dictionary.

        Returns:
            Dictionary representation suitable for JSON serialization
        """
        return {
            "table_id": self.table_id,
            "page_numbers": self.page_numbers,
            "columns": [col.to_dict() for col in self.columns],
            "structure_type": self.structure_type.value,
            "content_type": self.content_type.value,
            "bounds_per_page": {
                page: bounds.to_dict()
                for page, bounds in self.bounds_per_page.items()
            },
            "header_row_indices": self.header_row_indices,
            "header_repeats_on_pages": self.header_repeats_on_pages,
            "is_multi_page": self.is_multi_page,
            "continuation_confidence": self.continuation_confidence,
            "detection_confidence": self.detection_confidence,
            "warnings": self.warnings,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TableDefinition:
        """
        Create a TableDefinition from a dictionary.

        Args:
            data: Dictionary representation

        Returns:
            TableDefinition instance
        """
        columns = [
            ColumnDefinition(
                column_id=col["column_id"],
                x0=col["x0"],
                x1=col["x1"],
                header_text=col.get("header_text", ""),
                semantic_type=col.get("semantic_type"),
                data_type=DataType(col.get("data_type", "unknown")),
                confidence=col.get("confidence", 0.0),
                source_header_row=col.get("source_header_row"),
            )
            for col in data.get("columns", [])
        ]

        bounds_per_page = {
            int(page): BoundingBox.from_dict(bounds)
            for page, bounds in data.get("bounds_per_page", {}).items()
        }

        return cls(
            table_id=data.get("table_id", f"table_{uuid.uuid4().hex[:8]}"),
            page_numbers=data.get("page_numbers", []),
            columns=columns,
            structure_type=StructureType(data.get("structure_type", "unknown")),
            content_type=ContentType(data.get("content_type", "other")),
            bounds_per_page=bounds_per_page,
            header_row_indices=data.get("header_row_indices", []),
            header_repeats_on_pages=data.get("header_repeats_on_pages", False),
            is_multi_page=data.get("is_multi_page", False),
            continuation_confidence=data.get("continuation_confidence", 0.0),
            detection_confidence=data.get("detection_confidence", 0.0),
            warnings=data.get("warnings", []),
        )


# =============================================================================
# Detection Result Container
# =============================================================================

@dataclass
class TableDetectionResult:
    """
    Container for table detection results from a document.

    Holds all detected tables along with metadata about the detection process.

    Attributes:
        tables: List of detected TableDefinition objects
        pages_processed: List of page numbers that were processed
        pages_with_tables: List of page numbers containing tables
        pages_skipped: List of (page_number, reason) for skipped pages
        processing_time_ms: Time taken for detection in milliseconds
        warnings: Global warnings (not table-specific)

    Example:
        >>> result = TableDetectionResult(
        ...     tables=[table1, table2],
        ...     pages_processed=[1, 2, 3],
        ...     pages_with_tables=[1, 2],
        ... )
        >>> result.table_count
        2
    """

    tables: list[TableDefinition] = field(default_factory=list)
    pages_processed: list[int] = field(default_factory=list)
    pages_with_tables: list[int] = field(default_factory=list)
    pages_skipped: list[tuple[int, str]] = field(default_factory=list)
    processing_time_ms: int = 0
    warnings: list[str] = field(default_factory=list)

    @property
    def table_count(self) -> int:
        """Get the number of detected tables."""
        return len(self.tables)

    @property
    def has_tables(self) -> bool:
        """Check if any tables were detected."""
        return len(self.tables) > 0

    @property
    def transaction_tables(self) -> list[TableDefinition]:
        """Get only tables classified as transaction tables."""
        return [t for t in self.tables if t.content_type == ContentType.TRANSACTION]

    def get_tables_on_page(self, page_number: int) -> list[TableDefinition]:
        """
        Get all tables present on a specific page.

        Args:
            page_number: Page number to query

        Returns:
            List of tables on that page
        """
        return [t for t in self.tables if page_number in t.page_numbers]

    def add_warning(self, warning: str) -> None:
        """
        Add a global warning message.

        Args:
            warning: Warning message to add
        """
        if warning not in self.warnings:
            self.warnings.append(warning)

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize to dictionary.

        Returns:
            Dictionary representation suitable for JSON serialization
        """
        return {
            "tables": [t.to_dict() for t in self.tables],
            "pages_processed": self.pages_processed,
            "pages_with_tables": self.pages_with_tables,
            "pages_skipped": self.pages_skipped,
            "processing_time_ms": self.processing_time_ms,
            "warnings": self.warnings,
            "summary": {
                "table_count": self.table_count,
                "transaction_table_count": len(self.transaction_tables),
                "pages_with_tables_count": len(self.pages_with_tables),
            },
        }
