"""Data models for document classification and processing results."""

from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from enum import Enum
from typing import TYPE_CHECKING, Any, Literal, Optional

from tables.models.errors import ProcessingError, ProcessingWarning

if TYPE_CHECKING:
    import pdfplumber


class DocumentType(str, Enum):
    """Classification of document types."""

    BANK_STATEMENT = "bank_statement"
    NOT_BANK_STATEMENT = "not_bank_statement"
    UNKNOWN = "unknown"


class AccountType(str, Enum):
    """Classification of bank account types."""

    SAVINGS = "savings"
    CURRENT = "current"
    OVERDRAFT = "overdraft"
    CREDIT_CARD = "credit_card"  # Not supported in v1
    LOAN = "loan"  # Not supported in v1
    FIXED_DEPOSIT = "fixed_deposit"  # Not supported in v1
    UNKNOWN = "unknown"

    @classmethod
    def supported_types(cls) -> list["AccountType"]:
        """Return list of supported account types in v1."""
        return [cls.SAVINGS, cls.CURRENT, cls.OVERDRAFT]


class TableStructureType(str, Enum):
    """Classification of table structure types."""

    BORDERED = "bordered"
    SEMI_BORDERED = "semi_bordered"
    UNBORDERED = "unbordered"
    SHADED = "shaded"


class TableContentType(str, Enum):
    """Classification of table content types."""

    TRANSACTION = "transaction"
    SUMMARY = "summary"
    ACCOUNT_INFO = "account_info"
    OTHER = "other"


@dataclass
class ProcessingOptions:
    """Options to customize processing behavior."""

    # File size limits
    max_file_size_mb: float = 100.0
    max_pages: int = 500

    # Classification thresholds
    scanned_text_density_threshold: float = 0.1  # chars per pixel
    language_confidence_threshold: float = 0.8
    bank_statement_confidence_threshold: float = 0.7

    # Processing behavior
    skip_scanned_pages: bool = True
    strict_mode: bool = False  # If True, fail on low confidence instead of warning

    # ML options
    use_ml_fallback: bool = True
    header_similarity_threshold: float = 0.8

    def to_dict(self) -> dict[str, Any]:
        """Serialize options to dictionary."""
        return {
            "max_file_size_mb": self.max_file_size_mb,
            "max_pages": self.max_pages,
            "scanned_text_density_threshold": self.scanned_text_density_threshold,
            "language_confidence_threshold": self.language_confidence_threshold,
            "bank_statement_confidence_threshold": self.bank_statement_confidence_threshold,
            "skip_scanned_pages": self.skip_scanned_pages,
            "strict_mode": self.strict_mode,
            "use_ml_fallback": self.use_ml_fallback,
            "header_similarity_threshold": self.header_similarity_threshold,
        }


@dataclass
class BoundingBox:
    """Represents a rectangular region on a page."""

    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0

    @property
    def area(self) -> float:
        return self.width * self.height

    def to_dict(self) -> dict[str, float]:
        return {"x0": self.x0, "y0": self.y0, "x1": self.x1, "y1": self.y1}

    def contains(self, other: "BoundingBox") -> bool:
        """Check if this box contains another box."""
        return (
            self.x0 <= other.x0
            and self.y0 <= other.y0
            and self.x1 >= other.x1
            and self.y1 >= other.y1
        )

    def overlaps(self, other: "BoundingBox") -> bool:
        """Check if this box overlaps with another box."""
        return not (
            self.x1 < other.x0
            or other.x1 < self.x0
            or self.y1 < other.y0
            or other.y1 < self.y0
        )


@dataclass
class FileClassification:
    """Result of file classification with metadata."""

    status: Literal["success", "error"]
    error_code: Optional[str] = None
    error_message: Optional[str] = None

    # File metadata
    file_path: str = ""
    file_type: str = ""
    file_size_bytes: int = 0
    is_password_protected: bool = False
    is_scanned: bool = False

    # Content metadata
    page_count: int = 0
    pages_with_tables: list[int] = field(default_factory=list)
    language: str = ""
    language_confidence: float = 0.0
    document_type: DocumentType = DocumentType.UNKNOWN
    account_type: Optional[AccountType] = None

    # Confidence & warnings
    classification_confidence: float = 0.0
    warnings: list[ProcessingWarning] = field(default_factory=list)

    # For downstream processing (not serialized)
    pdf_document: Optional["pdfplumber.PDF"] = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible dict (excludes pdf_document)."""
        result = {
            "status": self.status,
            "file_path": self.file_path,
            "file_type": self.file_type,
            "file_size_bytes": self.file_size_bytes,
            "is_password_protected": self.is_password_protected,
            "is_scanned": self.is_scanned,
            "page_count": self.page_count,
            "pages_with_tables": self.pages_with_tables,
            "language": self.language,
            "language_confidence": self.language_confidence,
            "document_type": self.document_type.value,
            "account_type": self.account_type.value if self.account_type else None,
            "classification_confidence": self.classification_confidence,
            "warnings": [w.to_dict() for w in self.warnings],
        }

        if self.error_code:
            result["error_code"] = self.error_code
        if self.error_message:
            result["error_message"] = self.error_message

        return result

    @classmethod
    def from_error(cls, error: ProcessingError, file_path: str = "") -> "FileClassification":
        """Create a FileClassification from an error."""
        return cls(
            status="error",
            error_code=error.code.value,
            error_message=error.message,
            file_path=file_path,
        )


@dataclass
class ColumnSchema:
    """Schema definition for a table column."""

    name: str
    semantic_type: str  # "date", "description", "debit", "credit", "balance", "reference"
    data_type: str  # "date", "decimal", "string"
    nullable: bool = True
    source_column_index: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "semantic_type": self.semantic_type,
            "data_type": self.data_type,
            "nullable": self.nullable,
            "source_column_index": self.source_column_index,
        }


@dataclass
class TransactionRow:
    """A single transaction row extracted from a table."""

    row_id: int
    source_page: int
    source_row_indices: list[int] = field(default_factory=list)

    # Standard fields (nullable)
    transaction_date: Optional[date] = None
    value_date: Optional[date] = None
    description: str = ""
    reference: Optional[str] = None
    debit_amount: Optional[Decimal] = None
    credit_amount: Optional[Decimal] = None
    balance: Optional[Decimal] = None

    # Per-row confidence and warnings
    confidence: float = 1.0
    warnings: list[str] = field(default_factory=list)

    # Raw values for debugging/audit
    raw_cells: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible dict."""
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


@dataclass
class TransactionTable:
    """A complete transaction table extracted from the document."""

    table_id: str
    source_pages: list[int] = field(default_factory=list)

    # Schema
    columns: list[ColumnSchema] = field(default_factory=list)

    # Data
    rows: list[TransactionRow] = field(default_factory=list)

    # Metadata
    row_count: int = 0
    date_range: Optional[tuple[date, date]] = None
    total_debit: Optional[Decimal] = None
    total_credit: Optional[Decimal] = None

    # Quality metrics
    confidence_score: float = 1.0
    parse_warnings: list[ProcessingWarning] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible dict."""
        return {
            "table_id": self.table_id,
            "source_pages": self.source_pages,
            "columns": [c.to_dict() for c in self.columns],
            "rows": [r.to_dict() for r in self.rows],
            "row_count": self.row_count,
            "date_range": (
                [self.date_range[0].isoformat(), self.date_range[1].isoformat()]
                if self.date_range
                else None
            ),
            "total_debit": str(self.total_debit) if self.total_debit is not None else None,
            "total_credit": str(self.total_credit) if self.total_credit is not None else None,
            "confidence_score": self.confidence_score,
            "parse_warnings": [w.to_dict() for w in self.parse_warnings],
        }


@dataclass
class ProcessingResult:
    """Final result of processing a bank statement PDF."""

    success: bool
    data: list[TransactionTable] = field(default_factory=list)
    errors: list[ProcessingError] = field(default_factory=list)
    warnings: list[ProcessingWarning] = field(default_factory=list)

    # Processing metadata
    processing_time_ms: int = 0
    pages_processed: int = 0
    pages_skipped: list[tuple[int, str]] = field(default_factory=list)

    # Overall confidence
    overall_confidence: float = 1.0

    # Source file info
    file_classification: Optional[FileClassification] = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible dict."""
        return {
            "success": self.success,
            "data": [t.to_dict() for t in self.data],
            "errors": [e.to_dict() for e in self.errors],
            "warnings": [w.to_dict() for w in self.warnings],
            "processing_time_ms": self.processing_time_ms,
            "pages_processed": self.pages_processed,
            "pages_skipped": [{"page": p, "reason": r} for p, r in self.pages_skipped],
            "overall_confidence": self.overall_confidence,
            "file_classification": (
                self.file_classification.to_dict() if self.file_classification else None
            ),
        }
