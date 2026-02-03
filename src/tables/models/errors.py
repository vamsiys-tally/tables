"""Error codes, exceptions, and warning definitions."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class ErrorCode(str, Enum):
    """Enumeration of all error codes."""

    # File errors
    FILE_NOT_FOUND = "FILE_NOT_FOUND"
    FILE_EMPTY = "FILE_EMPTY"
    FILE_CORRUPTED = "FILE_CORRUPTED"
    FILE_PASSWORD_REQUIRED = "FILE_PASSWORD_REQUIRED"
    FILE_PASSWORD_INCORRECT = "FILE_PASSWORD_INCORRECT"
    FILE_INVALID_TYPE = "FILE_INVALID_TYPE"
    FILE_TOO_LARGE = "FILE_TOO_LARGE"

    # Content errors
    FILE_SCANNED_NOT_SUPPORTED = "FILE_SCANNED_NOT_SUPPORTED"
    LANGUAGE_NOT_SUPPORTED = "LANGUAGE_NOT_SUPPORTED"
    NOT_BANK_STATEMENT = "NOT_BANK_STATEMENT"
    ACCOUNT_TYPE_NOT_SUPPORTED = "ACCOUNT_TYPE_NOT_SUPPORTED"

    # Table errors
    NO_TABLES_FOUND = "NO_TABLES_FOUND"
    TABLE_STRUCTURE_UNCLEAR = "TABLE_STRUCTURE_UNCLEAR"
    HEADER_NOT_DETECTED = "HEADER_NOT_DETECTED"

    # Parsing errors
    DATE_PARSE_FAILED = "DATE_PARSE_FAILED"
    AMOUNT_PARSE_FAILED = "AMOUNT_PARSE_FAILED"
    ROW_STRUCTURE_INVALID = "ROW_STRUCTURE_INVALID"


# Human-readable error message templates
ERROR_MESSAGES = {
    # File errors
    ErrorCode.FILE_NOT_FOUND: "The specified file does not exist: {path}",
    ErrorCode.FILE_EMPTY: "The file is empty (0 bytes): {path}",
    ErrorCode.FILE_CORRUPTED: "The PDF file is corrupted and cannot be read: {path}",
    ErrorCode.FILE_PASSWORD_REQUIRED: "The PDF is password protected. Please provide the password.",
    ErrorCode.FILE_PASSWORD_INCORRECT: "The provided password is incorrect for this PDF.",
    ErrorCode.FILE_INVALID_TYPE: "Expected PDF file, got: {actual_type}",
    ErrorCode.FILE_TOO_LARGE: "File exceeds maximum size of {max_size}MB: {actual_size}MB",
    # Content errors
    ErrorCode.FILE_SCANNED_NOT_SUPPORTED: "This appears to be a scanned document. Only digitally generated PDFs are supported.",
    ErrorCode.LANGUAGE_NOT_SUPPORTED: "Document language '{detected_language}' is not supported. Only English documents are supported.",
    ErrorCode.NOT_BANK_STATEMENT: "This document does not appear to be a bank statement.",
    ErrorCode.ACCOUNT_TYPE_NOT_SUPPORTED: "Account type '{account_type}' is not currently supported. Supported types: Savings, Current, OD.",
    # Table errors
    ErrorCode.NO_TABLES_FOUND: "No transaction tables found in the document.",
    ErrorCode.TABLE_STRUCTURE_UNCLEAR: "Could not determine table structure on page {page}. The table format may not be supported.",
    ErrorCode.HEADER_NOT_DETECTED: "Could not identify table headers on page {page}.",
    # Parsing errors
    ErrorCode.DATE_PARSE_FAILED: "Could not parse date '{value}' in row {row}, column '{column}'.",
    ErrorCode.AMOUNT_PARSE_FAILED: "Could not parse amount '{value}' in row {row}, column '{column}'.",
    ErrorCode.ROW_STRUCTURE_INVALID: "Row {row} on page {page} has unexpected structure.",
}


class ProcessingError(Exception):
    """Exception raised during PDF processing with structured error information."""

    def __init__(
        self,
        code: ErrorCode,
        details: Optional[dict[str, Any]] = None,
        message: Optional[str] = None,
    ):
        """
        Initialize a ProcessingError.

        Args:
            code: The error code from ErrorCode enum
            details: Dictionary of details to format into the error message
            message: Optional custom message (overrides template)
        """
        self.code = code
        self.details = details or {}

        # Format the message from template if not provided
        if message is None:
            template = ERROR_MESSAGES.get(code, str(code))
            try:
                self.message = template.format(**self.details)
            except KeyError:
                self.message = template
        else:
            self.message = message

        super().__init__(self.message)

    def to_dict(self) -> dict[str, Any]:
        """Serialize error to dictionary for JSON output."""
        return {
            "code": self.code.value,
            "message": self.message,
            "details": self.details,
        }

    def __repr__(self) -> str:
        return f"ProcessingError(code={self.code.value!r}, message={self.message!r})"


class WarningCode(str, Enum):
    """Enumeration of warning codes."""

    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    PAGE_SKIPPED = "PAGE_SKIPPED"
    PARTIAL_EXTRACTION = "PARTIAL_EXTRACTION"
    DATE_FORMAT_AMBIGUOUS = "DATE_FORMAT_AMBIGUOUS"
    AMOUNT_FORMAT_UNUSUAL = "AMOUNT_FORMAT_UNUSUAL"
    ROW_MERGED = "ROW_MERGED"
    COLUMN_ALIGNMENT_DRIFT = "COLUMN_ALIGNMENT_DRIFT"
    LARGE_FILE = "LARGE_FILE"
    MIXED_CONTENT = "MIXED_CONTENT"


WARNING_MESSAGES = {
    WarningCode.LOW_CONFIDENCE: "Extraction confidence is below threshold: {confidence:.2f}",
    WarningCode.PAGE_SKIPPED: "Page {page} was skipped: {reason}",
    WarningCode.PARTIAL_EXTRACTION: "Only partial data could be extracted from page {page}",
    WarningCode.DATE_FORMAT_AMBIGUOUS: "Date '{value}' has ambiguous format, interpreted as {interpreted}",
    WarningCode.AMOUNT_FORMAT_UNUSUAL: "Amount '{value}' has unusual format",
    WarningCode.ROW_MERGED: "Rows {rows} were merged as a single transaction",
    WarningCode.COLUMN_ALIGNMENT_DRIFT: "Column alignment differs between pages {pages}",
    WarningCode.LARGE_FILE: "File has {page_count} pages, processing may be slow",
    WarningCode.MIXED_CONTENT: "Document contains mixed scanned/generated pages",
}


@dataclass
class ProcessingWarning:
    """A warning generated during processing that doesn't stop execution."""

    code: WarningCode
    message: str
    details: dict[str, Any] = field(default_factory=dict)
    page: Optional[int] = None
    row: Optional[int] = None

    @classmethod
    def create(
        cls,
        code: WarningCode,
        details: Optional[dict[str, Any]] = None,
        page: Optional[int] = None,
        row: Optional[int] = None,
    ) -> "ProcessingWarning":
        """Factory method to create a warning with formatted message."""
        details = details or {}
        template = WARNING_MESSAGES.get(code, str(code))
        try:
            message = template.format(**details)
        except KeyError:
            message = template

        return cls(code=code, message=message, details=details, page=page, row=row)

    def to_dict(self) -> dict[str, Any]:
        """Serialize warning to dictionary for JSON output."""
        result = {
            "code": self.code.value,
            "message": self.message,
        }
        if self.details:
            result["details"] = self.details
        if self.page is not None:
            result["page"] = self.page
        if self.row is not None:
            result["row"] = self.row
        return result
