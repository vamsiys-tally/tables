"""
Ground Truth Loader.

Loads ground truth files (CSV or JSON) and converts them to comparable transaction objects.
Supports flexible field names - auto-detects field types from value patterns.

Supported Formats:
    - JSON: Array of transaction objects (recommended)
    - CSV: With header row

Example:
    >>> gt = load_ground_truth("tests/data/ground_truth/yes_bank/yes_001.json")
    >>> print(f"Loaded {gt.transaction_count} transactions")
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Optional, Any
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class GroundTruthTransaction:
    """
    A single transaction from ground truth CSV.

    Attributes:
        row_number: 1-indexed row number in CSV (for error reporting)
        date: Transaction date
        value_date: Value date (optional)
        description: Transaction description
        reference: Reference number (optional)
        debit: Debit amount (optional)
        credit: Credit amount (optional)
        balance: Balance after transaction (optional)
        raw_data: Original CSV row data
    """

    row_number: int
    date: Optional[date] = None
    value_date: Optional[date] = None
    description: str = ""
    reference: Optional[str] = None
    debit: Optional[Decimal] = None
    credit: Optional[Decimal] = None
    balance: Optional[Decimal] = None
    raw_data: dict[str, str] = field(default_factory=dict)

    @property
    def amount(self) -> Optional[Decimal]:
        """Get the transaction amount (debit as negative, credit as positive)."""
        if self.debit:
            return -self.debit
        elif self.credit:
            return self.credit
        return None

    @property
    def has_amount(self) -> bool:
        """Check if transaction has either debit or credit."""
        return self.debit is not None or self.credit is not None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "row_number": self.row_number,
            "date": self.date.isoformat() if self.date else None,
            "value_date": self.value_date.isoformat() if self.value_date else None,
            "description": self.description,
            "reference": self.reference,
            "debit": str(self.debit) if self.debit else None,
            "credit": str(self.credit) if self.credit else None,
            "balance": str(self.balance) if self.balance else None,
        }


@dataclass
class GroundTruthFile:
    """
    Complete ground truth file with metadata.

    Attributes:
        file_path: Path to the CSV file
        pdf_path: Path to corresponding PDF file
        transactions: List of transactions
        transaction_count: Number of transactions
        parse_errors: Any errors encountered during parsing
    """

    file_path: Path
    pdf_path: Optional[Path] = None
    transactions: list[GroundTruthTransaction] = field(default_factory=list)
    parse_errors: list[str] = field(default_factory=list)

    @property
    def transaction_count(self) -> int:
        """Get number of transactions."""
        return len(self.transactions)

    @property
    def has_errors(self) -> bool:
        """Check if there were parsing errors."""
        return len(self.parse_errors) > 0


# =============================================================================
# Parsing Functions
# =============================================================================

def parse_date(value: str) -> Optional[date]:
    """
    Parse a date string in various formats.

    Supported formats:
        - DD/MM/YYYY
        - DD-MM-YYYY
        - YYYY-MM-DD
        - DD-Mon-YYYY (e.g., 15-Jan-2024)

    Args:
        value: Date string to parse

    Returns:
        Parsed date or None if parsing fails
    """
    if not value or not value.strip():
        return None

    value = value.strip()

    # Try different formats
    from datetime import datetime

    formats = [
        "%d/%m/%Y",      # DD/MM/YYYY
        "%d-%m-%Y",      # DD-MM-YYYY
        "%Y-%m-%d",      # YYYY-MM-DD
        "%d-%b-%Y",      # DD-Mon-YYYY
        "%d %b %Y",      # DD Mon YYYY
        "%d/%m/%y",      # DD/MM/YY
        "%d-%m-%y",      # DD-MM-YY
    ]

    for fmt in formats:
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            continue

    return None


def parse_amount(value: str) -> Optional[Decimal]:
    """
    Parse an amount string, handling Indian number format.

    Handles:
        - Standard: 1000.00, 1,000.00
        - Indian: 1,00,000.00
        - With currency: ₹1000, Rs. 1000
        - Negative: (1000), -1000

    Args:
        value: Amount string to parse

    Returns:
        Parsed Decimal or None if parsing fails
    """
    if not value or not value.strip():
        return None

    value = value.strip()

    # Check for empty placeholders
    if value in ("-", "--", "nil", "NIL", ""):
        return None

    # Track if negative
    is_negative = False
    if value.startswith("(") and value.endswith(")"):
        is_negative = True
        value = value[1:-1]
    elif value.startswith("-"):
        is_negative = True
        value = value[1:]

    # Remove currency symbols
    currency_symbols = ["₹", "Rs.", "Rs", "INR", "USD", "$", "€", "£"]
    for symbol in currency_symbols:
        value = value.replace(symbol, "")

    # Remove whitespace
    value = value.strip()

    # Remove commas (handles both 1,000.00 and 1,00,000.00)
    value = value.replace(",", "")

    # Remove any remaining whitespace within
    value = value.replace(" ", "")

    if not value:
        return None

    try:
        result = Decimal(value)
        if is_negative:
            result = -result
        return result
    except InvalidOperation:
        return None


def normalize_text(value: str) -> str:
    """
    Normalize text for comparison.

    - Lowercase
    - Collapse multiple whitespace to single space
    - Trim leading/trailing whitespace

    Args:
        value: Text to normalize

    Returns:
        Normalized text
    """
    if not value:
        return ""

    # Lowercase
    value = value.lower()

    # Collapse whitespace
    import re
    value = re.sub(r'\s+', ' ', value)

    # Trim
    value = value.strip()

    return value


# =============================================================================
# Loader Functions
# =============================================================================

def load_ground_truth(file_path: str | Path) -> GroundTruthFile:
    """
    Load ground truth transactions from a JSON or CSV file.

    Args:
        file_path: Path to the ground truth file (.json or .csv)

    Returns:
        GroundTruthFile with parsed transactions

    Example:
        >>> gt = load_ground_truth("tests/data/ground_truth/yes_bank/yes_001.json")
        >>> print(f"Loaded {gt.transaction_count} transactions")
    """
    file_path = Path(file_path)

    result = GroundTruthFile(file_path=file_path)

    # Find corresponding PDF
    pdf_path = file_path.with_suffix(".pdf")
    if pdf_path.exists():
        result.pdf_path = pdf_path

    if not file_path.exists():
        result.parse_errors.append(f"File not found: {file_path}")
        return result

    # Dispatch based on file extension
    suffix = file_path.suffix.lower()
    if suffix == ".json":
        _load_json_file(file_path, result)
    elif suffix == ".csv":
        _load_csv_file(file_path, result)
    else:
        result.parse_errors.append(f"Unsupported file format: {suffix}")

    logger.debug(f"Loaded {result.transaction_count} transactions from {file_path}")

    return result


def _load_json_file(file_path: Path, result: GroundTruthFile) -> None:
    """Load transactions from a JSON file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Expect array of transaction objects
        if not isinstance(data, list):
            result.parse_errors.append("JSON must be an array of transactions")
            return

        for row_num, row in enumerate(data, start=1):
            if not isinstance(row, dict):
                result.parse_errors.append(f"Row {row_num}: Expected object, got {type(row).__name__}")
                continue

            try:
                txn = _parse_row(row, row_num)
                result.transactions.append(txn)
            except Exception as e:
                result.parse_errors.append(f"Row {row_num}: {e}")

    except json.JSONDecodeError as e:
        result.parse_errors.append(f"Invalid JSON: {e}")
    except Exception as e:
        result.parse_errors.append(f"Failed to read JSON: {e}")


def _load_csv_file(file_path: Path, result: GroundTruthFile) -> None:
    """Load transactions from a CSV file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)

            for row_num, row in enumerate(reader, start=2):  # Start at 2 (1 is header)
                try:
                    txn = _parse_row(row, row_num)
                    result.transactions.append(txn)
                except Exception as e:
                    result.parse_errors.append(f"Row {row_num}: {e}")

    except Exception as e:
        result.parse_errors.append(f"Failed to read CSV: {e}")


def _parse_row(row: dict[str, Any], row_number: int) -> GroundTruthTransaction:
    """
    Parse a row into a GroundTruthTransaction.

    Auto-detects field types from values when field names don't match standard names.
    Stores original data in raw_data for direct comparison.

    Args:
        row: Row as dictionary (from JSON or CSV)
        row_number: Row number for error reporting

    Returns:
        Parsed GroundTruthTransaction
    """
    # Convert all values to strings for raw_data
    raw_data = {k: str(v) if v is not None else "" for k, v in row.items()}

    # Try to identify fields by name first, then by value patterns
    detected = _detect_fields(row)

    return GroundTruthTransaction(
        row_number=row_number,
        date=detected.get("date"),
        value_date=detected.get("value_date"),
        description=detected.get("description", ""),
        reference=detected.get("reference"),
        debit=detected.get("debit"),
        credit=detected.get("credit"),
        balance=detected.get("balance"),
        raw_data=raw_data,
    )


def _detect_fields(row: dict[str, Any]) -> dict[str, Any]:
    """
    Auto-detect field types from a row.

    Uses field name patterns and value patterns to identify:
    - Date fields (transaction date, value date)
    - Amount fields (debit, credit, balance)
    - Description field

    Args:
        row: Row dictionary

    Returns:
        Dictionary with detected field values
    """
    result: dict[str, Any] = {}

    # Normalize keys for matching
    key_map = {k.lower().strip().replace("_", " "): k for k in row.keys()}

    # Field name patterns (priority order)
    date_patterns = ["transaction date", "date", "txn date", "trans date", "posting date"]
    value_date_patterns = ["value date", "valuedate", "val date"]
    description_patterns = ["description", "particulars", "narration", "remarks", "details"]
    reference_patterns = [
        "cheque no/reference no", "reference no", "reference", "ref no", "ref",
        "cheque no", "chq no", "transaction id", "txn id", "utr"
    ]
    debit_patterns = ["withdrawals", "withdrawal", "debit", "dr", "debit amount"]
    credit_patterns = ["deposits", "deposit", "credit", "cr", "credit amount"]
    balance_patterns = ["running balance", "balance", "closing balance", "available balance"]

    def find_field(patterns: list[str]) -> Optional[str]:
        """Find the original key matching any pattern."""
        for pattern in patterns:
            if pattern in key_map:
                return key_map[pattern]
        return None

    def get_value(key: Optional[str]) -> Any:
        """Get value for a key, or None."""
        if key is None:
            return None
        return row.get(key)

    # Find fields by name patterns
    date_key = find_field(date_patterns)
    value_date_key = find_field(value_date_patterns)
    desc_key = find_field(description_patterns)
    ref_key = find_field(reference_patterns)
    debit_key = find_field(debit_patterns)
    credit_key = find_field(credit_patterns)
    balance_key = find_field(balance_patterns)

    # Parse found fields
    date_val = get_value(date_key)
    if date_val:
        result["date"] = parse_date(str(date_val))

    value_date_val = get_value(value_date_key)
    if value_date_val:
        result["value_date"] = parse_date(str(value_date_val))

    desc_val = get_value(desc_key)
    if desc_val:
        result["description"] = str(desc_val).strip()

    ref_val = get_value(ref_key)
    if ref_val and str(ref_val).strip():
        result["reference"] = str(ref_val).strip()

    debit_val = get_value(debit_key)
    if debit_val:
        result["debit"] = parse_amount(str(debit_val))

    credit_val = get_value(credit_key)
    if credit_val:
        result["credit"] = parse_amount(str(credit_val))

    balance_val = get_value(balance_key)
    if balance_val:
        result["balance"] = parse_amount(str(balance_val))

    return result


def discover_ground_truth_files(
    base_dir: str | Path,
    bank: Optional[str] = None,
) -> list[GroundTruthFile]:
    """
    Discover all ground truth files (JSON or CSV) in a directory.

    Args:
        base_dir: Base directory to search (e.g., tests/data/ground_truth)
        bank: Optional bank name to filter (e.g., "hdfc")

    Returns:
        List of GroundTruthFile objects (not yet loaded)

    Example:
        >>> files = discover_ground_truth_files("tests/data/ground_truth")
        >>> files = discover_ground_truth_files("tests/data/ground_truth", bank="hdfc")
    """
    base_dir = Path(base_dir)

    if not base_dir.exists():
        logger.warning(f"Ground truth directory not found: {base_dir}")
        return []

    # Build search pattern
    if bank:
        search_dir = base_dir / bank
        if not search_dir.exists():
            logger.warning(f"Bank directory not found: {search_dir}")
            return []
        gt_files = list(search_dir.glob("*.json")) + list(search_dir.glob("*.csv"))
    else:
        gt_files = list(base_dir.glob("**/*.json")) + list(base_dir.glob("**/*.csv"))

    # Create GroundTruthFile objects (lazy - not loaded yet)
    results = []
    for gt_path in sorted(gt_files):
        gt_file = GroundTruthFile(file_path=gt_path)
        pdf_path = gt_path.with_suffix(".pdf")
        if pdf_path.exists():
            gt_file.pdf_path = pdf_path
        results.append(gt_file)

    logger.info(f"Discovered {len(results)} ground truth files")

    return results


def load_all_ground_truth(
    base_dir: str | Path,
    bank: Optional[str] = None,
) -> list[GroundTruthFile]:
    """
    Discover and load all ground truth files.

    Args:
        base_dir: Base directory to search
        bank: Optional bank name to filter

    Returns:
        List of loaded GroundTruthFile objects
    """
    files = discover_ground_truth_files(base_dir, bank)

    loaded = []
    for gt_file in files:
        loaded_file = load_ground_truth(gt_file.file_path)
        loaded.append(loaded_file)

    return loaded
