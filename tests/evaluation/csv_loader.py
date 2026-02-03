"""
Ground Truth CSV Loader.

Loads ground truth CSV files and converts them to comparable transaction objects.

CSV Format:
    date,value_date,description,reference,debit,credit,balance

Example:
    >>> transactions = load_ground_truth("tests/data/ground_truth/hdfc/hdfc_001.csv")
    >>> len(transactions)
    45
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from datetime import date
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

def load_ground_truth(csv_path: str | Path) -> GroundTruthFile:
    """
    Load ground truth transactions from a CSV file.

    Args:
        csv_path: Path to the CSV file

    Returns:
        GroundTruthFile with parsed transactions

    Example:
        >>> gt = load_ground_truth("tests/data/ground_truth/hdfc/hdfc_001.csv")
        >>> print(f"Loaded {gt.transaction_count} transactions")
    """
    csv_path = Path(csv_path)

    result = GroundTruthFile(file_path=csv_path)

    # Find corresponding PDF
    pdf_path = csv_path.with_suffix(".pdf")
    if pdf_path.exists():
        result.pdf_path = pdf_path

    if not csv_path.exists():
        result.parse_errors.append(f"File not found: {csv_path}")
        return result

    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)

            for row_num, row in enumerate(reader, start=2):  # Start at 2 (1 is header)
                try:
                    txn = _parse_csv_row(row, row_num)
                    result.transactions.append(txn)
                except Exception as e:
                    result.parse_errors.append(f"Row {row_num}: {e}")

    except Exception as e:
        result.parse_errors.append(f"Failed to read CSV: {e}")

    logger.debug(f"Loaded {result.transaction_count} transactions from {csv_path}")

    return result


def _parse_csv_row(row: dict[str, str], row_number: int) -> GroundTruthTransaction:
    """
    Parse a single CSV row into a GroundTruthTransaction.

    Expected columns (case-insensitive):
        date, value_date, description, reference, debit, credit, balance

    Args:
        row: CSV row as dictionary
        row_number: Row number for error reporting

    Returns:
        Parsed GroundTruthTransaction
    """
    # Normalize column names (lowercase, strip)
    normalized_row = {k.lower().strip(): v for k, v in row.items()}

    return GroundTruthTransaction(
        row_number=row_number,
        date=parse_date(normalized_row.get("date", "")),
        value_date=parse_date(normalized_row.get("value_date", "")),
        description=normalized_row.get("description", "").strip(),
        reference=normalized_row.get("reference", "").strip() or None,
        debit=parse_amount(normalized_row.get("debit", "")),
        credit=parse_amount(normalized_row.get("credit", "")),
        balance=parse_amount(normalized_row.get("balance", "")),
        raw_data=dict(row),
    )


def discover_ground_truth_files(
    base_dir: str | Path,
    bank: Optional[str] = None,
) -> list[GroundTruthFile]:
    """
    Discover all ground truth CSV files in a directory.

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
        csv_files = list(search_dir.glob("*.csv"))
    else:
        csv_files = list(base_dir.glob("**/*.csv"))

    # Create GroundTruthFile objects (lazy - not loaded yet)
    results = []
    for csv_path in sorted(csv_files):
        gt_file = GroundTruthFile(file_path=csv_path)
        pdf_path = csv_path.with_suffix(".pdf")
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
