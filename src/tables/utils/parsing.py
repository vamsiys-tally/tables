"""
Data Parsing Utilities.

This module provides robust parsing functions for extracting structured data
from bank statement cell values. Handles the variety of formats found in
Indian bank statements.

Key Features:
    - Date parsing: Multiple formats (DD/MM/YY, DD-MMM-YYYY, etc.)
    - Amount parsing: Indian number formats (lakhs), currency symbols, negatives
    - Text cleaning: Normalize whitespace, remove noise

Supported Number Formats:
    - Standard: 1,000.00
    - Indian (lakhs): 1,00,000.00
    - Negative with parentheses: (500.00)
    - With currency: ₹ 1,000.00, Rs. 500, INR 1000

Supported Date Formats:
    - DD/MM/YYYY, DD-MM-YYYY
    - DD/MM/YY, DD-MM-YY
    - DD-MMM-YYYY, DD MMM YYYY (e.g., 15-Jan-2024)
    - YYYY-MM-DD (ISO format)

Example Usage:
    >>> from tables.utils.parsing import parse_amount, parse_date
    >>> parse_amount("₹ 1,00,000.50")
    Decimal('100000.50')
    >>> parse_date("15/01/2024")
    datetime.date(2024, 1, 15)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal, InvalidOperation
from typing import Optional, Tuple
import re
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

# Currency symbols and prefixes to strip (longer ones first to avoid partial matches)
CURRENCY_SYMBOLS = ["₹", "rs.", "rs", "inr", "usd", "$", "€", "£"]

# Regex patterns for amount parsing
AMOUNT_PATTERNS = {
    # Indian format with lakhs: 1,00,000.00 or 1,00,00,000.00
    "indian_lakhs": re.compile(
        r"^[(\s]*"  # Optional opening paren and whitespace
        r"[-−]?\s*"  # Optional leading minus
        r"(\d{1,2}(?:,\d{2})*(?:,\d{3})?)"  # Indian grouping
        r"(?:\.(\d{1,2}))?"  # Optional decimal part
        r"[\s)]*$"  # Optional closing paren and whitespace
    ),
    # Standard format: 1,000,000.00
    "standard": re.compile(
        r"^[(\s]*"
        r"[-−]?\s*"
        r"(\d{1,3}(?:,\d{3})*)"  # Standard grouping
        r"(?:\.(\d{1,2}))?"
        r"[\s)]*$"
    ),
    # Simple number: 1000.00 or 1000
    "simple": re.compile(
        r"^[(\s]*"
        r"[-−]?\s*"
        r"(\d+)"
        r"(?:\.(\d{1,2}))?"
        r"[\s)]*$"
    ),
}

# Date format patterns (order matters - more specific first)
DATE_FORMATS = [
    # DD-MMM-YYYY or DD MMM YYYY (e.g., 15-Jan-2024, 15 Jan 2024)
    (r"(\d{1,2})[-/\s](jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*[-/\s](\d{4})", "%d-%b-%Y"),
    (r"(\d{1,2})[-/\s](jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*[-/\s](\d{2})", "%d-%b-%y"),
    # DD/MM/YYYY or DD-MM-YYYY
    (r"(\d{1,2})[/-](\d{1,2})[/-](\d{4})", "%d/%m/%Y"),
    # DD/MM/YY or DD-MM-YY
    (r"(\d{1,2})[/-](\d{1,2})[/-](\d{2})", "%d/%m/%y"),
    # YYYY-MM-DD (ISO format)
    (r"(\d{4})[/-](\d{1,2})[/-](\d{1,2})", "%Y-%m-%d"),
    # YYYY/MM/DD
    (r"(\d{4})[/-](\d{2})[/-](\d{2})", "%Y/%m/%d"),
]

# Month name mapping for parsing
MONTH_MAP = {
    "jan": 1, "january": 1,
    "feb": 2, "february": 2,
    "mar": 3, "march": 3,
    "apr": 4, "april": 4,
    "may": 5,
    "jun": 6, "june": 6,
    "jul": 7, "july": 7,
    "aug": 8, "august": 8,
    "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10,
    "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}


# =============================================================================
# Parse Result Types
# =============================================================================

@dataclass
class AmountParseResult:
    """
    Result of amount parsing.

    Attributes:
        value: Parsed Decimal value, or None if parsing failed
        success: Whether parsing succeeded
        is_negative: Whether the amount is negative
        original: Original input string
        error: Error message if parsing failed
    """

    value: Optional[Decimal]
    success: bool
    is_negative: bool = False
    original: str = ""
    error: Optional[str] = None


@dataclass
class DateParseResult:
    """
    Result of date parsing.

    Attributes:
        value: Parsed date value, or None if parsing failed
        success: Whether parsing succeeded
        format_used: The format pattern that matched
        original: Original input string
        error: Error message if parsing failed
    """

    value: Optional[date]
    success: bool
    format_used: Optional[str] = None
    original: str = ""
    error: Optional[str] = None


# =============================================================================
# Amount Parsing
# =============================================================================

def parse_amount(
    text: str,
    allow_negative: bool = True,
) -> Optional[Decimal]:
    """
    Parse an amount string to Decimal.

    Handles various formats including:
    - Indian format (lakhs): 1,00,000.00
    - Standard format: 1,000,000.00
    - Currency symbols: ₹, Rs, INR, $
    - Negative with parentheses: (500.00)
    - Negative with minus: -500.00

    Args:
        text: Amount string to parse
        allow_negative: Whether to allow negative values

    Returns:
        Decimal value, or None if parsing failed

    Examples:
        >>> parse_amount("1,00,000.50")
        Decimal('100000.50')
        >>> parse_amount("₹ 5,000.00")
        Decimal('5000.00')
        >>> parse_amount("(1,000.00)")
        Decimal('-1000.00')
        >>> parse_amount("invalid")
        None
    """
    result = parse_amount_detailed(text, allow_negative)
    return result.value


def parse_amount_detailed(
    text: str,
    allow_negative: bool = True,
) -> AmountParseResult:
    """
    Parse an amount string with detailed result.

    Args:
        text: Amount string to parse
        allow_negative: Whether to allow negative values

    Returns:
        AmountParseResult with parsing details
    """
    if not text or not text.strip():
        return AmountParseResult(
            value=None,
            success=False,
            original=text,
            error="Empty input",
        )

    original = text
    text = text.strip()

    # Check for negative indicators
    is_negative = False

    # Parentheses indicate negative: (500.00) → -500.00
    if text.startswith("(") and text.endswith(")"):
        is_negative = True
        text = text[1:-1].strip()

    # Leading minus sign
    if text.startswith("-") or text.startswith("−"):
        is_negative = True
        text = text[1:].strip()

    # Trailing minus (some formats use this)
    if text.endswith("-") or text.endswith("−"):
        is_negative = True
        text = text[:-1].strip()

    # "Dr" or "Cr" suffix (debit/credit indicators)
    text_lower = text.lower()
    if text_lower.endswith(" dr") or text_lower.endswith("dr"):
        is_negative = True
        text = text[:-2].strip()
    elif text_lower.endswith(" cr") or text_lower.endswith("cr"):
        # Credit is positive, already default
        text = text[:-2].strip()

    # Strip currency symbols
    text_lower = text.lower()
    for symbol in CURRENCY_SYMBOLS:
        if text_lower.startswith(symbol):
            text = text[len(symbol):].strip()
            text_lower = text.lower()

    # Remove any remaining whitespace within the number
    # (some formats have spaces: "1 000 000")
    text = text.replace(" ", "")

    # Try to parse the cleaned number
    try:
        # First, check if it's Indian format (has comma pattern like 1,00,000)
        if _is_indian_format(text):
            value = _parse_indian_format(text)
        else:
            # Standard parsing: remove commas
            text = text.replace(",", "")
            value = Decimal(text)

        # Apply negative sign
        if is_negative and allow_negative:
            value = -value
        elif is_negative and not allow_negative:
            value = abs(value)

        return AmountParseResult(
            value=value,
            success=True,
            is_negative=is_negative,
            original=original,
        )

    except (InvalidOperation, ValueError) as e:
        return AmountParseResult(
            value=None,
            success=False,
            is_negative=is_negative,
            original=original,
            error=f"Invalid number format: {str(e)}",
        )


def _is_indian_format(text: str) -> bool:
    """
    Check if a number string uses Indian number format (lakhs).

    Indian format groups digits as: 1,00,00,000 (crores)
    First group from right is 3 digits, then 2 digits each.

    Args:
        text: Number string with commas

    Returns:
        True if appears to be Indian format
    """
    if "," not in text:
        return False

    # Split by comma and check grouping pattern
    parts = text.replace(".", ",").split(",")

    # Remove decimal part if present
    if "." in text:
        parts = text.split(".")[0].split(",")

    if len(parts) < 2:
        return False

    # Indian format: rightmost group is 3 digits, others are 2
    # e.g., "1,00,000" → ["1", "00", "000"]
    if len(parts[-1]) == 3:
        # Check if middle groups are 2 digits
        for part in parts[1:-1]:
            if len(part) != 2:
                return False
        return True

    return False


def _parse_indian_format(text: str) -> Decimal:
    """
    Parse a number in Indian format (lakhs/crores).

    Args:
        text: Number string like "1,00,000.50"

    Returns:
        Decimal value
    """
    # Remove commas and parse
    text = text.replace(",", "")
    return Decimal(text)


def normalize_amount(value: Decimal, decimal_places: int = 2) -> Decimal:
    """
    Normalize an amount to standard decimal places.

    Args:
        value: Decimal value to normalize
        decimal_places: Number of decimal places

    Returns:
        Normalized Decimal
    """
    quantize_str = "0." + "0" * decimal_places
    return value.quantize(Decimal(quantize_str))


# =============================================================================
# Date Parsing
# =============================================================================

def parse_date(text: str) -> Optional[date]:
    """
    Parse a date string to a date object.

    Handles various formats common in Indian bank statements:
    - DD/MM/YYYY, DD-MM-YYYY
    - DD/MM/YY, DD-MM-YY
    - DD-MMM-YYYY (e.g., 15-Jan-2024)
    - YYYY-MM-DD (ISO format)

    Args:
        text: Date string to parse

    Returns:
        date object, or None if parsing failed

    Examples:
        >>> parse_date("15/01/2024")
        datetime.date(2024, 1, 15)
        >>> parse_date("15-Jan-2024")
        datetime.date(2024, 1, 15)
        >>> parse_date("2024-01-15")
        datetime.date(2024, 1, 15)
    """
    result = parse_date_detailed(text)
    return result.value


def parse_date_detailed(text: str) -> DateParseResult:
    """
    Parse a date string with detailed result.

    Args:
        text: Date string to parse

    Returns:
        DateParseResult with parsing details
    """
    if not text or not text.strip():
        return DateParseResult(
            value=None,
            success=False,
            original=text,
            error="Empty input",
        )

    original = text
    text = text.strip()

    # Try each format pattern
    text_lower = text.lower()

    # Try parsing with month names first (DD-MMM-YYYY)
    month_pattern = re.compile(
        r"(\d{1,2})[-/\s]([a-z]+)[-/\s](\d{2,4})",
        re.IGNORECASE
    )
    match = month_pattern.match(text_lower)
    if match:
        day_str, month_str, year_str = match.groups()
        month_key = month_str[:3].lower()
        if month_key in MONTH_MAP:
            try:
                day = int(day_str)
                month = MONTH_MAP[month_key]
                year = int(year_str)
                if year < 100:
                    # Two-digit year: assume 2000s for now
                    year = 2000 + year if year < 50 else 1900 + year

                parsed_date = date(year, month, day)
                return DateParseResult(
                    value=parsed_date,
                    success=True,
                    format_used="DD-MMM-YYYY",
                    original=original,
                )
            except ValueError as e:
                pass  # Continue to other formats

    # Try numeric formats
    # DD/MM/YYYY or DD-MM-YYYY
    numeric_pattern = re.compile(r"(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})")
    match = numeric_pattern.match(text)
    if match:
        part1, part2, part3 = match.groups()
        try:
            # Assume DD/MM/YYYY for Indian format
            day = int(part1)
            month = int(part2)
            year = int(part3)

            if year < 100:
                year = 2000 + year if year < 50 else 1900 + year

            # Validate and swap if needed (in case of MM/DD/YYYY)
            if month > 12 and day <= 12:
                # Likely MM/DD/YYYY format
                day, month = month, day

            parsed_date = date(year, month, day)
            return DateParseResult(
                value=parsed_date,
                success=True,
                format_used="DD/MM/YYYY",
                original=original,
            )
        except ValueError:
            pass

    # Try ISO format: YYYY-MM-DD
    iso_pattern = re.compile(r"(\d{4})[/-](\d{1,2})[/-](\d{1,2})")
    match = iso_pattern.match(text)
    if match:
        year_str, month_str, day_str = match.groups()
        try:
            parsed_date = date(int(year_str), int(month_str), int(day_str))
            return DateParseResult(
                value=parsed_date,
                success=True,
                format_used="YYYY-MM-DD",
                original=original,
            )
        except ValueError:
            pass

    return DateParseResult(
        value=None,
        success=False,
        original=original,
        error=f"Could not parse date: {text}",
    )


def format_date(d: date, format_str: str = "%d/%m/%Y") -> str:
    """
    Format a date object to string.

    Args:
        d: date object to format
        format_str: strftime format string

    Returns:
        Formatted date string
    """
    return d.strftime(format_str)


# =============================================================================
# Text Cleaning
# =============================================================================

def clean_text(text: str) -> str:
    """
    Clean and normalize text from a cell.

    Operations:
    - Strip leading/trailing whitespace
    - Normalize internal whitespace (multiple spaces → single)
    - Remove control characters

    Args:
        text: Raw text to clean

    Returns:
        Cleaned text
    """
    if not text:
        return ""

    # Remove control characters
    text = "".join(char for char in text if ord(char) >= 32 or char in "\t\n")

    # Normalize whitespace
    text = " ".join(text.split())

    return text.strip()


def clean_description(text: str) -> str:
    """
    Clean a transaction description.

    Additional cleaning for description fields:
    - Remove redundant punctuation
    - Normalize dashes

    Args:
        text: Raw description text

    Returns:
        Cleaned description
    """
    text = clean_text(text)

    # Normalize different dash characters
    text = text.replace("–", "-").replace("—", "-")

    # Remove multiple consecutive dashes
    text = re.sub(r"-{2,}", "-", text)

    # Remove trailing punctuation (except parentheses)
    text = text.rstrip(".,;:")

    return text


def is_empty_cell(text: str) -> bool:
    """
    Check if a cell value should be considered empty.

    Considers:
    - Null/None
    - Empty string
    - Whitespace only
    - Common placeholder values

    Args:
        text: Cell value to check

    Returns:
        True if cell should be treated as empty
    """
    if not text:
        return True

    cleaned = text.strip().lower()

    if not cleaned:
        return True

    # Common empty placeholders
    empty_values = {"", "-", "--", "---", "nil", "n/a", "na", "none", "."}
    return cleaned in empty_values


def extract_reference(text: str) -> Optional[str]:
    """
    Extract a reference number from text.

    Looks for patterns like:
    - UTR numbers
    - Cheque numbers
    - Transaction IDs

    Args:
        text: Text to extract reference from

    Returns:
        Extracted reference or None
    """
    if not text or is_empty_cell(text):
        return None

    text = clean_text(text)

    # Try to find UTR pattern
    utr_pattern = re.compile(r"\b([A-Z]{4}\d{13,})\b")
    match = utr_pattern.search(text.upper())
    if match:
        return match.group(1)

    # Try to find cheque number
    # Patterns like: "CHQ NO: 123456", "CHEQUE 123456", "CQ:123456"
    chq_pattern = re.compile(
        r"\b(CHQ|CHEQUE|CQ)[\s]*(?:NO\.?|NUMBER)?[\s:]*(\d{6,})\b",
        re.IGNORECASE
    )
    match = chq_pattern.search(text)
    if match:
        return f"CHQ{match.group(2)}"

    # If short alphanumeric, might be a reference itself
    if re.match(r"^[A-Z0-9]{6,20}$", text.upper()):
        return text.upper()

    return None
