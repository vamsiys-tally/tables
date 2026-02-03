"""
Header Keywords and Semantic Type Definitions.

This module defines comprehensive keyword mappings for identifying
table headers in bank statements. Keywords are organized by semantic
type (date, description, debit, credit, balance, reference) and include
variations commonly found in Indian bank statements.

The keyword matching system uses:
1. Exact match (highest confidence)
2. Contains match (medium confidence)
3. Fuzzy/SLM match (fallback for ambiguous cases)

Keyword Organization:
    - Primary keywords: Most common, unambiguous terms
    - Secondary keywords: Less common variations
    - Abbreviated forms: Common abbreviations (e.g., "Txn", "Chq")
    - Bank-specific terms: Terminology unique to certain banks

Example Usage:
    >>> from tables.detector.keywords import HEADER_KEYWORDS, normalize_header_text
    >>>
    >>> text = "Transaction Date"
    >>> normalized = normalize_header_text(text)
    >>> for semantic_type, keywords in HEADER_KEYWORDS.items():
    ...     if any(kw in normalized for kw in keywords):
    ...         print(f"Matched: {semantic_type}")
"""

import re
from typing import Optional


# =============================================================================
# Semantic Type Definitions
# =============================================================================

SEMANTIC_TYPES = {
    "date": "Date-related columns (transaction date, value date, posting date)",
    "description": "Transaction description, narration, or particulars",
    "reference": "Reference numbers, cheque numbers, UTR, transaction IDs",
    "debit": "Debit/withdrawal amounts",
    "credit": "Credit/deposit amounts",
    "balance": "Account balance (running, closing, available)",
    "amount": "Generic amount column (could be debit or credit)",
    "serial": "Serial number or row index",
}


# =============================================================================
# Header Keywords by Semantic Type
# =============================================================================

HEADER_KEYWORDS: dict[str, list[str]] = {
    # -------------------------------------------------------------------------
    # Date Columns
    # -------------------------------------------------------------------------
    "date": [
        # Primary terms
        "date",
        "transaction date",
        "txn date",
        "trans date",
        "posting date",
        "value date",
        "effective date",
        "entry date",
        # Abbreviated forms
        "txn dt",
        "trans dt",
        "val date",
        "val dt",
        "post date",
        "post dt",
        # Bank-specific variations
        "tran date",
        "trn date",
        "transaction dt",
        "dated",
    ],

    # -------------------------------------------------------------------------
    # Description/Narration Columns
    # -------------------------------------------------------------------------
    "description": [
        # Primary terms
        "description",
        "particulars",
        "narration",
        "transaction particulars",
        "transaction description",
        "remarks",
        "details",
        "transaction details",
        # Abbreviated forms
        "desc",
        "particular",
        "naration",  # Common typo in some statements
        "remark",
        # Bank-specific variations
        "mode of transaction",
        "mode",
        "payment details",
        "transaction remarks",
        "trans particulars",
        "txn particulars",
        "transaction narration",
        "trans description",
    ],

    # -------------------------------------------------------------------------
    # Reference/ID Columns
    # -------------------------------------------------------------------------
    "reference": [
        # Primary terms
        "reference",
        "reference number",
        "ref no",
        "ref number",
        "transaction id",
        "transaction reference",
        # Cheque-related
        "cheque no",
        "cheque number",
        "chq no",
        "chq number",
        "check no",
        "check number",
        "instrument no",
        "instrument number",
        # UTR and banking codes
        "utr",
        "utr no",
        "utr number",
        "rrn",
        "rrn no",
        # Abbreviated forms
        "ref",
        "txn id",
        "trans id",
        "txn ref",
        "trans ref",
        # Bank-specific variations
        "branch code",
        "clearing ref",
        "clearing reference",
        "tran id",
        "transaction no",
        "voucher no",
        "voucher number",
        "slip no",
    ],

    # -------------------------------------------------------------------------
    # Debit/Withdrawal Columns
    # -------------------------------------------------------------------------
    "debit": [
        # Primary terms
        "debit",
        "debit amount",
        "withdrawal",
        "withdrawals",
        "withdrawal amount",
        # Abbreviated forms
        "dr",
        "dr amt",
        "debit amt",
        "dr amount",
        "withdraw",
        # Bank-specific variations
        "money out",
        "outflow",
        "amount debited",
        "debited amount",
        "debit (dr)",
        "debit(dr)",
        "payments",
        "payment",
    ],

    # -------------------------------------------------------------------------
    # Credit/Deposit Columns
    # -------------------------------------------------------------------------
    "credit": [
        # Primary terms
        "credit",
        "credit amount",
        "deposit",
        "deposits",
        "deposit amount",
        # Abbreviated forms
        "cr",
        "cr amt",
        "credit amt",
        "cr amount",
        # Bank-specific variations
        "money in",
        "inflow",
        "amount credited",
        "credited amount",
        "credit (cr)",
        "credit(cr)",
        "receipts",
        "receipt",
    ],

    # -------------------------------------------------------------------------
    # Balance Columns
    # -------------------------------------------------------------------------
    "balance": [
        # Primary terms
        "balance",
        "closing balance",
        "running balance",
        "available balance",
        "current balance",
        # Abbreviated forms
        "bal",
        "bal amt",
        "balance amt",
        "closing bal",
        "avail bal",
        "avl bal",
        "run bal",
        # Bank-specific variations
        "account balance",
        "ledger balance",
        "book balance",
        "net balance",
        "cumulative balance",
        "balance after transaction",
        "balance (inr)",
    ],

    # -------------------------------------------------------------------------
    # Generic Amount Columns
    # -------------------------------------------------------------------------
    "amount": [
        # Primary terms
        "amount",
        "transaction amount",
        "txn amount",
        "trans amount",
        # Abbreviated forms
        "amt",
        "txn amt",
        "trans amt",
        # Bank-specific variations
        "amount (inr)",
        "amount (rs)",
        "amount rs",
        "value",
        "sum",
    ],

    # -------------------------------------------------------------------------
    # Serial Number Columns
    # -------------------------------------------------------------------------
    "serial": [
        # Primary terms
        "serial",
        "serial no",
        "serial number",
        "sl no",
        "s no",
        "sno",
        "sr no",
        "#",
        "no",
        "no.",
        "number",
        # Abbreviated forms
        "sl",
        "sr",
        "s.no",
        "s.no.",
        "sl.no",
        "sl.no.",
        "sr.no",
        "sr.no.",
    ],
}


# =============================================================================
# Multi-word Header Patterns
# =============================================================================

# Some headers span multiple rows or have complex formatting
# These patterns help identify such cases
MULTI_ROW_HEADER_PATTERNS: list[str] = [
    r"debit\s*/\s*credit",  # "Debit / Credit" split
    r"withdrawal\s*/\s*deposit",
    r"dr\s*/\s*cr",
    r"amount\s*\(\s*dr\s*\)",
    r"amount\s*\(\s*cr\s*\)",
]


# =============================================================================
# Header Exclusion Patterns
# =============================================================================

# Text that looks like headers but should be excluded
HEADER_EXCLUSIONS: list[str] = [
    "opening balance",  # Usually a summary row, not a header
    "closing balance",  # Usually a summary row when appearing alone
    "total",
    "totals",
    "subtotal",
    "sub total",
    "grand total",
    "statement period",
    "account number",
    "account holder",
    "branch",
    "ifsc",
    "page",
]


# =============================================================================
# Utility Functions
# =============================================================================

def normalize_header_text(text: str) -> str:
    """
    Normalize header text for matching.

    Performs the following normalizations:
    1. Convert to lowercase
    2. Replace multiple whitespace with single space
    3. Remove leading/trailing whitespace
    4. Remove common punctuation (but preserve meaningful ones)

    Args:
        text: Raw header text from PDF

    Returns:
        Normalized text suitable for keyword matching

    Example:
        >>> normalize_header_text("  Transaction   DATE  ")
        'transaction date'
        >>> normalize_header_text("Debit (Dr.)")
        'debit (dr)'
    """
    if not text:
        return ""

    # Convert to lowercase
    normalized = text.lower()

    # Replace multiple whitespace with single space
    normalized = re.sub(r"\s+", " ", normalized)

    # Remove leading/trailing whitespace
    normalized = normalized.strip()

    # Remove trailing periods but keep parentheses
    normalized = re.sub(r"\.+$", "", normalized)

    return normalized


def get_keywords_for_type(semantic_type: str) -> list[str]:
    """
    Get all keywords for a specific semantic type.

    Args:
        semantic_type: One of the defined semantic types
            (date, description, reference, debit, credit, balance, amount, serial)

    Returns:
        List of keywords for the type, or empty list if type not found

    Example:
        >>> keywords = get_keywords_for_type("date")
        >>> "transaction date" in keywords
        True
    """
    return HEADER_KEYWORDS.get(semantic_type, [])


def find_semantic_type(text: str, threshold: float = 0.0) -> Optional[tuple[str, float]]:
    """
    Find the semantic type for a given header text.

    Uses exact and contains matching against the keyword database.
    Returns the best matching semantic type with confidence score.

    Args:
        text: Header text to classify
        threshold: Minimum confidence threshold (0.0-1.0)

    Returns:
        Tuple of (semantic_type, confidence) or None if no match above threshold

    Confidence Scoring:
        - Exact match: 1.0
        - Text contains keyword: 0.8
        - Keyword contains text: 0.6

    Example:
        >>> find_semantic_type("Transaction Date")
        ('date', 1.0)
        >>> find_semantic_type("Txn Details")
        ('description', 0.8)
    """
    normalized = normalize_header_text(text)

    if not normalized:
        return None

    best_match: Optional[tuple[str, float]] = None
    best_confidence = 0.0

    for semantic_type, keywords in HEADER_KEYWORDS.items():
        for keyword in keywords:
            confidence = 0.0

            # Exact match (highest confidence)
            if normalized == keyword:
                confidence = 1.0
            # Text contains the keyword
            elif keyword in normalized:
                # Longer keywords matching = higher confidence
                confidence = 0.7 + (0.2 * len(keyword) / len(normalized))
                confidence = min(confidence, 0.9)  # Cap at 0.9 for contains
            # Keyword contains the text (less confident)
            elif normalized in keyword and len(normalized) >= 3:
                confidence = 0.5 + (0.2 * len(normalized) / len(keyword))
                confidence = min(confidence, 0.7)

            if confidence > best_confidence:
                best_confidence = confidence
                best_match = (semantic_type, confidence)

    if best_match and best_confidence >= threshold:
        return best_match

    return None


def is_likely_header_row(texts: list[str], min_matches: int = 2) -> bool:
    """
    Determine if a list of texts likely represents a header row.

    A row is considered a header if it contains multiple recognized
    header keywords and does not contain exclusion patterns.

    Args:
        texts: List of cell texts from a potential header row
        min_matches: Minimum number of semantic type matches required

    Returns:
        True if the row appears to be a header row

    Example:
        >>> is_likely_header_row(["Date", "Description", "Debit", "Credit", "Balance"])
        True
        >>> is_likely_header_row(["01/01/2024", "ATM Withdrawal", "500.00", "", "10000.00"])
        False
    """
    matches = 0
    matched_types: set[str] = set()

    for text in texts:
        normalized = normalize_header_text(text)

        # Check for exclusions
        if any(excl in normalized for excl in HEADER_EXCLUSIONS):
            continue

        result = find_semantic_type(text, threshold=0.5)
        if result:
            semantic_type, confidence = result
            if semantic_type not in matched_types:
                matches += 1
                matched_types.add(semantic_type)

    return matches >= min_matches
