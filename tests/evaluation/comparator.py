"""
Transaction Comparator.

Compares extracted transactions against ground truth to determine matches.

Matching Strategy:
    1. Primary key: Date + normalized description
    2. Tiebreaker: Amount (debit or credit)
    3. A transaction matches if ALL fields match

Example:
    >>> comparator = TransactionComparator()
    >>> result = comparator.compare(ground_truth, extracted)
    >>> print(f"Matched: {result.matched_count}/{result.gt_count}")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from enum import Enum
from typing import Optional, Any
import logging

from tests.evaluation.csv_loader import (
    GroundTruthTransaction,
    normalize_text,
    parse_date,
    parse_amount,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Error Types
# =============================================================================

class MismatchType(Enum):
    """Types of mismatches between ground truth and extracted."""

    MATCH = "MATCH"  # Perfect match
    DATE_MISMATCH = "DATE_MISMATCH"
    DESCRIPTION_MISMATCH = "DESC_MISMATCH"
    DEBIT_MISMATCH = "DEBIT_MISMATCH"
    CREDIT_MISMATCH = "CREDIT_MISMATCH"
    BALANCE_MISMATCH = "BALANCE_MISMATCH"
    REFERENCE_MISMATCH = "REF_MISMATCH"
    NOT_EXTRACTED = "NOT_EXTRACTED"  # GT row not found in extracted
    EXTRA_ROW = "EXTRA_ROW"  # Extracted row not in GT


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class FieldComparison:
    """Comparison result for a single field."""

    field_name: str
    gt_value: Any
    extracted_value: Any
    matches: bool

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "field": self.field_name,
            "ground_truth": str(self.gt_value) if self.gt_value is not None else None,
            "extracted": str(self.extracted_value) if self.extracted_value is not None else None,
            "matches": self.matches,
        }


@dataclass
class TransactionMatch:
    """
    Result of matching a single transaction.

    Attributes:
        gt_row: Ground truth row number (None if extra extracted row)
        extracted_row: Extracted row index (None if not extracted)
        is_match: Whether all fields match
        mismatch_type: Type of mismatch if not a match
        field_comparisons: Detailed field-by-field comparison
        gt_transaction: The ground truth transaction
        extracted_transaction: The extracted transaction (dict)
    """

    gt_row: Optional[int] = None
    extracted_row: Optional[int] = None
    is_match: bool = False
    mismatch_type: MismatchType = MismatchType.NOT_EXTRACTED
    field_comparisons: list[FieldComparison] = field(default_factory=list)
    gt_transaction: Optional[GroundTruthTransaction] = None
    extracted_transaction: Optional[dict[str, Any]] = None

    @property
    def primary_mismatch_field(self) -> Optional[str]:
        """Get the first field that didn't match."""
        for fc in self.field_comparisons:
            if not fc.matches:
                return fc.field_name
        return None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "gt_row": self.gt_row,
            "extracted_row": self.extracted_row,
            "is_match": self.is_match,
            "mismatch_type": self.mismatch_type.value,
            "field_comparisons": [fc.to_dict() for fc in self.field_comparisons],
        }


@dataclass
class ComparisonResult:
    """
    Complete comparison result for a file.

    Attributes:
        file_name: Name of the file compared
        gt_count: Number of ground truth transactions
        extracted_count: Number of extracted transactions
        matched_count: Number of matched transactions
        missed_count: GT rows not extracted
        extra_count: Extracted rows not in GT
        matches: List of all transaction matches
        accuracy: Matched / GT count
    """

    file_name: str = ""
    gt_count: int = 0
    extracted_count: int = 0
    matched_count: int = 0
    missed_count: int = 0
    extra_count: int = 0
    matches: list[TransactionMatch] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        """Calculate accuracy as matched / ground truth count."""
        if self.gt_count == 0:
            return 1.0 if self.extracted_count == 0 else 0.0
        return self.matched_count / self.gt_count

    @property
    def is_perfect(self) -> bool:
        """Check if all transactions matched perfectly."""
        return self.matched_count == self.gt_count and self.extra_count == 0

    def get_mismatched(self) -> list[TransactionMatch]:
        """Get all non-matching transactions."""
        return [m for m in self.matches if not m.is_match]

    def get_missed(self) -> list[TransactionMatch]:
        """Get GT rows that were not extracted."""
        return [m for m in self.matches if m.mismatch_type == MismatchType.NOT_EXTRACTED]

    def get_extra(self) -> list[TransactionMatch]:
        """Get extracted rows not in GT."""
        return [m for m in self.matches if m.mismatch_type == MismatchType.EXTRA_ROW]

    def get_error_breakdown(self) -> dict[str, int]:
        """Get count of each error type."""
        breakdown: dict[str, int] = {}
        for match in self.matches:
            if not match.is_match:
                error_type = match.mismatch_type.value
                breakdown[error_type] = breakdown.get(error_type, 0) + 1
        return breakdown

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "file_name": self.file_name,
            "gt_count": self.gt_count,
            "extracted_count": self.extracted_count,
            "matched_count": self.matched_count,
            "missed_count": self.missed_count,
            "extra_count": self.extra_count,
            "accuracy": self.accuracy,
            "is_perfect": self.is_perfect,
            "error_breakdown": self.get_error_breakdown(),
        }


# =============================================================================
# Comparator
# =============================================================================

class TransactionComparator:
    """
    Compares extracted transactions against ground truth.

    Matching Strategy:
        1. For each GT transaction, find best matching extracted transaction
        2. Match by date + description similarity
        3. Verify all fields match for a successful match
        4. Track unmatched GT rows (missed) and unmatched extracted rows (extra)

    Example:
        >>> comparator = TransactionComparator()
        >>> result = comparator.compare(gt_transactions, extracted_rows)
    """

    def __init__(
        self,
        description_similarity_threshold: float = 0.8,
    ):
        """
        Initialize the comparator.

        Args:
            description_similarity_threshold: Min similarity for description match
        """
        self.description_similarity_threshold = description_similarity_threshold

    def compare(
        self,
        ground_truth: list[GroundTruthTransaction],
        extracted: list[dict[str, Any]],
        file_name: str = "",
    ) -> ComparisonResult:
        """
        Compare extracted transactions against ground truth.

        Args:
            ground_truth: List of ground truth transactions
            extracted: List of extracted transaction dicts
            file_name: Name of the file for reporting

        Returns:
            ComparisonResult with detailed comparison
        """
        result = ComparisonResult(
            file_name=file_name,
            gt_count=len(ground_truth),
            extracted_count=len(extracted),
        )

        # Track which extracted rows have been matched
        matched_extracted_indices: set[int] = set()

        # Match each GT transaction
        for gt_txn in ground_truth:
            match = self._find_best_match(gt_txn, extracted, matched_extracted_indices)
            result.matches.append(match)

            if match.is_match:
                result.matched_count += 1
                if match.extracted_row is not None:
                    matched_extracted_indices.add(match.extracted_row)
            elif match.mismatch_type == MismatchType.NOT_EXTRACTED:
                result.missed_count += 1
            else:
                # Partial match (some fields matched but not all)
                if match.extracted_row is not None:
                    matched_extracted_indices.add(match.extracted_row)

        # Find extra extracted rows (not matched to any GT)
        for idx, ext_txn in enumerate(extracted):
            if idx not in matched_extracted_indices:
                result.extra_count += 1
                result.matches.append(TransactionMatch(
                    gt_row=None,
                    extracted_row=idx,
                    is_match=False,
                    mismatch_type=MismatchType.EXTRA_ROW,
                    extracted_transaction=ext_txn,
                ))

        logger.debug(
            f"{file_name}: {result.matched_count}/{result.gt_count} matched, "
            f"{result.missed_count} missed, {result.extra_count} extra"
        )

        return result

    def _find_best_match(
        self,
        gt_txn: GroundTruthTransaction,
        extracted: list[dict[str, Any]],
        already_matched: set[int],
    ) -> TransactionMatch:
        """
        Find the best matching extracted transaction for a GT transaction.

        Args:
            gt_txn: Ground truth transaction to match
            extracted: List of extracted transactions
            already_matched: Set of already matched extracted indices

        Returns:
            TransactionMatch with best match or NOT_EXTRACTED
        """
        best_match: Optional[TransactionMatch] = None
        best_score = -1

        for idx, ext_txn in enumerate(extracted):
            if idx in already_matched:
                continue

            # Check if this could be a match
            match_result = self._compare_transactions(gt_txn, ext_txn, idx)

            # Score based on number of matching fields
            score = sum(1 for fc in match_result.field_comparisons if fc.matches)

            # Prioritize date match
            date_matches = any(
                fc.field_name == "date" and fc.matches
                for fc in match_result.field_comparisons
            )
            if date_matches:
                score += 10

            if score > best_score:
                best_score = score
                best_match = match_result

        if best_match is None or best_score < 1:
            # No match found
            return TransactionMatch(
                gt_row=gt_txn.row_number,
                extracted_row=None,
                is_match=False,
                mismatch_type=MismatchType.NOT_EXTRACTED,
                gt_transaction=gt_txn,
            )

        return best_match

    def _compare_transactions(
        self,
        gt_txn: GroundTruthTransaction,
        ext_txn: dict[str, Any],
        ext_idx: int,
    ) -> TransactionMatch:
        """
        Compare a GT transaction with an extracted transaction.

        Args:
            gt_txn: Ground truth transaction
            ext_txn: Extracted transaction dict
            ext_idx: Index of extracted transaction

        Returns:
            TransactionMatch with field-by-field comparison
        """
        comparisons: list[FieldComparison] = []

        # Compare date
        ext_date = self._extract_date(ext_txn)
        date_matches = self._compare_dates(gt_txn.date, ext_date)
        comparisons.append(FieldComparison(
            field_name="date",
            gt_value=gt_txn.date,
            extracted_value=ext_date,
            matches=date_matches,
        ))

        # Compare description
        ext_desc = ext_txn.get("description", "")
        desc_matches = self._compare_descriptions(gt_txn.description, ext_desc)
        comparisons.append(FieldComparison(
            field_name="description",
            gt_value=gt_txn.description,
            extracted_value=ext_desc,
            matches=desc_matches,
        ))

        # Compare debit
        ext_debit = self._extract_amount(ext_txn, "debit")
        debit_matches = self._compare_amounts(gt_txn.debit, ext_debit)
        comparisons.append(FieldComparison(
            field_name="debit",
            gt_value=gt_txn.debit,
            extracted_value=ext_debit,
            matches=debit_matches,
        ))

        # Compare credit
        ext_credit = self._extract_amount(ext_txn, "credit")
        credit_matches = self._compare_amounts(gt_txn.credit, ext_credit)
        comparisons.append(FieldComparison(
            field_name="credit",
            gt_value=gt_txn.credit,
            extracted_value=ext_credit,
            matches=credit_matches,
        ))

        # Compare balance
        ext_balance = self._extract_amount(ext_txn, "balance")
        balance_matches = self._compare_amounts(gt_txn.balance, ext_balance)
        comparisons.append(FieldComparison(
            field_name="balance",
            gt_value=gt_txn.balance,
            extracted_value=ext_balance,
            matches=balance_matches,
        ))

        # Compare reference (optional)
        ext_ref = ext_txn.get("reference", "")
        ref_matches = self._compare_references(gt_txn.reference, ext_ref)
        comparisons.append(FieldComparison(
            field_name="reference",
            gt_value=gt_txn.reference,
            extracted_value=ext_ref,
            matches=ref_matches,
        ))

        # Determine overall match
        all_match = all(fc.matches for fc in comparisons)

        # Determine mismatch type
        mismatch_type = MismatchType.MATCH
        if not all_match:
            if not date_matches:
                mismatch_type = MismatchType.DATE_MISMATCH
            elif not desc_matches:
                mismatch_type = MismatchType.DESCRIPTION_MISMATCH
            elif not debit_matches:
                mismatch_type = MismatchType.DEBIT_MISMATCH
            elif not credit_matches:
                mismatch_type = MismatchType.CREDIT_MISMATCH
            elif not balance_matches:
                mismatch_type = MismatchType.BALANCE_MISMATCH
            elif not ref_matches:
                mismatch_type = MismatchType.REFERENCE_MISMATCH

        return TransactionMatch(
            gt_row=gt_txn.row_number,
            extracted_row=ext_idx,
            is_match=all_match,
            mismatch_type=mismatch_type,
            field_comparisons=comparisons,
            gt_transaction=gt_txn,
            extracted_transaction=ext_txn,
        )

    def _extract_date(self, ext_txn: dict[str, Any]) -> Optional[date]:
        """Extract date from extracted transaction."""
        date_val = ext_txn.get("transaction_date") or ext_txn.get("date")

        if isinstance(date_val, date):
            return date_val
        elif isinstance(date_val, str):
            return parse_date(date_val)

        return None

    def _extract_amount(
        self,
        ext_txn: dict[str, Any],
        field: str,
    ) -> Optional[Decimal]:
        """Extract amount from extracted transaction."""
        # Try direct field
        amount_val = ext_txn.get(field) or ext_txn.get(f"{field}_amount")

        if isinstance(amount_val, Decimal):
            return amount_val
        elif isinstance(amount_val, (int, float)):
            return Decimal(str(amount_val))
        elif isinstance(amount_val, str):
            return parse_amount(amount_val)

        return None

    def _compare_dates(
        self,
        gt_date: Optional[date],
        ext_date: Optional[date],
    ) -> bool:
        """Compare two dates."""
        if gt_date is None and ext_date is None:
            return True
        if gt_date is None or ext_date is None:
            return False
        return gt_date == ext_date

    def _compare_descriptions(
        self,
        gt_desc: str,
        ext_desc: str,
    ) -> bool:
        """Compare two descriptions using normalized comparison."""
        gt_normalized = normalize_text(gt_desc)
        ext_normalized = normalize_text(ext_desc)

        # Exact match after normalization
        if gt_normalized == ext_normalized:
            return True

        # Check if one contains the other (for partial extractions)
        if gt_normalized in ext_normalized or ext_normalized in gt_normalized:
            # Only accept if similarity is high enough
            shorter = min(len(gt_normalized), len(ext_normalized))
            longer = max(len(gt_normalized), len(ext_normalized))
            if longer > 0 and shorter / longer >= self.description_similarity_threshold:
                return True

        return False

    def _compare_amounts(
        self,
        gt_amount: Optional[Decimal],
        ext_amount: Optional[Decimal],
    ) -> bool:
        """Compare two amounts."""
        if gt_amount is None and ext_amount is None:
            return True
        if gt_amount is None or ext_amount is None:
            return False

        # Compare absolute values (handle sign differences)
        return abs(gt_amount) == abs(ext_amount)

    def _compare_references(
        self,
        gt_ref: Optional[str],
        ext_ref: Optional[str],
    ) -> bool:
        """Compare two references."""
        # Reference is optional - if GT doesn't have it, always match
        if gt_ref is None or gt_ref.strip() == "":
            return True

        if ext_ref is None or ext_ref.strip() == "":
            return False

        return normalize_text(gt_ref) == normalize_text(ext_ref)
