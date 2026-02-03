"""
Row Builder for Transaction Tables.

This module converts extracted cell rows into structured TransactionRow objects,
handling wrapped transactions that span multiple visual rows.

Key Features:
    - Row merging: Detects and merges wrapped transaction rows
    - Value parsing: Parses dates, amounts using utilities
    - Column mapping: Maps cells to semantic transaction fields
    - Missing cell handling: Inserts null placeholders

Row Merging Logic:
    A transaction may span multiple visual rows when text wraps.
    Detection signals:
    1. Date column empty in subsequent row
    2. Balance column empty (only complete transactions have balance)
    3. Indentation in description column

Example Usage:
    >>> from tables.recognizer.row_builder import RowBuilder
    >>> builder = RowBuilder(table_definition)
    >>> transactions = builder.build_transactions(extracted_rows)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from typing import Any, Optional
import logging

from tables.models.table import TableDefinition, ColumnDefinition
from tables.models.transaction import (
    TransactionRow,
    TransactionTable,
    ColumnSchema,
    ParseWarning,
    SemanticType,
)
from tables.recognizer.cell_extractor import ExtractedRow, Cell
from tables.utils.parsing import (
    parse_date,
    parse_amount,
    clean_text,
    clean_description,
    is_empty_cell,
    extract_reference,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

# Semantic types that indicate date columns
DATE_TYPES = {"date", "value_date", "transaction_date", "posting_date"}

# Semantic types that indicate amount columns
AMOUNT_TYPES = {"debit", "credit", "amount", "balance", "withdrawal", "deposit"}

# Semantic types that indicate balance columns (used for merge detection)
BALANCE_TYPES = {"balance", "running_balance", "closing_balance"}


# =============================================================================
# Row Merge Context
# =============================================================================

@dataclass
class MergeContext:
    """
    Context for determining whether rows should be merged.

    Tracks information about the table structure that affects
    merge decisions.

    Attributes:
        has_date_column: Whether table has a date column
        has_balance_column: Whether table has a balance column
        date_column_index: Index of date column
        balance_column_index: Index of balance column
    """

    has_date_column: bool = False
    has_balance_column: bool = False
    date_column_index: Optional[int] = None
    balance_column_index: Optional[int] = None

    @classmethod
    def from_columns(cls, columns: list[ColumnDefinition]) -> MergeContext:
        """Create merge context from column definitions."""
        context = cls()

        for col in columns:
            semantic = col.semantic_type
            if semantic in DATE_TYPES or semantic == "date":
                context.has_date_column = True
                context.date_column_index = col.column_id
            if semantic in BALANCE_TYPES or semantic == "balance":
                context.has_balance_column = True
                context.balance_column_index = col.column_id

        return context


# =============================================================================
# Row Builder
# =============================================================================

class RowBuilder:
    """
    Builds TransactionRow objects from extracted cell rows.

    Handles:
    - Merging wrapped transaction rows
    - Parsing cell values (dates, amounts)
    - Mapping cells to semantic transaction fields
    - Tracking parse warnings

    Attributes:
        table_definition: Source table definition
        columns: Column definitions for cell mapping
        merge_context: Context for merge decisions

    Example:
        >>> builder = RowBuilder(table_definition)
        >>> rows = builder.build_transactions(extracted_rows)
        >>> for row in rows:
        ...     print(f"{row.transaction_date}: {row.description}")
    """

    def __init__(
        self,
        table_definition: TableDefinition,
        enable_merge: bool = True,
    ):
        """
        Initialize the row builder.

        Args:
            table_definition: Table definition with column info
            enable_merge: Whether to enable row merging
        """
        self.table_definition = table_definition
        self.columns = table_definition.columns
        self.enable_merge = enable_merge
        self.merge_context = MergeContext.from_columns(self.columns)
        self._warnings: list[ParseWarning] = []

    def build_transactions(
        self,
        extracted_rows: list[ExtractedRow],
    ) -> list[TransactionRow]:
        """
        Build TransactionRow objects from extracted rows.

        Applies row merging if enabled, then parses each row.

        Args:
            extracted_rows: List of ExtractedRow objects

        Returns:
            List of TransactionRow objects
        """
        self._warnings = []

        if not extracted_rows:
            return []

        # Apply row merging if enabled
        if self.enable_merge:
            merged_rows = self._merge_wrapped_rows(extracted_rows)
        else:
            merged_rows = [[row] for row in extracted_rows]

        # Build transaction rows
        transactions = []
        for row_idx, row_group in enumerate(merged_rows):
            try:
                txn = self._build_single_transaction(row_idx, row_group)
                if txn is not None:
                    transactions.append(txn)
            except Exception as e:
                logger.warning(
                    f"Failed to build transaction from row {row_idx}: {e}"
                )
                self._add_warning(
                    row_idx,
                    "row",
                    f"Failed to parse row: {str(e)}",
                )

        logger.debug(
            f"Built {len(transactions)} transactions from "
            f"{len(extracted_rows)} extracted rows"
        )

        return transactions

    def get_warnings(self) -> list[ParseWarning]:
        """Get all parse warnings from the last build."""
        return self._warnings

    def build_column_schemas(self) -> list[ColumnSchema]:
        """
        Build ColumnSchema objects from table definition columns.

        Returns:
            List of ColumnSchema objects
        """
        schemas = []
        for col in self.columns:
            semantic = self._normalize_semantic_type(col.semantic_type)
            data_type = self._get_python_data_type(semantic)

            schema = ColumnSchema(
                name=col.header_text or f"Column_{col.column_id}",
                semantic_type=semantic,
                data_type=data_type,
                nullable=semantic not in {SemanticType.DESCRIPTION},
                source_column_index=col.column_id,
                source_header_text=col.header_text,
            )
            schemas.append(schema)

        return schemas

    def _merge_wrapped_rows(
        self,
        rows: list[ExtractedRow],
    ) -> list[list[ExtractedRow]]:
        """
        Merge rows that represent wrapped text from a single transaction.

        Detection signals:
        1. Date column empty in subsequent row
        2. Balance column empty (only populated for complete transactions)

        Args:
            rows: List of extracted rows

        Returns:
            List of row groups (each group = one logical transaction)
        """
        if not rows:
            return []

        merged_groups: list[list[ExtractedRow]] = []
        current_group: list[ExtractedRow] = []

        for row in rows:
            if not current_group:
                # Start new group
                current_group.append(row)
            elif self._should_merge_with_previous(row, current_group[-1]):
                # Merge with current group
                current_group.append(row)
            else:
                # Start new group
                merged_groups.append(current_group)
                current_group = [row]

        # Don't forget the last group
        if current_group:
            merged_groups.append(current_group)

        logger.debug(
            f"Merged {len(rows)} extracted rows into "
            f"{len(merged_groups)} transaction groups"
        )

        return merged_groups

    def _should_merge_with_previous(
        self,
        current: ExtractedRow,
        previous: ExtractedRow,
    ) -> bool:
        """
        Determine if current row should merge with previous row.

        A row is merged if it appears to be a continuation of the previous
        transaction. Primary signal: date column is empty.

        Args:
            current: Current row being processed
            previous: Previous row (last in current group)

        Returns:
            True if rows should be merged
        """
        # First check: If current row has a non-empty date, it's a new transaction
        if self.merge_context.has_date_column:
            date_col_id = self.merge_context.date_column_index
            if date_col_id is not None:
                date_cell = current.cells.get(date_col_id)
                # If date is present and non-empty, don't merge
                if date_cell and not is_empty_cell(date_cell.text):
                    return False
                # If date is empty, this suggests continuation - merge
                if date_cell and is_empty_cell(date_cell.text):
                    return True

        # Rule 2: If no date column but balance column exists and is empty, merge
        # (only complete transactions have balance)
        if self.merge_context.has_balance_column and not self.merge_context.has_date_column:
            balance_col_id = self.merge_context.balance_column_index
            if balance_col_id is not None:
                balance_cell = current.cells.get(balance_col_id)
                if balance_cell and is_empty_cell(balance_cell.text):
                    return True

        # Rule 3: Check if all required fields are empty
        # (suggesting this is continuation text)
        has_any_key_field = False
        for col in self.columns:
            if col.semantic_type in {"date", "debit", "credit", "amount", "balance"}:
                cell = current.cells.get(col.column_id)
                if cell and not is_empty_cell(cell.text):
                    has_any_key_field = True
                    break

        if not has_any_key_field:
            # Only description has content - likely a continuation
            for col in self.columns:
                if col.semantic_type == "description":
                    cell = current.cells.get(col.column_id)
                    if cell and not is_empty_cell(cell.text):
                        return True

        return False

    def _build_single_transaction(
        self,
        row_idx: int,
        row_group: list[ExtractedRow],
    ) -> Optional[TransactionRow]:
        """
        Build a single TransactionRow from a group of extracted rows.

        Args:
            row_idx: Logical row index
            row_group: Group of extracted rows (may be merged)

        Returns:
            TransactionRow or None if row is empty
        """
        # Check if all rows are empty
        if all(row.all_cells_empty for row in row_group):
            return None

        # Merge cells from all rows in the group
        merged_cells = self._merge_cells(row_group)

        # Get source information
        source_page = row_group[0].page_number
        source_indices = [row.row_index for row in row_group]

        # Create transaction row
        txn = TransactionRow(
            row_id=row_idx,
            source_page=source_page,
            source_row_indices=source_indices,
        )

        # Parse each cell and assign to transaction fields
        for col in self.columns:
            cell = merged_cells.get(col.column_id)
            if cell is None:
                continue

            raw_text = cell.text
            txn.raw_cells[col.semantic_type or f"col_{col.column_id}"] = raw_text

            if is_empty_cell(raw_text):
                continue

            self._assign_cell_value(txn, col, raw_text, row_idx)

        # Calculate confidence
        txn.confidence = self._calculate_row_confidence(txn)

        return txn

    def _merge_cells(
        self,
        rows: list[ExtractedRow],
    ) -> dict[int, Cell]:
        """
        Merge cells from multiple rows.

        For each column, concatenates text from all rows (with space separator).

        Args:
            rows: List of rows to merge

        Returns:
            Dictionary mapping column_id to merged Cell
        """
        merged: dict[int, Cell] = {}

        for row in rows:
            for col_id, cell in row.cells.items():
                if col_id not in merged:
                    merged[col_id] = Cell(
                        column_index=cell.column_index,
                        column=cell.column,
                    )

                if not is_empty_cell(cell.text):
                    if merged[col_id].text:
                        merged[col_id].text += " " + cell.text
                    else:
                        merged[col_id].text = cell.text
                    merged[col_id].is_empty = False
                    merged[col_id].words.extend(cell.words)

        return merged

    def _assign_cell_value(
        self,
        txn: TransactionRow,
        column: ColumnDefinition,
        raw_text: str,
        row_idx: int,
    ) -> None:
        """
        Parse and assign a cell value to the appropriate transaction field.

        Args:
            txn: TransactionRow to update
            column: Column definition
            raw_text: Raw cell text
            row_idx: Row index for warnings
        """
        semantic_type = column.semantic_type or ""
        cleaned_text = clean_text(raw_text)

        # Date fields
        if semantic_type in {"date", "transaction_date", "posting_date"}:
            parsed_date = parse_date(cleaned_text)
            if parsed_date:
                txn.transaction_date = parsed_date
            else:
                self._add_warning(row_idx, semantic_type, f"Could not parse date: {raw_text}", raw_text)

        elif semantic_type == "value_date":
            parsed_date = parse_date(cleaned_text)
            if parsed_date:
                txn.value_date = parsed_date
            else:
                self._add_warning(row_idx, semantic_type, f"Could not parse value date: {raw_text}", raw_text)

        # Description
        elif semantic_type == "description":
            txn.description = clean_description(cleaned_text)

        # Reference
        elif semantic_type == "reference":
            txn.reference = extract_reference(cleaned_text) or cleaned_text

        # Amount fields
        elif semantic_type in {"debit", "withdrawal"}:
            amount = parse_amount(cleaned_text)
            if amount is not None:
                txn.debit_amount = abs(amount)
            else:
                self._add_warning(row_idx, semantic_type, f"Could not parse debit: {raw_text}", raw_text)

        elif semantic_type in {"credit", "deposit"}:
            amount = parse_amount(cleaned_text)
            if amount is not None:
                txn.credit_amount = abs(amount)
            else:
                self._add_warning(row_idx, semantic_type, f"Could not parse credit: {raw_text}", raw_text)

        elif semantic_type == "amount":
            # Generic amount - need to determine if debit or credit
            amount = parse_amount(cleaned_text)
            if amount is not None:
                if amount < 0:
                    txn.debit_amount = abs(amount)
                else:
                    txn.credit_amount = amount
            else:
                self._add_warning(row_idx, semantic_type, f"Could not parse amount: {raw_text}", raw_text)

        elif semantic_type in {"balance", "running_balance", "closing_balance"}:
            amount = parse_amount(cleaned_text)
            if amount is not None:
                txn.balance = amount
            else:
                self._add_warning(row_idx, semantic_type, f"Could not parse balance: {raw_text}", raw_text)

    def _calculate_row_confidence(self, txn: TransactionRow) -> float:
        """
        Calculate confidence score for a transaction row.

        Higher confidence for rows with:
        - Valid date
        - Non-empty description
        - At least one valid amount

        Args:
            txn: TransactionRow to evaluate

        Returns:
            Confidence score (0.0 - 1.0)
        """
        confidence = 1.0

        # Deduct for missing date
        if txn.transaction_date is None:
            confidence -= 0.3

        # Deduct for empty description
        if not txn.description:
            confidence -= 0.2

        # Deduct for missing amounts
        if txn.debit_amount is None and txn.credit_amount is None:
            confidence -= 0.3

        # Deduct for each warning
        confidence -= 0.05 * len(txn.warnings)

        return max(0.0, min(1.0, confidence))

    def _normalize_semantic_type(self, semantic_type: Optional[str]) -> SemanticType:
        """Convert string semantic type to SemanticType enum."""
        if not semantic_type:
            return SemanticType.OTHER

        mapping = {
            "date": SemanticType.DATE,
            "transaction_date": SemanticType.DATE,
            "posting_date": SemanticType.DATE,
            "value_date": SemanticType.VALUE_DATE,
            "description": SemanticType.DESCRIPTION,
            "narration": SemanticType.DESCRIPTION,
            "particulars": SemanticType.DESCRIPTION,
            "reference": SemanticType.REFERENCE,
            "ref": SemanticType.REFERENCE,
            "debit": SemanticType.DEBIT,
            "withdrawal": SemanticType.DEBIT,
            "credit": SemanticType.CREDIT,
            "deposit": SemanticType.CREDIT,
            "balance": SemanticType.BALANCE,
            "running_balance": SemanticType.BALANCE,
            "closing_balance": SemanticType.BALANCE,
            "amount": SemanticType.AMOUNT,
        }

        return mapping.get(semantic_type.lower(), SemanticType.OTHER)

    def _get_python_data_type(self, semantic_type: SemanticType) -> str:
        """Get Python data type string for a semantic type."""
        if semantic_type in {SemanticType.DATE, SemanticType.VALUE_DATE}:
            return "date"
        elif semantic_type in {
            SemanticType.DEBIT,
            SemanticType.CREDIT,
            SemanticType.BALANCE,
            SemanticType.AMOUNT,
        }:
            return "Decimal"
        else:
            return "str"

    def _add_warning(
        self,
        row_idx: int,
        column_name: str,
        message: str,
        raw_value: str = "",
    ) -> None:
        """Add a parse warning."""
        warning = ParseWarning(
            row_id=row_idx,
            column_name=column_name,
            message=message,
            raw_value=raw_value,
        )
        self._warnings.append(warning)


# =============================================================================
# Convenience Functions
# =============================================================================

def build_transactions(
    table_definition: TableDefinition,
    extracted_rows: list[ExtractedRow],
    enable_merge: bool = True,
) -> tuple[list[TransactionRow], list[ParseWarning]]:
    """
    Convenience function to build transactions from extracted rows.

    Args:
        table_definition: Table definition with column info
        extracted_rows: List of extracted rows
        enable_merge: Whether to enable row merging

    Returns:
        Tuple of (transactions, warnings)

    Example:
        >>> transactions, warnings = build_transactions(table_def, rows)
        >>> print(f"Extracted {len(transactions)} transactions")
    """
    builder = RowBuilder(table_definition, enable_merge=enable_merge)
    transactions = builder.build_transactions(extracted_rows)
    warnings = builder.get_warnings()
    return transactions, warnings
