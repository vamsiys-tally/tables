"""
Table Structure Recognition - Main Orchestrator.

This module provides the main TableRecognizer class that orchestrates
the complete table structure recognition pipeline:

1. Cell extraction from detected tables
2. Row construction and merging
3. Cross-page table handling
4. Value parsing and validation
5. Transaction table output

This is the primary entry point for Module 3 functionality.

Example Usage:
    >>> from tables.recognizer import TableRecognizer
    >>> from tables.detector import detect_tables
    >>>
    >>> detection_result = detect_tables(pdf_document)
    >>> recognizer = TableRecognizer()
    >>> extraction_result = recognizer.recognize(pdf_document, detection_result)
    >>>
    >>> for table in extraction_result.tables:
    ...     for row in table.rows:
    ...         print(f"{row.transaction_date}: {row.description}")
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional
import logging
import time

from tables.models.table import TableDefinition, TableDetectionResult, ContentType
from tables.models.transaction import (
    TransactionTable,
    TransactionRow,
    ColumnSchema,
    ExtractionResult,
    ParseWarning,
)
from tables.recognizer.cell_extractor import CellExtractor, ExtractedRow
from tables.recognizer.row_builder import RowBuilder
from tables.recognizer.table_merger import TableMerger

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

# Minimum rows required for a valid transaction table
MIN_TRANSACTION_ROWS = 1

# Confidence threshold for acceptable extraction
MIN_CONFIDENCE_THRESHOLD = 0.5


# =============================================================================
# Table Recognizer
# =============================================================================

class TableRecognizer:
    """
    Main orchestrator for table structure recognition.

    Processes TableDefinitions from Module 2 (detection) and produces
    TransactionTables with parsed data.

    Pipeline:
        1. For each transaction table:
           a. Extract cells using column boundaries
           b. Merge rows from multiple pages if needed
           c. Apply row merging for wrapped transactions
           d. Parse values (dates, amounts)
           e. Build TransactionTable output
        2. Aggregate results into ExtractionResult

    Attributes:
        enable_row_merge: Whether to merge wrapped transaction rows
        min_rows: Minimum rows for a valid table
        cell_extractor: Cell extraction component
        table_merger: Cross-page merging component

    Example:
        >>> recognizer = TableRecognizer()
        >>> result = recognizer.recognize(pdf_document, detection_result)
        >>> print(f"Extracted {result.total_transactions} transactions")
    """

    def __init__(
        self,
        enable_row_merge: bool = True,
        min_rows: int = MIN_TRANSACTION_ROWS,
    ):
        """
        Initialize the table recognizer.

        Args:
            enable_row_merge: Enable merging of wrapped transaction rows
            min_rows: Minimum rows to consider a valid table
        """
        self.enable_row_merge = enable_row_merge
        self.min_rows = min_rows
        self.cell_extractor = CellExtractor()
        self.table_merger = TableMerger()

    def recognize(
        self,
        pdf_document: Any,
        detection_result: TableDetectionResult,
        table_ids: Optional[list[str]] = None,
    ) -> ExtractionResult:
        """
        Recognize and extract transaction data from detected tables.

        Args:
            pdf_document: PDFDocument instance
            detection_result: TableDetectionResult from Module 2
            table_ids: Optional list of specific table IDs to process
                      (processes all transaction tables if not specified)

        Returns:
            ExtractionResult containing extracted TransactionTables

        Example:
            >>> result = recognizer.recognize(pdf, detection_result)
            >>> for table in result.tables:
            ...     print(f"Table {table.table_id}: {table.row_count} rows")
        """
        start_time = time.time()

        result = ExtractionResult(
            pages_processed=detection_result.pages_processed.copy(),
        )

        # Filter tables to process
        tables_to_process = self._filter_tables(
            detection_result.tables,
            table_ids,
        )

        if not tables_to_process:
            logger.info("No transaction tables to process")
            return result

        # Process each table
        for table_def in tables_to_process:
            try:
                txn_table = self._process_table(pdf_document, table_def)
                if txn_table is not None:
                    result.add_table(txn_table)
                    result.source_table_ids.append(table_def.table_id)
            except Exception as e:
                logger.error(
                    f"Failed to process table {table_def.table_id}: {e}"
                )
                result.add_error(
                    f"Table {table_def.table_id}: {str(e)}"
                )

        # Calculate overall confidence
        if result.tables:
            total_confidence = sum(t.confidence_score for t in result.tables)
            result.overall_confidence = total_confidence / len(result.tables)

        # Record processing time
        result.processing_time_ms = int((time.time() - start_time) * 1000)

        logger.info(
            f"Extracted {result.total_transactions} transactions from "
            f"{result.table_count} tables in {result.processing_time_ms}ms"
        )

        return result

    def recognize_table(
        self,
        pdf_document: Any,
        table_definition: TableDefinition,
    ) -> Optional[TransactionTable]:
        """
        Recognize a single table.

        Convenience method for processing one table at a time.

        Args:
            pdf_document: PDFDocument instance
            table_definition: Single table definition

        Returns:
            TransactionTable or None if extraction failed
        """
        return self._process_table(pdf_document, table_definition)

    def _filter_tables(
        self,
        tables: list[TableDefinition],
        table_ids: Optional[list[str]],
    ) -> list[TableDefinition]:
        """
        Filter tables to process.

        Args:
            tables: All detected tables
            table_ids: Optional specific IDs to process

        Returns:
            Filtered list of tables
        """
        # Filter by content type (only transaction tables)
        transaction_tables = [
            t for t in tables
            if t.content_type == ContentType.TRANSACTION
        ]

        # Filter by specific IDs if provided
        if table_ids:
            transaction_tables = [
                t for t in transaction_tables
                if t.table_id in table_ids
            ]

        return transaction_tables

    def _process_table(
        self,
        pdf_document: Any,
        table_definition: TableDefinition,
    ) -> Optional[TransactionTable]:
        """
        Process a single table definition into a TransactionTable.

        Args:
            pdf_document: PDFDocument instance
            table_definition: Table to process

        Returns:
            TransactionTable or None if too few rows
        """
        logger.debug(
            f"Processing table {table_definition.table_id} "
            f"(pages {table_definition.page_numbers})"
        )

        # Extract rows (handles multi-page)
        if table_definition.is_multi_page:
            extracted_rows = self.table_merger.merge_multi_page_table(
                pdf_document,
                table_definition,
            )
        else:
            # Single page table
            page_number = table_definition.page_numbers[0]
            extracted_rows = self.cell_extractor.extract_rows(
                pdf_document,
                page_number,
                table_definition,
                skip_header_rows=True,
            )

        if not extracted_rows:
            logger.debug(f"No data rows found in table {table_definition.table_id}")
            return None

        # Build transaction rows
        row_builder = RowBuilder(
            table_definition,
            enable_merge=self.enable_row_merge,
        )
        transaction_rows = row_builder.build_transactions(extracted_rows)
        warnings = row_builder.get_warnings()
        column_schemas = row_builder.build_column_schemas()

        # Check minimum rows
        if len(transaction_rows) < self.min_rows:
            logger.debug(
                f"Table {table_definition.table_id} has only "
                f"{len(transaction_rows)} rows (min: {self.min_rows})"
            )
            return None

        # Build transaction table
        txn_table = TransactionTable(
            table_id=table_definition.table_id,
            source_pages=table_definition.page_numbers,
            columns=column_schemas,
            rows=transaction_rows,
            parse_warnings=warnings,
        )

        # Calculate confidence
        txn_table.confidence_score = self._calculate_table_confidence(
            txn_table,
            table_definition,
        )

        logger.debug(
            f"Extracted {txn_table.row_count} transactions from "
            f"table {table_definition.table_id} "
            f"(confidence: {txn_table.confidence_score:.2f})"
        )

        return txn_table

    def _calculate_table_confidence(
        self,
        txn_table: TransactionTable,
        table_definition: TableDefinition,
    ) -> float:
        """
        Calculate overall confidence score for a table.

        Args:
            txn_table: Extracted transaction table
            table_definition: Source table definition

        Returns:
            Confidence score (0.0 - 1.0)
        """
        if not txn_table.rows:
            return 0.0

        # Start with detection confidence
        confidence = table_definition.detection_confidence

        # Average of row confidences
        row_confidences = [r.confidence for r in txn_table.rows]
        avg_row_confidence = sum(row_confidences) / len(row_confidences)
        confidence = (confidence + avg_row_confidence) / 2

        # Deduct for warnings
        warning_penalty = min(0.3, len(txn_table.parse_warnings) * 0.02)
        confidence -= warning_penalty

        # Bonus for having date range
        if txn_table.date_range:
            confidence += 0.05

        # Bonus for having totals
        if txn_table.total_debit or txn_table.total_credit:
            confidence += 0.05

        return max(0.0, min(1.0, confidence))


# =============================================================================
# Convenience Functions
# =============================================================================

def recognize_tables(
    pdf_document: Any,
    detection_result: TableDetectionResult,
    enable_row_merge: bool = True,
) -> ExtractionResult:
    """
    Convenience function to recognize tables from detection result.

    Args:
        pdf_document: PDFDocument instance
        detection_result: TableDetectionResult from detect_tables()
        enable_row_merge: Enable merging of wrapped transaction rows

    Returns:
        ExtractionResult with extracted transactions

    Example:
        >>> from tables.detector import detect_tables
        >>> from tables.recognizer import recognize_tables
        >>>
        >>> detection = detect_tables(pdf)
        >>> extraction = recognize_tables(pdf, detection)
        >>> print(f"Found {extraction.total_transactions} transactions")
    """
    recognizer = TableRecognizer(enable_row_merge=enable_row_merge)
    return recognizer.recognize(pdf_document, detection_result)


def extract_transactions(
    pdf_document: Any,
    table_definition: TableDefinition,
    enable_row_merge: bool = True,
) -> Optional[TransactionTable]:
    """
    Convenience function to extract transactions from a single table.

    Args:
        pdf_document: PDFDocument instance
        table_definition: Table definition to process
        enable_row_merge: Enable merging of wrapped transaction rows

    Returns:
        TransactionTable or None if extraction failed

    Example:
        >>> table = extract_transactions(pdf, table_def)
        >>> if table:
        ...     for row in table.rows:
        ...         print(f"{row.transaction_date}: {row.amount}")
    """
    recognizer = TableRecognizer(enable_row_merge=enable_row_merge)
    return recognizer.recognize_table(pdf_document, table_definition)
