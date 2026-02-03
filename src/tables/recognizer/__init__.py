"""
Table Structure Recognition (Module 3).

This package provides table structure recognition functionality for
extracting structured transaction data from detected tables.

Main Components:
    - TableRecognizer: Main orchestrator for the recognition pipeline
    - CellExtractor: Extracts cells using column boundaries
    - RowBuilder: Constructs and merges transaction rows
    - TableMerger: Handles cross-page table merging

Key Features:
    - Cell extraction using column x-boundaries (handles text over borders)
    - Row merging for wrapped transactions
    - Cross-page table handling with semantic column mapping
    - Robust parsing for dates and amounts (including Indian formats)

Pipeline Flow:
    TableDefinition (from Module 2)
        → CellExtractor (extract cells by column)
        → TableMerger (handle multi-page)
        → RowBuilder (merge wrapped rows, parse values)
        → TransactionTable (structured output)

Example Usage:
    >>> from tables.reader import PDFDocument
    >>> from tables.detector import detect_tables
    >>> from tables.recognizer import recognize_tables
    >>>
    >>> pdf = PDFDocument.open("statement.pdf")
    >>> detection = detect_tables(pdf)
    >>> extraction = recognize_tables(pdf, detection)
    >>>
    >>> for table in extraction.tables:
    ...     print(f"Table {table.table_id}: {table.row_count} transactions")
    ...     for row in table.rows:
    ...         print(f"  {row.transaction_date}: {row.description} - {row.amount}")
"""

from tables.recognizer.cell_extractor import (
    CellExtractor,
    Cell,
    ExtractedRow,
    extract_table_rows,
    get_cell_text_by_column,
)
from tables.recognizer.row_builder import (
    RowBuilder,
    MergeContext,
    build_transactions,
)
from tables.recognizer.table_merger import (
    TableMerger,
    SemanticColumnMapper,
    PageBoundaryInfo,
    merge_table_pages,
)
from tables.recognizer.table_recognizer import (
    TableRecognizer,
    recognize_tables,
    extract_transactions,
)

__all__ = [
    # Main recognizer
    "TableRecognizer",
    "recognize_tables",
    "extract_transactions",
    # Cell extraction
    "CellExtractor",
    "Cell",
    "ExtractedRow",
    "extract_table_rows",
    "get_cell_text_by_column",
    # Row building
    "RowBuilder",
    "MergeContext",
    "build_transactions",
    # Table merging
    "TableMerger",
    "SemanticColumnMapper",
    "PageBoundaryInfo",
    "merge_table_pages",
]
