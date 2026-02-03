"""
Table Detection Module.

This module provides functionality for detecting and classifying tables
within PDF bank statements. It handles:

- Table presence detection on each page
- Table structure classification (bordered, semi-bordered, unbordered)
- Header detection using keyword matching with SLM fallback
- Table content classification (transaction, summary, account info)
- Cross-page table continuity analysis

The module is designed for deterministic processing with ML fallback
only when keyword matching confidence is below threshold.

Main Components:
    - TableDetector: Main orchestration class for table detection
    - HeaderDetector: Header identification using keywords and embeddings
    - StructureClassifier: Table type classification from visual elements
    - HEADER_KEYWORDS: Comprehensive keyword definitions for header matching

Example Usage:
    >>> from tables.detector import TableDetector
    >>> from tables.reader import PDFDocument
    >>>
    >>> pdf = PDFDocument.open("statement.pdf")
    >>> detector = TableDetector()
    >>> tables = detector.detect_tables(pdf)
    >>> for table in tables:
    ...     print(f"Table on pages {table.page_numbers}: {table.content_type}")
"""

from tables.detector.keywords import (
    HEADER_KEYWORDS,
    SEMANTIC_TYPES,
    get_keywords_for_type,
    normalize_header_text,
)
from tables.detector.structure_classifier import (
    StructureClassifier,
    StructureType,
)
from tables.detector.header_detector import (
    HeaderDetector,
    HeaderMatch,
)
from tables.detector.table_detector import (
    TableDetector,
    detect_tables,
)
from tables.detector.watermark_filter import (
    WatermarkFilter,
    filter_watermarks,
    is_likely_watermark,
)

__all__ = [
    # Keywords
    "HEADER_KEYWORDS",
    "SEMANTIC_TYPES",
    "get_keywords_for_type",
    "normalize_header_text",
    # Structure classification
    "StructureClassifier",
    "StructureType",
    # Header detection
    "HeaderDetector",
    "HeaderMatch",
    # Watermark filtering
    "WatermarkFilter",
    "filter_watermarks",
    "is_likely_watermark",
    # Main detection
    "TableDetector",
    "detect_tables",
]
