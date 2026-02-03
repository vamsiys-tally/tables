"""
Table Detection Orchestration.

This module provides the main TableDetector class that orchestrates
the complete table detection pipeline for PDF bank statements.

Detection Pipeline:
    1. For each page in the document:
       a. Extract visual elements (lines, rectangles, text)
       b. Detect table regions using spatial analysis
       c. Classify table structure (bordered, unbordered, etc.)
       d. Detect and classify headers
       e. Determine column boundaries
    2. Analyze cross-page table continuity
    3. Classify table content type (transaction, summary, etc.)
    4. Return TableDefinition objects

Design Principles:
    - Pure structural detection (no content-based validation)
    - Deterministic methods preferred, ML as fallback
    - Comprehensive metadata for downstream processing

Example Usage:
    >>> from tables.detector import TableDetector
    >>> from tables.reader import PDFDocument
    >>>
    >>> pdf = PDFDocument.open("statement.pdf")
    >>> detector = TableDetector()
    >>> result = detector.detect(pdf)
    >>>
    >>> for table in result.tables:
    ...     print(f"Table on pages {table.page_numbers}")
    ...     print(f"  Structure: {table.structure_type.value}")
    ...     print(f"  Columns: {table.column_count}")
"""

from __future__ import annotations

import time
import logging
from dataclasses import dataclass, field
from typing import Optional, Any

from tables.utils.geometry import (
    BoundingBox,
    cluster_by_y_coordinate,
    cluster_by_x_coordinate,
    find_gaps_in_range,
)
from tables.models.table import (
    TableDefinition,
    TableDetectionResult,
    ColumnDefinition,
    StructureType,
    ContentType,
    DataType,
)
from tables.detector.structure_classifier import (
    StructureClassifier,
    StructureAnalysis,
)
from tables.detector.header_detector import (
    HeaderDetector,
    HeaderRow,
)
from tables.detector.keywords import (
    HEADER_KEYWORDS,
    normalize_header_text,
)
from tables.detector.watermark_filter import (
    WatermarkFilter,
    filter_watermarks,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

# Minimum table height (in PDF points)
MIN_TABLE_HEIGHT = 50.0

# Minimum table width (in PDF points)
MIN_TABLE_WIDTH = 100.0

# Minimum rows for a valid table
MIN_TABLE_ROWS = 2

# Maximum gap between table rows before splitting
MAX_ROW_GAP = 30.0

# Minimum column width
MIN_COLUMN_WIDTH = 20.0

# Content type detection keywords
TRANSACTION_KEYWORDS = {"date", "debit", "credit", "balance", "amount", "description"}
SUMMARY_KEYWORDS = {"total", "opening", "closing", "summary", "period"}
ACCOUNT_INFO_KEYWORDS = {"account", "holder", "branch", "ifsc", "customer"}


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class TextBlock:
    """
    A block of text with its bounding box.

    Represents a text element extracted from the PDF with
    its position for spatial analysis.

    Attributes:
        text: The text content
        bbox: Bounding box of the text
        page: Page number where this text appears
    """

    text: str
    bbox: BoundingBox
    page: int = 0

    @property
    def center_y(self) -> float:
        """Get vertical center of text."""
        return (self.bbox.y0 + self.bbox.y1) / 2

    @property
    def center_x(self) -> float:
        """Get horizontal center of text."""
        return (self.bbox.x0 + self.bbox.x1) / 2


@dataclass
class TableRegion:
    """
    A detected table region on a page.

    Contains the bounding box and extracted elements
    before full table analysis.

    Attributes:
        bounds: Bounding box of the table region
        page: Page number
        text_blocks: Text blocks within this region
        rows: Text blocks grouped by row
        structure: Structure analysis result
    """

    bounds: BoundingBox
    page: int
    text_blocks: list[TextBlock] = field(default_factory=list)
    rows: list[list[TextBlock]] = field(default_factory=list)
    structure: Optional[StructureAnalysis] = None


# =============================================================================
# Table Detector
# =============================================================================

class TableDetector:
    """
    Main class for detecting tables in PDF documents.

    Orchestrates the complete table detection pipeline including
    region detection, structure classification, header detection,
    and cross-page analysis.

    Attributes:
        structure_classifier: Classifier for table visual structure
        header_detector: Detector for table headers
        watermark_filter: Filter for removing watermark text
        use_slm: Whether to enable SLM-based header detection
        filter_watermarks: Whether to filter watermark text
        min_table_rows: Minimum rows for a valid table

    Example:
        >>> detector = TableDetector(use_slm=True)
        >>> result = detector.detect(pdf_document)
        >>> print(f"Found {result.table_count} tables")
    """

    def __init__(
        self,
        use_slm: bool = True,
        min_table_rows: int = MIN_TABLE_ROWS,
        filter_watermarks: bool = True,
    ):
        """
        Initialize the table detector.

        Args:
            use_slm: Enable SLM-based header detection fallback
            min_table_rows: Minimum rows to consider a valid table
            filter_watermarks: Enable watermark text filtering
        """
        self.structure_classifier = StructureClassifier()
        self.header_detector = HeaderDetector(use_slm=use_slm)
        self.watermark_filter = WatermarkFilter() if filter_watermarks else None
        self.use_slm = use_slm
        self.filter_watermarks = filter_watermarks
        self.min_table_rows = min_table_rows

    def detect(
        self,
        pdf_document: Any,  # PDFDocument, using Any to avoid circular import
        pages: Optional[list[int]] = None,
    ) -> TableDetectionResult:
        """
        Detect all tables in a PDF document.

        Main entry point for table detection. Processes each page
        and returns all detected tables with their definitions.

        Args:
            pdf_document: PDFDocument instance to analyze
            pages: Optional list of page numbers to process (0-indexed)
                   Defaults to all pages

        Returns:
            TableDetectionResult with all detected tables

        Example:
            >>> result = detector.detect(pdf)
            >>> for table in result.transaction_tables:
            ...     print(f"Transaction table on pages {table.page_numbers}")
        """
        start_time = time.time()
        result = TableDetectionResult()

        # Determine which pages to process
        if pages is None:
            pages = list(range(pdf_document.page_count))

        result.pages_processed = pages

        # Process each page
        page_tables: dict[int, list[TableDefinition]] = {}

        for page_num in pages:
            try:
                tables = self._detect_tables_on_page(pdf_document, page_num)
                if tables:
                    page_tables[page_num] = tables
                    result.pages_with_tables.append(page_num)
            except Exception as e:
                logger.warning(f"Failed to process page {page_num}: {e}")
                result.pages_skipped.append((page_num, str(e)))

        # Merge cross-page tables
        merged_tables = self._merge_cross_page_tables(page_tables, pdf_document)

        # Classify content types
        for table in merged_tables:
            self._classify_content_type(table, pdf_document)

        result.tables = merged_tables
        result.processing_time_ms = int((time.time() - start_time) * 1000)

        logger.info(
            f"Detected {result.table_count} tables across "
            f"{len(result.pages_with_tables)} pages in {result.processing_time_ms}ms"
        )

        return result

    def _detect_tables_on_page(
        self,
        pdf_document: Any,
        page_number: int,
    ) -> list[TableDefinition]:
        """
        Detect tables on a single page.

        Args:
            pdf_document: PDFDocument instance
            page_number: Page number to analyze (0-indexed)

        Returns:
            List of TableDefinition for tables on this page
        """
        # Extract text blocks with positions
        text_blocks = self._extract_text_blocks(pdf_document, page_number)

        if not text_blocks:
            logger.debug(f"No text found on page {page_number}")
            return []

        # Detect table regions using text clustering
        regions = self._detect_table_regions(text_blocks, page_number)

        if not regions:
            logger.debug(f"No table regions detected on page {page_number}")
            return []

        # Analyze each region
        tables: list[TableDefinition] = []

        for region in regions:
            # Classify structure
            region.structure = self.structure_classifier.classify_from_elements(
                lines=self._get_lines_in_region(pdf_document, page_number, region.bounds),
                rectangles=self._get_rects_in_region(pdf_document, page_number, region.bounds),
                region=region.bounds,
            )

            # Group text blocks into rows
            region.rows = self._group_into_rows(region.text_blocks)

            if len(region.rows) < self.min_table_rows:
                continue

            # Detect headers
            header_rows = self._detect_headers(region)

            if not header_rows:
                logger.debug(f"No headers detected in region on page {page_number}")
                continue

            # Detect columns
            columns = self._detect_columns(region, header_rows)

            if not columns:
                logger.debug(f"Could not detect columns in region on page {page_number}")
                continue

            # Create table definition
            table = TableDefinition(
                page_numbers=[page_number],
                columns=columns,
                structure_type=region.structure.structure_type,
                bounds_per_page={page_number: region.bounds},
                header_row_indices=[hr.row_index for hr in header_rows],
                detection_confidence=region.structure.confidence,
            )

            tables.append(table)

        return tables

    def _extract_text_blocks(
        self,
        pdf_document: Any,
        page_number: int,
    ) -> list[TextBlock]:
        """
        Extract text blocks with positions from a page.

        Optionally filters watermark text to avoid interference
        with table detection.

        Args:
            pdf_document: PDFDocument instance
            page_number: Page number

        Returns:
            List of TextBlock objects
        """
        blocks: list[TextBlock] = []

        try:
            # Get words with positions from pdfplumber
            words = pdf_document.get_page_words(page_number)

            # Apply watermark filtering if enabled
            if self.watermark_filter and words:
                result = self.watermark_filter.filter_words(words)
                if result.removal_count > 0:
                    logger.debug(
                        f"Page {page_number}: Filtered {result.removal_count} "
                        f"watermark words"
                    )
                words = result.filtered_words

            for word in words:
                text = word.get("text", "").strip()
                if not text:
                    continue

                bbox = BoundingBox(
                    x0=word.get("x0", 0),
                    y0=word.get("top", 0),
                    x1=word.get("x1", 0),
                    y1=word.get("bottom", 0),
                )

                blocks.append(TextBlock(text=text, bbox=bbox, page=page_number))

        except Exception as e:
            logger.warning(f"Failed to extract text from page {page_number}: {e}")

        return blocks

    def _detect_table_regions(
        self,
        text_blocks: list[TextBlock],
        page_number: int,
    ) -> list[TableRegion]:
        """
        Detect potential table regions using header-first detection.

        Strategy:
            1. Group all text blocks into rows
            2. Sort rows by y-coordinate (top to bottom)
            3. Find rows that look like table headers (using keyword matching)
            4. For each header row, build a table region from header down
            5. Include all subsequent rows until a clear non-table row is found

        Args:
            text_blocks: Text blocks on the page
            page_number: Page number

        Returns:
            List of TableRegion objects
        """
        if not text_blocks:
            return []

        # Group text blocks by y-coordinate (rows)
        boxes = [block.bbox for block in text_blocks]
        row_clusters = cluster_by_y_coordinate(boxes, tolerance=5.0)

        if len(row_clusters) < self.min_table_rows:
            return []

        # Sort rows by y-coordinate (top to bottom)
        row_clusters_sorted = sorted(row_clusters, key=lambda c: min(b.y0 for b in c))

        # Build text rows (text blocks per row) - sorted by y
        text_rows: list[list[TextBlock]] = []
        for cluster in row_clusters_sorted:
            row_blocks: list[TextBlock] = []
            for block in text_blocks:
                if any(block.bbox.is_horizontally_aligned(box, tolerance=5.0) for box in cluster):
                    if block not in row_blocks:
                        row_blocks.append(block)
            row_blocks.sort(key=lambda b: b.bbox.x0)
            text_rows.append(row_blocks)

        # Find header rows
        regions: list[TableRegion] = []
        used_rows: set[int] = set()

        for row_idx, text_row in enumerate(text_rows):
            if row_idx in used_rows:
                continue

            # Check if this row is a potential header row
            row_texts = [block.text for block in text_row]

            # Headers typically have 3-15 columns
            # Less than 3 = probably not a table header
            # More than 15 = probably a paragraph (each word is a "column")
            if len(row_texts) < 3 or len(row_texts) > 15:
                continue

            from tables.detector.keywords import is_likely_header_row
            if not is_likely_header_row(row_texts, min_matches=2):
                continue

            # Found a header row! Build table region from here
            logger.debug(f"Found potential header row at index {row_idx}: {row_texts}")

            # Collect rows for this table
            table_row_clusters: list[list[BoundingBox]] = [row_clusters_sorted[row_idx]]
            table_text_blocks: list[TextBlock] = list(text_row)
            used_rows.add(row_idx)

            # Get expected column count from header
            expected_cols = len(text_row)
            consecutive_non_table_rows = 0
            max_consecutive_non_table = 2  # Allow up to 2 non-table rows before stopping

            # Include subsequent rows that look like data rows
            for data_idx in range(row_idx + 1, len(text_rows)):
                if data_idx in used_rows:
                    continue

                data_row = text_rows[data_idx]
                data_cluster = row_clusters_sorted[data_idx]

                # Check if this row could be a data row
                col_count = len(data_row)

                # Data row criteria:
                # 1. Has similar column count (within range of header ±3)
                # 2. OR has at least 2 columns
                is_data_like = (
                    abs(col_count - expected_cols) <= 3 or
                    (col_count >= 2 and col_count <= expected_cols + 2)
                )

                if is_data_like:
                    table_row_clusters.append(data_cluster)
                    for block in data_row:
                        if block not in table_text_blocks:
                            table_text_blocks.append(block)
                    used_rows.add(data_idx)
                    consecutive_non_table_rows = 0
                else:
                    consecutive_non_table_rows += 1
                    if consecutive_non_table_rows >= max_consecutive_non_table:
                        break

            # Create region if we have enough rows
            if len(table_row_clusters) >= self.min_table_rows:
                region = self._create_region_from_rows(
                    table_row_clusters,
                    table_text_blocks,
                    page_number,
                )
                if region:
                    regions.append(region)

        # If no header-based regions found, fall back to column-count method
        if not regions:
            regions = self._detect_regions_by_column_count(
                text_blocks, row_clusters_sorted, page_number
            )

        return regions

    def _detect_regions_by_column_count(
        self,
        text_blocks: list[TextBlock],
        row_clusters: list[list[BoundingBox]],
        page_number: int,
    ) -> list[TableRegion]:
        """
        Fallback method: detect regions by consistent column count.

        Used when header-based detection doesn't find any tables.

        Args:
            text_blocks: Text blocks on the page
            row_clusters: Rows sorted by y-coordinate
            page_number: Page number

        Returns:
            List of TableRegion objects
        """
        regions: list[TableRegion] = []
        current_region_rows: list[list[BoundingBox]] = []
        current_region_blocks: list[TextBlock] = []

        for i, cluster in enumerate(row_clusters):
            # Check if this row has multiple columns (table-like)
            if len(cluster) >= 2:  # At least 2 columns
                current_region_rows.append(cluster)
                # Find corresponding text blocks
                for block in text_blocks:
                    if any(block.bbox.is_horizontally_aligned(box) for box in cluster):
                        if block not in current_region_blocks:
                            current_region_blocks.append(block)
            else:
                # Single column row - might be end of table
                if len(current_region_rows) >= self.min_table_rows:
                    # Create region from accumulated rows
                    region = self._create_region_from_rows(
                        current_region_rows,
                        current_region_blocks,
                        page_number,
                    )
                    if region:
                        regions.append(region)

                current_region_rows = []
                current_region_blocks = []

        # Don't forget the last region
        if len(current_region_rows) >= self.min_table_rows:
            region = self._create_region_from_rows(
                current_region_rows,
                current_region_blocks,
                page_number,
            )
            if region:
                regions.append(region)

        return regions

    def _create_region_from_rows(
        self,
        row_clusters: list[list[BoundingBox]],
        text_blocks: list[TextBlock],
        page_number: int,
    ) -> Optional[TableRegion]:
        """
        Create a TableRegion from clustered rows.

        Args:
            row_clusters: Rows of bounding boxes
            text_blocks: Text blocks in this region
            page_number: Page number

        Returns:
            TableRegion or None if invalid
        """
        if not row_clusters:
            return None

        # Calculate bounding box for the region
        all_boxes = [box for row in row_clusters for box in row]
        bounds = BoundingBox.union(all_boxes)

        if bounds is None:
            return None

        # Filter text blocks to those within bounds
        region_blocks = [
            block for block in text_blocks
            if bounds.contains_box(block.bbox, tolerance=5.0)
        ]

        return TableRegion(
            bounds=bounds,
            page=page_number,
            text_blocks=region_blocks,
        )

    def _group_into_rows(
        self,
        text_blocks: list[TextBlock],
    ) -> list[list[TextBlock]]:
        """
        Group text blocks into rows based on y-coordinate.

        Args:
            text_blocks: Text blocks to group

        Returns:
            List of rows, each row is a list of TextBlock
        """
        if not text_blocks:
            return []

        boxes = [block.bbox for block in text_blocks]
        clusters = cluster_by_y_coordinate(boxes, tolerance=5.0)

        rows: list[list[TextBlock]] = []
        for cluster in clusters:
            row_blocks: list[TextBlock] = []
            for block in text_blocks:
                if any(block.bbox.is_horizontally_aligned(box, tolerance=5.0) for box in cluster):
                    if block not in row_blocks:
                        row_blocks.append(block)

            # Sort by x-coordinate (left to right)
            row_blocks.sort(key=lambda b: b.bbox.x0)
            rows.append(row_blocks)

        # Sort rows by y-coordinate (top to bottom) so header rows come first
        rows.sort(key=lambda row: min(b.bbox.y0 for b in row) if row else 0)

        return rows

    def _detect_headers(
        self,
        region: TableRegion,
    ) -> list[HeaderRow]:
        """
        Detect header rows in a table region.

        Args:
            region: Table region to analyze

        Returns:
            List of detected HeaderRow objects
        """
        if not region.rows:
            return []

        # Extract text from first few rows
        candidate_rows: list[list[str]] = []
        for row in region.rows[:3]:  # Check first 3 rows
            texts = [block.text for block in row]
            candidate_rows.append(texts)

        return self.header_detector.find_header_rows(candidate_rows)

    def _detect_columns(
        self,
        region: TableRegion,
        header_rows: list[HeaderRow],
    ) -> list[ColumnDefinition]:
        """
        Detect column boundaries and create column definitions.

        Uses header information and text alignment to determine
        column boundaries. For unbordered tables, uses whitespace
        gap detection for more accurate column separation.

        Args:
            region: Table region
            header_rows: Detected header rows

        Returns:
            List of ColumnDefinition objects
        """
        if not header_rows or not region.rows:
            return []

        # Use the first header row for column structure
        primary_header = header_rows[0]

        # Get column positions from header row
        if primary_header.row_index < len(region.rows):
            header_blocks = region.rows[primary_header.row_index]
        else:
            return []

        # For unbordered tables, use whitespace gap detection
        is_unbordered = (
            region.structure and
            region.structure.structure_type == StructureType.UNBORDERED
        )

        if is_unbordered:
            return self._detect_columns_by_gaps(
                region, header_rows, header_blocks, primary_header
            )

        # Standard approach for bordered/semi-bordered tables
        return self._detect_columns_by_alignment(
            region, header_blocks, primary_header
        )

    def _detect_columns_by_alignment(
        self,
        region: TableRegion,
        header_blocks: list[TextBlock],
        primary_header: HeaderRow,
    ) -> list[ColumnDefinition]:
        """
        Detect columns using text block alignment.

        Standard approach for bordered and semi-bordered tables
        where text alignment is reliable.

        Args:
            region: Table region
            header_blocks: Text blocks from header row
            primary_header: Header row detection result

        Returns:
            List of ColumnDefinition objects
        """
        columns: list[ColumnDefinition] = []

        for i, (block, match) in enumerate(zip(header_blocks, primary_header.matches)):
            # Determine column boundaries
            # Use the text block position, with some expansion
            x0 = block.bbox.x0 - 5
            x1 = block.bbox.x1 + 5

            # Expand to cover data in this column
            # Look at cells below this header
            for row in region.rows[primary_header.row_index + 1:]:
                for data_block in row:
                    # Check if this block belongs to this column
                    if data_block.bbox.is_vertically_aligned(block.bbox, tolerance=20):
                        x0 = min(x0, data_block.bbox.x0 - 2)
                        x1 = max(x1, data_block.bbox.x1 + 2)

            column = self._create_column_definition(
                i, x0, x1, match, primary_header.row_index
            )
            columns.append(column)

        return columns

    def _detect_columns_by_gaps(
        self,
        region: TableRegion,
        header_rows: list[HeaderRow],
        header_blocks: list[TextBlock],
        primary_header: HeaderRow,
    ) -> list[ColumnDefinition]:
        """
        Detect columns using whitespace gap detection.

        For unbordered tables where text alignment alone may not
        be sufficient. Uses gaps between text blocks to identify
        column boundaries.

        Args:
            region: Table region
            header_rows: All detected header rows
            header_blocks: Text blocks from header row
            primary_header: Primary header row result

        Returns:
            List of ColumnDefinition objects
        """
        # Collect all x-coordinates from all text blocks (excluding headers)
        data_rows = region.rows[primary_header.row_index + 1:]
        if not data_rows:
            # Fall back to alignment-based detection
            return self._detect_columns_by_alignment(
                region, header_blocks, primary_header
            )

        # Get all text block boundaries
        all_x_positions: list[tuple[float, float]] = []  # (x0, x1) for each block

        for row in data_rows:
            for block in row:
                all_x_positions.append((block.bbox.x0, block.bbox.x1))

        # Add header positions too
        for block in header_blocks:
            all_x_positions.append((block.bbox.x0, block.bbox.x1))

        if not all_x_positions:
            return self._detect_columns_by_alignment(
                region, header_blocks, primary_header
            )

        # Find column boundaries using gap detection
        column_boundaries = self._find_column_boundaries_from_gaps(
            all_x_positions, region.bounds
        )

        if len(column_boundaries) < len(header_blocks):
            # If gap detection doesn't match headers, fall back
            return self._detect_columns_by_alignment(
                region, header_blocks, primary_header
            )

        # Map headers to column boundaries
        columns: list[ColumnDefinition] = []

        for i, (block, match) in enumerate(zip(header_blocks, primary_header.matches)):
            # Find which column boundary this header belongs to
            header_center = block.center_x
            best_col_idx = 0
            best_distance = float('inf')

            for col_idx, (x0, x1) in enumerate(column_boundaries):
                col_center = (x0 + x1) / 2
                distance = abs(header_center - col_center)
                if distance < best_distance:
                    best_distance = distance
                    best_col_idx = col_idx

            if best_col_idx < len(column_boundaries):
                x0, x1 = column_boundaries[best_col_idx]
            else:
                x0, x1 = block.bbox.x0 - 5, block.bbox.x1 + 5

            column = self._create_column_definition(
                i, x0, x1, match, primary_header.row_index
            )
            columns.append(column)

        return columns

    def _find_column_boundaries_from_gaps(
        self,
        x_positions: list[tuple[float, float]],
        region_bounds: BoundingBox,
    ) -> list[tuple[float, float]]:
        """
        Find column boundaries by detecting whitespace gaps.

        Analyzes the x-coordinate distribution of text blocks
        to find gaps that indicate column separations.

        Args:
            x_positions: List of (x0, x1) tuples for all text blocks
            region_bounds: Bounds of the table region

        Returns:
            List of (x0, x1) tuples for each detected column
        """
        if not x_positions:
            return []

        # Sort by x0
        sorted_positions = sorted(x_positions, key=lambda p: p[0])

        # Find gaps between text blocks
        # A gap is where the right edge of one block is far from the left edge of the next
        min_gap_size = MIN_COLUMN_WIDTH * 0.5  # At least half column width gap

        # Build a list of all "coverage" ranges
        coverage_ranges: list[tuple[float, float]] = []
        current_start = sorted_positions[0][0]
        current_end = sorted_positions[0][1]

        for x0, x1 in sorted_positions[1:]:
            if x0 <= current_end + min_gap_size:
                # Overlaps or close - extend current range
                current_end = max(current_end, x1)
            else:
                # Gap found - save current range and start new
                coverage_ranges.append((current_start, current_end))
                current_start = x0
                current_end = x1

        # Don't forget the last range
        coverage_ranges.append((current_start, current_end))

        # Convert coverage ranges to column boundaries
        # Expand slightly for tolerance
        column_boundaries: list[tuple[float, float]] = []
        for i, (start, end) in enumerate(coverage_ranges):
            # Use midpoint between columns as boundary
            x0 = start - 2
            if i > 0:
                prev_end = coverage_ranges[i - 1][1]
                x0 = (prev_end + start) / 2

            x1 = end + 2
            if i < len(coverage_ranges) - 1:
                next_start = coverage_ranges[i + 1][0]
                x1 = (end + next_start) / 2

            column_boundaries.append((x0, x1))

        return column_boundaries

    def _create_column_definition(
        self,
        column_id: int,
        x0: float,
        x1: float,
        match: Any,  # HeaderMatch
        header_row_index: int,
    ) -> ColumnDefinition:
        """
        Create a ColumnDefinition from header match and boundaries.

        Args:
            column_id: Column index
            x0: Left boundary
            x1: Right boundary
            match: HeaderMatch object
            header_row_index: Source header row index

        Returns:
            ColumnDefinition object
        """
        # Determine data type from semantic type
        data_type = DataType.UNKNOWN
        if match.semantic_type == "date":
            data_type = DataType.DATE
        elif match.semantic_type in ("debit", "credit", "balance", "amount"):
            data_type = DataType.NUMERIC
        elif match.semantic_type in ("description", "reference"):
            data_type = DataType.TEXT

        return ColumnDefinition(
            column_id=column_id,
            x0=x0,
            x1=x1,
            header_text=match.text,
            semantic_type=match.semantic_type,
            data_type=data_type,
            confidence=match.confidence,
            source_header_row=header_row_index,
        )

    def _get_lines_in_region(
        self,
        pdf_document: Any,
        page_number: int,
        region: BoundingBox,
    ) -> list[Any]:
        """
        Get lines within a region.

        Args:
            pdf_document: PDFDocument instance
            page_number: Page number
            region: Region to filter

        Returns:
            List of Line objects within the region
        """
        from tables.utils.geometry import Line

        # Use get_page_lines_raw to get dict format
        all_lines = pdf_document.get_page_lines_raw(page_number)
        result: list[Line] = []

        for line_dict in all_lines:
            line = Line(
                x0=line_dict.get("x0", 0),
                y0=line_dict.get("y0", line_dict.get("top", 0)),
                x1=line_dict.get("x1", 0),
                y1=line_dict.get("y1", line_dict.get("bottom", 0)),
            )

            line_box = BoundingBox(
                x0=min(line.x0, line.x1),
                y0=min(line.y0, line.y1),
                x1=max(line.x0, line.x1),
                y1=max(line.y0, line.y1),
            )

            if region.overlaps(line_box):
                result.append(line)

        return result

    def _get_rects_in_region(
        self,
        pdf_document: Any,
        page_number: int,
        region: BoundingBox,
    ) -> list[BoundingBox]:
        """
        Get rectangles within a region.

        Args:
            pdf_document: PDFDocument instance
            page_number: Page number
            region: Region to filter

        Returns:
            List of BoundingBox objects within the region
        """
        all_rects = pdf_document.get_page_rects(page_number)
        result: list[BoundingBox] = []

        for rect_dict in all_rects:
            rect = BoundingBox.from_dict(rect_dict)
            if region.overlaps(rect):
                result.append(rect)

        return result

    def _merge_cross_page_tables(
        self,
        page_tables: dict[int, list[TableDefinition]],
        pdf_document: Any,
    ) -> list[TableDefinition]:
        """
        Merge tables that span multiple pages.

        Identifies tables that continue across page boundaries
        and merges them into single TableDefinition objects.

        Args:
            page_tables: Tables detected per page
            pdf_document: PDFDocument instance

        Returns:
            List of merged TableDefinition objects
        """
        if not page_tables:
            return []

        all_tables: list[TableDefinition] = []
        processed_pages: set[int] = set()

        # Sort pages
        sorted_pages = sorted(page_tables.keys())

        for page_num in sorted_pages:
            if page_num in processed_pages:
                continue

            tables_on_page = page_tables[page_num]

            for table in tables_on_page:
                # Check if this table continues to the next page
                merged_table = self._try_merge_with_next_pages(
                    table,
                    page_num,
                    page_tables,
                    processed_pages,
                )
                all_tables.append(merged_table)

            processed_pages.add(page_num)

        return all_tables

    def _try_merge_with_next_pages(
        self,
        table: TableDefinition,
        start_page: int,
        page_tables: dict[int, list[TableDefinition]],
        processed_pages: set[int],
    ) -> TableDefinition:
        """
        Try to merge a table with continuation tables on subsequent pages.

        Args:
            table: Starting table
            start_page: Starting page number
            page_tables: All detected tables by page
            processed_pages: Set of already processed pages

        Returns:
            Merged TableDefinition (may be unchanged if no merge)
        """
        current_table = table
        current_page = start_page

        while True:
            next_page = current_page + 1

            if next_page not in page_tables or next_page in processed_pages:
                break

            # Look for a continuation table on the next page
            next_tables = page_tables[next_page]
            continuation = self._find_continuation_table(current_table, next_tables)

            if continuation:
                # Merge the tables
                current_table = self._merge_tables(current_table, continuation)
                processed_pages.add(next_page)
                current_page = next_page
            else:
                break

        return current_table

    def _find_continuation_table(
        self,
        table: TableDefinition,
        candidates: list[TableDefinition],
    ) -> Optional[TableDefinition]:
        """
        Find a table that is a continuation of the given table.

        Criteria for continuation:
        - Similar column structure
        - Similar horizontal position
        - Either no headers or repeated headers

        Args:
            table: Table to find continuation for
            candidates: Candidate tables on next page

        Returns:
            Continuation table or None
        """
        for candidate in candidates:
            # Check column count
            if abs(len(candidate.columns) - len(table.columns)) > 1:
                continue

            # Check column semantic types match
            table_types = [c.semantic_type for c in table.columns if c.semantic_type]
            candidate_types = [c.semantic_type for c in candidate.columns if c.semantic_type]

            # At least 50% of types should match
            if table_types and candidate_types:
                matches = sum(1 for t in table_types if t in candidate_types)
                if matches / len(table_types) < 0.5:
                    continue

            # Check horizontal alignment (similar x-range)
            table_bounds = list(table.bounds_per_page.values())[0]
            candidate_bounds = list(candidate.bounds_per_page.values())[0]

            x_overlap = table_bounds.horizontal_overlap(candidate_bounds)
            if x_overlap / min(table_bounds.width, candidate_bounds.width) < 0.7:
                continue

            return candidate

        return None

    def _merge_tables(
        self,
        table1: TableDefinition,
        table2: TableDefinition,
    ) -> TableDefinition:
        """
        Merge two tables into one.

        Args:
            table1: First table (earlier pages)
            table2: Second table (later pages)

        Returns:
            Merged TableDefinition
        """
        # Combine page numbers
        merged_pages = sorted(set(table1.page_numbers + table2.page_numbers))

        # Combine bounds
        merged_bounds = {**table1.bounds_per_page, **table2.bounds_per_page}

        # Use columns from first table (assumes consistent structure)
        merged_columns = table1.columns

        # Check if headers repeat
        headers_repeat = (
            len(table2.header_row_indices) > 0 and
            len(table1.header_row_indices) > 0
        )

        return TableDefinition(
            table_id=table1.table_id,
            page_numbers=merged_pages,
            columns=merged_columns,
            structure_type=table1.structure_type,
            content_type=table1.content_type,
            bounds_per_page=merged_bounds,
            header_row_indices=table1.header_row_indices,
            header_repeats_on_pages=headers_repeat,
            is_multi_page=True,
            continuation_confidence=0.8,
            detection_confidence=min(table1.detection_confidence, table2.detection_confidence),
            warnings=table1.warnings + table2.warnings,
        )

    def _classify_content_type(
        self,
        table: TableDefinition,
        pdf_document: Any,
    ) -> None:
        """
        Classify the content type of a table.

        Analyzes column semantic types to determine if the table
        contains transactions, summary data, or account info.

        Args:
            table: Table to classify
            pdf_document: PDFDocument instance (for additional context)
        """
        semantic_types = {
            col.semantic_type for col in table.columns
            if col.semantic_type
        }

        header_texts = {
            normalize_header_text(col.header_text)
            for col in table.columns
        }
        header_text_combined = " ".join(header_texts)

        # Check for summary table FIRST - "summary" in headers is a strong signal
        # that this is NOT a transaction table even if it has date/amount columns
        if any(kw in header_text_combined for kw in SUMMARY_KEYWORDS):
            table.content_type = ContentType.SUMMARY
            return

        # Check for account info (before transaction)
        if any(kw in header_text_combined for kw in ACCOUNT_INFO_KEYWORDS):
            table.content_type = ContentType.ACCOUNT_INFO
            return

        # Check for transaction table
        if semantic_types & TRANSACTION_KEYWORDS:
            # Must have date and at least one amount-related column
            has_date = "date" in semantic_types
            has_amount = bool(semantic_types & {"debit", "credit", "balance", "amount"})

            if has_date and has_amount:
                # Additional check: headers should contain typical transaction keywords
                # This prevents tables like "SAC / HSN code" from being classified as transaction
                transaction_header_keywords = {"transaction", "txn", "debit", "credit", "withdrawal", "deposit", "balance"}
                has_transaction_header = any(
                    any(kw in header.lower() for kw in transaction_header_keywords)
                    for header in header_texts
                )
                if has_transaction_header:
                    table.content_type = ContentType.TRANSACTION
                    return

        # Default to OTHER
        table.content_type = ContentType.OTHER


# =============================================================================
# Convenience Functions
# =============================================================================

def detect_tables(
    pdf_document: Any,
    pages: Optional[list[int]] = None,
    use_slm: bool = True,
    filter_watermarks: bool = True,
) -> TableDetectionResult:
    """
    Convenience function to detect tables in a PDF document.

    Args:
        pdf_document: PDFDocument instance
        pages: Optional list of page numbers to process
        use_slm: Enable SLM-based header detection
        filter_watermarks: Enable watermark text filtering

    Returns:
        TableDetectionResult with detected tables

    Example:
        >>> from tables.reader import PDFDocument
        >>> pdf = PDFDocument.open("statement.pdf")
        >>> result = detect_tables(pdf)
        >>> print(f"Found {result.table_count} tables")
    """
    detector = TableDetector(use_slm=use_slm, filter_watermarks=filter_watermarks)
    return detector.detect(pdf_document, pages=pages)
