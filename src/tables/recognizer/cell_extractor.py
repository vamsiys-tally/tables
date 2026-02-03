"""
Cell Extraction from Tables.

This module extracts cell content from PDF tables using column boundaries
defined in TableDefinition. It assigns words to cells based on x-coordinates,
which correctly handles text that overlaps or touches cell borders.

Key Design Decision:
    Cells are defined by column x-boundaries, NOT line intersections.
    This ensures text at cell edges is correctly captured even when
    it touches or overlaps border lines.

Operations:
    1. Extract words from a page within table bounds
    2. Assign each word to a column based on x-coordinates
    3. Group words into rows based on y-coordinates
    4. Extract cell text for each row/column combination

Example Usage:
    >>> from tables.recognizer.cell_extractor import CellExtractor
    >>> extractor = CellExtractor()
    >>> rows = extractor.extract_rows(pdf_document, 0, table_definition)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional
import logging

from tables.models.table import TableDefinition, ColumnDefinition
from tables.utils.geometry import BoundingBox

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

# Tolerance for assigning words to columns (extends column bounds)
COLUMN_TOLERANCE = 3.0

# Tolerance for grouping words into rows
ROW_TOLERANCE = 3.0

# Minimum overlap percentage for word-to-column assignment
MIN_OVERLAP_PERCENT = 0.3


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class Cell:
    """
    A single cell extracted from a table.

    Attributes:
        column_index: Index of the column this cell belongs to
        column: ColumnDefinition for this cell's column
        text: Extracted text content
        words: List of word dictionaries that make up this cell
        bbox: Bounding box of the cell content
        is_empty: Whether the cell is empty
    """

    column_index: int
    column: ColumnDefinition
    text: str = ""
    words: list[dict[str, Any]] = field(default_factory=list)
    bbox: Optional[BoundingBox] = None
    is_empty: bool = True

    def add_word(self, word: dict[str, Any]) -> None:
        """Add a word to this cell."""
        self.words.append(word)
        self.is_empty = False

        # Update text
        word_text = word.get("text", "").strip()
        if self.text:
            self.text += " " + word_text
        else:
            self.text = word_text

        # Update bounding box
        word_bbox = BoundingBox(
            x0=word.get("x0", 0),
            y0=word.get("top", 0),
            x1=word.get("x1", 0),
            y1=word.get("bottom", 0),
        )

        if self.bbox is None:
            self.bbox = word_bbox
        else:
            # Use class method to compute union of two boxes
            self.bbox = BoundingBox.union([self.bbox, word_bbox])


@dataclass
class ExtractedRow:
    """
    A row of cells extracted from a table.

    Represents a single visual row (not necessarily a complete transaction).

    Attributes:
        row_index: Sequential row index (0-indexed from table start)
        page_number: Page where this row was found
        cells: Dictionary mapping column_index to Cell
        y_position: Y-coordinate of the row (top of first word)
        height: Height of the row
        is_header: Whether this row appears to be a header
    """

    row_index: int
    page_number: int
    cells: dict[int, Cell] = field(default_factory=dict)
    y_position: float = 0.0
    height: float = 0.0
    is_header: bool = False

    def get_cell(self, column_index: int) -> Optional[Cell]:
        """Get cell by column index."""
        return self.cells.get(column_index)

    def get_cell_text(self, column_index: int) -> str:
        """Get cell text by column index, or empty string."""
        cell = self.cells.get(column_index)
        return cell.text if cell else ""

    def get_cell_by_semantic_type(
        self,
        semantic_type: str,
        columns: list[ColumnDefinition],
    ) -> Optional[Cell]:
        """Get cell by column semantic type."""
        for col in columns:
            if col.semantic_type == semantic_type:
                return self.cells.get(col.column_id)
        return None

    @property
    def all_cells_empty(self) -> bool:
        """Check if all cells in the row are empty."""
        return all(cell.is_empty for cell in self.cells.values())

    @property
    def word_count(self) -> int:
        """Total number of words in the row."""
        return sum(len(cell.words) for cell in self.cells.values())


# =============================================================================
# Cell Extractor
# =============================================================================

class CellExtractor:
    """
    Extracts cell content from PDF tables using column boundaries.

    Uses column x-coordinates (not line intersections) to assign words
    to cells, which correctly handles text that overlaps cell borders.

    Attributes:
        column_tolerance: Tolerance for extending column bounds
        row_tolerance: Tolerance for grouping words into rows
        min_overlap: Minimum overlap percentage for column assignment

    Example:
        >>> extractor = CellExtractor()
        >>> rows = extractor.extract_rows(pdf_document, page_num, table_def)
        >>> for row in rows:
        ...     print([cell.text for cell in row.cells.values()])
    """

    def __init__(
        self,
        column_tolerance: float = COLUMN_TOLERANCE,
        row_tolerance: float = ROW_TOLERANCE,
        min_overlap: float = MIN_OVERLAP_PERCENT,
    ):
        """
        Initialize the cell extractor.

        Args:
            column_tolerance: Tolerance for extending column bounds
            row_tolerance: Tolerance for grouping words into rows
            min_overlap: Minimum overlap percentage for column assignment
        """
        self.column_tolerance = column_tolerance
        self.row_tolerance = row_tolerance
        self.min_overlap = min_overlap

    def extract_rows(
        self,
        pdf_document: Any,
        page_number: int,
        table_definition: TableDefinition,
        skip_header_rows: bool = True,
    ) -> list[ExtractedRow]:
        """
        Extract rows of cells from a table on a specific page.

        Args:
            pdf_document: PDFDocument instance
            page_number: Page number to extract from
            table_definition: Table definition with column information
            skip_header_rows: Whether to skip detected header rows

        Returns:
            List of ExtractedRow objects in reading order (top to bottom)
        """
        # Get table bounds for this page
        bounds = table_definition.get_bounds(page_number)
        if bounds is None:
            logger.warning(
                f"No bounds for table {table_definition.table_id} on page {page_number}"
            )
            return []

        # Get words from the page within table bounds
        words = self._get_words_in_region(pdf_document, page_number, bounds)
        if not words:
            logger.debug(f"No words found in table region on page {page_number}")
            return []

        # Group words into rows by y-coordinate
        word_rows = self._group_words_into_rows(words)

        # Convert to ExtractedRow objects
        extracted_rows = []
        for row_idx, row_words in enumerate(word_rows):
            row = self._create_extracted_row(
                row_idx=row_idx,
                page_number=page_number,
                words=row_words,
                columns=table_definition.columns,
            )
            extracted_rows.append(row)

        # Mark header rows
        if table_definition.header_row_indices:
            for header_idx in table_definition.header_row_indices:
                if 0 <= header_idx < len(extracted_rows):
                    extracted_rows[header_idx].is_header = True

        # Filter out header rows if requested
        if skip_header_rows:
            extracted_rows = [r for r in extracted_rows if not r.is_header]

        logger.debug(
            f"Page {page_number}: Extracted {len(extracted_rows)} data rows "
            f"from table {table_definition.table_id}"
        )

        return extracted_rows

    def extract_cells_from_row(
        self,
        words: list[dict[str, Any]],
        columns: list[ColumnDefinition],
    ) -> dict[int, Cell]:
        """
        Assign words to columns to create cells for a single row.

        Args:
            words: List of word dictionaries
            columns: List of column definitions

        Returns:
            Dictionary mapping column_id to Cell
        """
        cells: dict[int, Cell] = {}

        # Initialize empty cells for each column
        for col in columns:
            cells[col.column_id] = Cell(
                column_index=col.column_id,
                column=col,
            )

        # Assign each word to the best matching column
        for word in words:
            word_x0 = word.get("x0", 0)
            word_x1 = word.get("x1", 0)

            best_column = self._find_best_column(word_x0, word_x1, columns)
            if best_column is not None:
                cells[best_column.column_id].add_word(word)

        return cells

    def _get_words_in_region(
        self,
        pdf_document: Any,
        page_number: int,
        bounds: BoundingBox,
    ) -> list[dict[str, Any]]:
        """
        Get words from a page that fall within the specified region.

        Args:
            pdf_document: PDFDocument instance
            page_number: Page number
            bounds: Region bounding box

        Returns:
            List of word dictionaries within the region
        """
        all_words = pdf_document.get_page_words(page_number)
        filtered_words = []

        for word in all_words:
            word_x0 = word.get("x0", 0)
            word_y0 = word.get("top", 0)
            word_x1 = word.get("x1", 0)
            word_y1 = word.get("bottom", 0)

            # Check if word is within bounds (with some tolerance)
            if (bounds.x0 - self.column_tolerance <= word_x0 <= bounds.x1 + self.column_tolerance and
                bounds.y0 - self.row_tolerance <= word_y0 <= bounds.y1 + self.row_tolerance):
                filtered_words.append(word)

        return filtered_words

    def _group_words_into_rows(
        self,
        words: list[dict[str, Any]],
    ) -> list[list[dict[str, Any]]]:
        """
        Group words into rows based on y-coordinate clustering.

        Args:
            words: List of word dictionaries

        Returns:
            List of word lists, each representing a row
        """
        if not words:
            return []

        # Sort words by y-coordinate (top to bottom in pdfplumber coordinates)
        sorted_words = sorted(words, key=lambda w: w.get("top", 0))

        # Group into rows by y-coordinate proximity
        rows: list[list[dict[str, Any]]] = []
        current_row: list[dict[str, Any]] = []
        current_y: Optional[float] = None

        for word in sorted_words:
            word_y = word.get("top", 0)

            if current_y is None or abs(word_y - current_y) <= self.row_tolerance:
                # Same row
                current_row.append(word)
                if current_y is None:
                    current_y = word_y
                else:
                    # Update running average of y-coordinate
                    current_y = (current_y * (len(current_row) - 1) + word_y) / len(current_row)
            else:
                # New row
                if current_row:
                    rows.append(current_row)
                current_row = [word]
                current_y = word_y

        # Don't forget last row
        if current_row:
            rows.append(current_row)

        # Sort words within each row by x-position (left to right)
        for row in rows:
            row.sort(key=lambda w: w.get("x0", 0))

        return rows

    def _create_extracted_row(
        self,
        row_idx: int,
        page_number: int,
        words: list[dict[str, Any]],
        columns: list[ColumnDefinition],
    ) -> ExtractedRow:
        """
        Create an ExtractedRow from a list of words.

        Args:
            row_idx: Row index
            page_number: Page number
            words: Words in this row
            columns: Column definitions

        Returns:
            ExtractedRow with cells populated
        """
        cells = self.extract_cells_from_row(words, columns)

        # Calculate row position and height
        y_positions = [w.get("top", 0) for w in words]
        y_bottoms = [w.get("bottom", 0) for w in words]

        y_position = min(y_positions) if y_positions else 0.0
        y_bottom = max(y_bottoms) if y_bottoms else 0.0
        height = y_bottom - y_position

        return ExtractedRow(
            row_index=row_idx,
            page_number=page_number,
            cells=cells,
            y_position=y_position,
            height=height,
        )

    def _find_best_column(
        self,
        word_x0: float,
        word_x1: float,
        columns: list[ColumnDefinition],
    ) -> Optional[ColumnDefinition]:
        """
        Find the best column for a word based on x-coordinate overlap.

        Uses overlap percentage to handle words that span column boundaries.
        Falls back to center-point containment for small words.

        Args:
            word_x0: Word left edge
            word_x1: Word right edge
            columns: List of column definitions

        Returns:
            Best matching column, or None if no match
        """
        word_width = word_x1 - word_x0
        word_center = (word_x0 + word_x1) / 2

        best_column = None
        best_overlap = 0.0

        for col in columns:
            # Extend column bounds by tolerance
            col_x0 = col.x0 - self.column_tolerance
            col_x1 = col.x1 + self.column_tolerance

            # Calculate overlap
            overlap_start = max(word_x0, col_x0)
            overlap_end = min(word_x1, col_x1)
            overlap = max(0.0, overlap_end - overlap_start)

            if word_width > 0:
                overlap_percent = overlap / word_width
            else:
                overlap_percent = 0.0

            if overlap_percent > best_overlap:
                best_overlap = overlap_percent
                best_column = col

        # Require minimum overlap
        if best_overlap < self.min_overlap:
            # Fall back to center-point containment
            for col in columns:
                if col.contains_x(word_center, tolerance=self.column_tolerance):
                    return col
            return None

        return best_column


# =============================================================================
# Convenience Functions
# =============================================================================

def extract_table_rows(
    pdf_document: Any,
    page_number: int,
    table_definition: TableDefinition,
    skip_headers: bool = True,
) -> list[ExtractedRow]:
    """
    Convenience function to extract rows from a table.

    Args:
        pdf_document: PDFDocument instance
        page_number: Page number to extract from
        table_definition: Table definition with column information
        skip_headers: Whether to skip detected header rows

    Returns:
        List of ExtractedRow objects

    Example:
        >>> rows = extract_table_rows(pdf, 0, table_def)
        >>> for row in rows:
        ...     print(row.get_cell_text(0))  # First column
    """
    extractor = CellExtractor()
    return extractor.extract_rows(
        pdf_document,
        page_number,
        table_definition,
        skip_header_rows=skip_headers,
    )


def get_cell_text_by_column(
    row: ExtractedRow,
    columns: list[ColumnDefinition],
    semantic_type: str,
) -> str:
    """
    Get cell text from a row by semantic column type.

    Args:
        row: ExtractedRow to get cell from
        columns: Column definitions
        semantic_type: Semantic type to look for (e.g., "date", "debit")

    Returns:
        Cell text, or empty string if not found

    Example:
        >>> date_text = get_cell_text_by_column(row, columns, "date")
    """
    for col in columns:
        if col.semantic_type == semantic_type:
            cell = row.cells.get(col.column_id)
            if cell:
                return cell.text
    return ""
