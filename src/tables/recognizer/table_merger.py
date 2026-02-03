"""
Cross-Page Table Merging.

This module handles merging tables that span multiple pages, including:
- Identifying continuation tables
- Merging rows while maintaining integrity
- Handling mid-row page breaks
- Semantic column mapping (not position-based)

Key Features:
    - Detects if a table continues from the previous page
    - Merges rows from multiple pages into a single table
    - Handles case where a transaction is split across pages
    - Maps columns by semantic type, not x-position

Example Usage:
    >>> from tables.recognizer.table_merger import TableMerger
    >>> merger = TableMerger()
    >>> merged_rows = merger.merge_multi_page_table(
    ...     pdf_document, table_definition
    ... )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional
import logging

from tables.models.table import TableDefinition, ColumnDefinition
from tables.recognizer.cell_extractor import CellExtractor, ExtractedRow

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

# Maximum y-distance from page bottom to consider row as potentially split
PAGE_BOTTOM_THRESHOLD = 50.0

# Maximum y-distance from page top to consider row as continuation
PAGE_TOP_THRESHOLD = 50.0


# =============================================================================
# Page Boundary Detection
# =============================================================================

@dataclass
class PageBoundaryInfo:
    """
    Information about row positioning relative to page boundaries.

    Used to detect mid-row page breaks.

    Attributes:
        page_number: The page number
        page_height: Height of the page
        last_row_y_bottom: Y-coordinate of bottom of last row
        first_row_y_top: Y-coordinate of top of first row
        is_near_bottom: Whether last row is near page bottom
        is_near_top: Whether first row is near page top
    """

    page_number: int
    page_height: float = 0.0
    last_row_y_bottom: float = 0.0
    first_row_y_top: float = 0.0
    is_near_bottom: bool = False
    is_near_top: bool = False


# =============================================================================
# Column Mapper
# =============================================================================

class SemanticColumnMapper:
    """
    Maps columns between pages by semantic type, not position.

    Handles the case where columns might be slightly misaligned
    across pages, but have the same semantic meaning.
    """

    def __init__(self, reference_columns: list[ColumnDefinition]):
        """
        Initialize with reference columns (usually from first page).

        Args:
            reference_columns: Column definitions to map to
        """
        self.reference_columns = reference_columns
        self._type_to_column: dict[str, ColumnDefinition] = {}

        for col in reference_columns:
            if col.semantic_type:
                self._type_to_column[col.semantic_type] = col

    def map_cell_to_reference(
        self,
        source_column: ColumnDefinition,
    ) -> Optional[ColumnDefinition]:
        """
        Map a source column to the reference column by semantic type.

        Args:
            source_column: Column from another page

        Returns:
            Matching reference column, or None
        """
        if source_column.semantic_type:
            return self._type_to_column.get(source_column.semantic_type)

        # Fall back to position-based matching
        for ref_col in self.reference_columns:
            if ref_col.column_id == source_column.column_id:
                return ref_col

        return None

    def remap_row(
        self,
        row: ExtractedRow,
        source_columns: list[ColumnDefinition],
    ) -> ExtractedRow:
        """
        Remap a row's cells to reference column IDs.

        Args:
            row: Row to remap
            source_columns: Source column definitions

        Returns:
            New ExtractedRow with remapped cell keys
        """
        new_cells = {}

        for col_id, cell in row.cells.items():
            # Find the source column
            source_col = next(
                (c for c in source_columns if c.column_id == col_id),
                None
            )
            if source_col is None:
                continue

            # Map to reference column
            ref_col = self.map_cell_to_reference(source_col)
            if ref_col is not None:
                # Update cell to use reference column
                cell.column_index = ref_col.column_id
                cell.column = ref_col
                new_cells[ref_col.column_id] = cell

        return ExtractedRow(
            row_index=row.row_index,
            page_number=row.page_number,
            cells=new_cells,
            y_position=row.y_position,
            height=row.height,
            is_header=row.is_header,
        )


# =============================================================================
# Table Merger
# =============================================================================

class TableMerger:
    """
    Merges multi-page tables into unified row sequences.

    Handles:
    - Extracting rows from each page of a multi-page table
    - Detecting and handling mid-row page breaks
    - Mapping columns semantically across pages
    - Renumbering rows for the merged sequence

    Attributes:
        cell_extractor: CellExtractor instance for row extraction

    Example:
        >>> merger = TableMerger()
        >>> all_rows = merger.merge_multi_page_table(pdf, table_def)
        >>> print(f"Merged {len(all_rows)} rows from {len(table_def.page_numbers)} pages")
    """

    def __init__(self):
        """Initialize the table merger."""
        self.cell_extractor = CellExtractor()

    def merge_multi_page_table(
        self,
        pdf_document: Any,
        table_definition: TableDefinition,
    ) -> list[ExtractedRow]:
        """
        Extract and merge rows from a multi-page table.

        Args:
            pdf_document: PDFDocument instance
            table_definition: Table definition spanning multiple pages

        Returns:
            List of ExtractedRow objects in order
        """
        if not table_definition.page_numbers:
            return []

        all_rows: list[ExtractedRow] = []
        column_mapper = SemanticColumnMapper(table_definition.columns)

        # Process each page
        previous_page_rows: list[ExtractedRow] = []
        previous_boundary_info: Optional[PageBoundaryInfo] = None

        for page_idx, page_number in enumerate(table_definition.page_numbers):
            # Extract rows from this page
            # Always skip headers on first page
            # On subsequent pages, skip only if headers repeat
            skip_headers = (page_idx == 0) or table_definition.header_repeats_on_pages
            page_rows = self.cell_extractor.extract_rows(
                pdf_document,
                page_number,
                table_definition,
                skip_header_rows=skip_headers,
            )

            if not page_rows:
                continue

            # Get page dimensions for boundary detection
            page_info = pdf_document.get_page_info(page_number)
            page_height = page_info.height if page_info else 800.0

            # Calculate boundary info
            boundary_info = PageBoundaryInfo(
                page_number=page_number,
                page_height=page_height,
                first_row_y_top=page_rows[0].y_position if page_rows else 0,
                last_row_y_bottom=(
                    page_rows[-1].y_position + page_rows[-1].height
                    if page_rows else 0
                ),
            )

            # Check for mid-row page break
            if previous_page_rows and previous_boundary_info:
                if self._detect_mid_row_break(previous_boundary_info, boundary_info, page_rows):
                    # Merge first row of current page with last row of previous page
                    logger.debug(
                        f"Detected mid-row page break between pages "
                        f"{previous_boundary_info.page_number} and {page_number}"
                    )
                    if all_rows and page_rows:
                        merged_row = self._merge_split_row(
                            all_rows[-1],
                            page_rows[0],
                        )
                        all_rows[-1] = merged_row
                        page_rows = page_rows[1:]  # Remove merged row

            # Add rows from this page
            for row in page_rows:
                # Remap columns if this isn't the first page
                if page_idx > 0:
                    row = column_mapper.remap_row(row, table_definition.columns)
                all_rows.append(row)

            previous_page_rows = page_rows
            previous_boundary_info = boundary_info

        # Renumber rows
        for idx, row in enumerate(all_rows):
            row.row_index = idx

        logger.debug(
            f"Merged {len(all_rows)} rows from "
            f"{len(table_definition.page_numbers)} pages"
        )

        return all_rows

    def extract_single_page_rows(
        self,
        pdf_document: Any,
        table_definition: TableDefinition,
        page_number: int,
    ) -> list[ExtractedRow]:
        """
        Extract rows from a single page of a table.

        Convenience method for extracting from one page.

        Args:
            pdf_document: PDFDocument instance
            table_definition: Table definition
            page_number: Specific page to extract from

        Returns:
            List of ExtractedRow objects
        """
        return self.cell_extractor.extract_rows(
            pdf_document,
            page_number,
            table_definition,
            skip_header_rows=True,
        )

    def _detect_mid_row_break(
        self,
        previous_info: PageBoundaryInfo,
        current_info: PageBoundaryInfo,
        current_rows: list[ExtractedRow],
    ) -> bool:
        """
        Detect if a transaction row was split across pages.

        Signals:
        1. Last row of previous page is near bottom
        2. First row of current page is near top
        3. First row of current page has incomplete data (e.g., no date)

        Args:
            previous_info: Boundary info for previous page
            current_info: Boundary info for current page
            current_rows: Rows from current page

        Returns:
            True if mid-row break detected
        """
        if not current_rows:
            return False

        # Check position-based signals
        distance_from_bottom = (
            previous_info.page_height - previous_info.last_row_y_bottom
        )
        distance_from_top = current_info.first_row_y_top

        # If both are near their respective edges, might be split
        position_indicates_split = (
            distance_from_bottom < PAGE_BOTTOM_THRESHOLD and
            distance_from_top < PAGE_TOP_THRESHOLD
        )

        if not position_indicates_split:
            return False

        # Check content-based signals
        first_row = current_rows[0]

        # If first row has no date (for date-prefixed tables), might be continuation
        has_date = False
        for cell in first_row.cells.values():
            if cell.column.semantic_type == "date" and not cell.is_empty:
                has_date = True
                break

        # Only consider split if position suggests it AND date is missing
        return not has_date

    def _merge_split_row(
        self,
        previous_row: ExtractedRow,
        continuation_row: ExtractedRow,
    ) -> ExtractedRow:
        """
        Merge two rows that were split across pages.

        Concatenates text for each column.

        Args:
            previous_row: Row from previous page
            continuation_row: Continuation row from current page

        Returns:
            Merged ExtractedRow
        """
        merged_cells = dict(previous_row.cells)

        for col_id, cont_cell in continuation_row.cells.items():
            if cont_cell.is_empty:
                continue

            if col_id in merged_cells:
                # Append text to existing cell
                prev_cell = merged_cells[col_id]
                if not prev_cell.is_empty:
                    prev_cell.text += " " + cont_cell.text
                else:
                    prev_cell.text = cont_cell.text
                prev_cell.is_empty = False
                prev_cell.words.extend(cont_cell.words)
            else:
                merged_cells[col_id] = cont_cell

        return ExtractedRow(
            row_index=previous_row.row_index,
            page_number=previous_row.page_number,  # Keep original page
            cells=merged_cells,
            y_position=previous_row.y_position,
            height=previous_row.height + continuation_row.height,
            is_header=False,
        )


# =============================================================================
# Convenience Functions
# =============================================================================

def merge_table_pages(
    pdf_document: Any,
    table_definition: TableDefinition,
) -> list[ExtractedRow]:
    """
    Convenience function to merge rows from a multi-page table.

    Args:
        pdf_document: PDFDocument instance
        table_definition: Table definition

    Returns:
        List of merged ExtractedRow objects

    Example:
        >>> rows = merge_table_pages(pdf, table_def)
        >>> print(f"Got {len(rows)} rows from multi-page table")
    """
    merger = TableMerger()
    return merger.merge_multi_page_table(pdf_document, table_definition)
