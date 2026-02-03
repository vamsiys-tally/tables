"""
Table Structure Classification.

This module provides functionality to classify the visual structure
of tables detected in PDF pages. It determines whether tables are:

- Bordered: Full grid with horizontal and vertical lines
- Semi-bordered: Partial lines (often horizontal separators only)
- Unbordered: No lines, relies on whitespace/text alignment
- Shaded: Uses background colors to differentiate rows

Classification Strategy:
    1. Extract all lines from the page (horizontal and vertical)
    2. Extract all rectangles (potential cell boundaries or shading)
    3. Analyze line patterns (grid formation, intersections)
    4. Detect shading patterns
    5. Classify based on dominant structure

The classifier uses geometric analysis without relying on content.
This aligns with the module design principle of pure structural extraction.

Example Usage:
    >>> from tables.detector.structure_classifier import StructureClassifier
    >>> from tables.reader import PDFDocument
    >>>
    >>> pdf = PDFDocument.open("statement.pdf")
    >>> classifier = StructureClassifier()
    >>> structure = classifier.classify_page(pdf, page_number=0)
    >>> print(structure.structure_type)
    StructureType.BORDERED
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Any
from enum import Enum

from tables.utils.geometry import (
    BoundingBox,
    Line,
    Point,
    cluster_by_y_coordinate,
    cluster_by_x_coordinate,
)
from tables.models.table import StructureType


# =============================================================================
# Configuration Constants
# =============================================================================

# Minimum line length to consider (in PDF points)
MIN_LINE_LENGTH = 20.0

# Maximum gap between line endpoints to consider them connected
LINE_CONNECTION_TOLERANCE = 5.0

# Minimum number of horizontal lines to indicate bordered/semi-bordered
MIN_HORIZONTAL_LINES = 2

# Minimum number of vertical lines for a bordered table
MIN_VERTICAL_LINES = 3

# Ratio thresholds for classification
# If vertical_lines / horizontal_lines >= this, likely bordered
BORDERED_RATIO_THRESHOLD = 0.5

# Minimum rectangle count to indicate shaded structure
MIN_SHADED_RECTANGLES = 3

# Alignment tolerance for detecting grid patterns
GRID_ALIGNMENT_TOLERANCE = 3.0


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class LineAnalysis:
    """
    Analysis results for lines on a page.

    Contains extracted and categorized line information used
    to determine table structure.

    Attributes:
        horizontal_lines: All horizontal lines found
        vertical_lines: All vertical lines found
        horizontal_clusters: Horizontal lines grouped by y-coordinate
        vertical_clusters: Vertical lines grouped by x-coordinate
        grid_intersections: Points where horizontal and vertical lines meet
        has_grid_pattern: Whether lines form a regular grid
    """

    horizontal_lines: list[Line] = field(default_factory=list)
    vertical_lines: list[Line] = field(default_factory=list)
    horizontal_clusters: list[list[Line]] = field(default_factory=list)
    vertical_clusters: list[list[Line]] = field(default_factory=list)
    grid_intersections: list[Point] = field(default_factory=list)
    has_grid_pattern: bool = False


@dataclass
class RectangleAnalysis:
    """
    Analysis results for rectangles on a page.

    Contains information about filled rectangles that may indicate
    shaded table structures.

    Attributes:
        all_rectangles: All rectangles found on the page
        filled_rectangles: Rectangles with fill color (shading)
        row_shading_pattern: Whether rectangles form row shading pattern
        column_shading_pattern: Whether rectangles form column shading pattern
    """

    all_rectangles: list[BoundingBox] = field(default_factory=list)
    filled_rectangles: list[BoundingBox] = field(default_factory=list)
    row_shading_pattern: bool = False
    column_shading_pattern: bool = False


@dataclass
class StructureAnalysis:
    """
    Complete structure analysis result for a page region.

    Combines line and rectangle analysis with the final classification.

    Attributes:
        structure_type: Determined structure type
        confidence: Confidence in the classification (0.0-1.0)
        line_analysis: Detailed line analysis
        rectangle_analysis: Detailed rectangle analysis
        bounds: Bounding box of the analyzed region
        evidence: Human-readable explanation of classification
    """

    structure_type: StructureType = StructureType.UNKNOWN
    confidence: float = 0.0
    line_analysis: LineAnalysis = field(default_factory=LineAnalysis)
    rectangle_analysis: RectangleAnalysis = field(default_factory=RectangleAnalysis)
    bounds: Optional[BoundingBox] = None
    evidence: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for debugging/logging."""
        return {
            "structure_type": self.structure_type.value,
            "confidence": self.confidence,
            "evidence": self.evidence,
            "horizontal_line_count": len(self.line_analysis.horizontal_lines),
            "vertical_line_count": len(self.line_analysis.vertical_lines),
            "has_grid_pattern": self.line_analysis.has_grid_pattern,
            "filled_rectangle_count": len(self.rectangle_analysis.filled_rectangles),
            "bounds": self.bounds.to_dict() if self.bounds else None,
        }


# =============================================================================
# Structure Classifier
# =============================================================================

class StructureClassifier:
    """
    Classifier for table visual structure.

    Analyzes PDF page elements (lines, rectangles) to determine
    the visual structure type of tables. This information guides
    the cell extraction strategy in Module 3.

    The classifier is stateless and can be reused across multiple pages.

    Attributes:
        min_line_length: Minimum line length to consider
        grid_tolerance: Tolerance for detecting grid alignment

    Example:
        >>> classifier = StructureClassifier()
        >>> analysis = classifier.classify_page(pdf, page_number=0)
        >>> if analysis.structure_type == StructureType.BORDERED:
        ...     print("Use line-based cell extraction")
    """

    def __init__(
        self,
        min_line_length: float = MIN_LINE_LENGTH,
        grid_tolerance: float = GRID_ALIGNMENT_TOLERANCE,
    ):
        """
        Initialize the structure classifier.

        Args:
            min_line_length: Minimum line length to consider significant
            grid_tolerance: Tolerance for detecting aligned grid lines
        """
        self.min_line_length = min_line_length
        self.grid_tolerance = grid_tolerance

    def classify_page(
        self,
        pdf_document: Any,  # PDFDocument, using Any to avoid circular import
        page_number: int,
        region: Optional[BoundingBox] = None,
    ) -> StructureAnalysis:
        """
        Classify the table structure on a PDF page.

        Analyzes the visual elements on a page (or region of a page)
        to determine the dominant table structure type.

        Args:
            pdf_document: PDFDocument instance
            page_number: Page number to analyze (0-indexed)
            region: Optional region to analyze (defaults to full page)

        Returns:
            StructureAnalysis with classification and supporting data

        Example:
            >>> analysis = classifier.classify_page(pdf, 0)
            >>> print(f"Structure: {analysis.structure_type.value}")
            >>> print(f"Confidence: {analysis.confidence:.2f}")
        """
        # Extract raw elements from PDF
        # Use get_page_lines_raw to get dict format for processing
        lines = pdf_document.get_page_lines_raw(page_number)
        rects = pdf_document.get_page_rects(page_number)

        # Convert to our geometry types and filter by region
        processed_lines = self._process_lines(lines, region)
        processed_rects = self._process_rectangles(rects, region)

        # Analyze lines
        line_analysis = self._analyze_lines(processed_lines)

        # Analyze rectangles
        rect_analysis = self._analyze_rectangles(processed_rects)

        # Determine structure type
        return self._classify_structure(line_analysis, rect_analysis, region)

    def classify_from_elements(
        self,
        lines: list[Line],
        rectangles: list[BoundingBox],
        region: Optional[BoundingBox] = None,
    ) -> StructureAnalysis:
        """
        Classify structure from pre-extracted elements.

        Useful when elements have already been extracted and filtered.

        Args:
            lines: List of Line objects
            rectangles: List of BoundingBox objects (potential cells/shading)
            region: Optional bounding region

        Returns:
            StructureAnalysis with classification
        """
        line_analysis = self._analyze_lines(lines)
        rect_analysis = self._analyze_rectangles(rectangles)
        return self._classify_structure(line_analysis, rect_analysis, region)

    def _process_lines(
        self,
        raw_lines: list[dict],
        region: Optional[BoundingBox],
    ) -> list[Line]:
        """
        Convert raw line dictionaries to Line objects with filtering.

        Args:
            raw_lines: Lines from pdfplumber (list of dicts)
            region: Optional region to filter lines

        Returns:
            List of Line objects meeting criteria
        """
        processed: list[Line] = []

        for raw in raw_lines:
            line = Line(
                x0=raw.get("x0", 0),
                y0=raw.get("y0", raw.get("top", 0)),
                x1=raw.get("x1", 0),
                y1=raw.get("y1", raw.get("bottom", 0)),
            )

            # Filter by length
            if line.length < self.min_line_length:
                continue

            # Filter by region if specified
            if region:
                line_box = BoundingBox(
                    x0=min(line.x0, line.x1),
                    y0=min(line.y0, line.y1),
                    x1=max(line.x0, line.x1),
                    y1=max(line.y0, line.y1),
                )
                if not region.overlaps(line_box):
                    continue

            processed.append(line)

        return processed

    def _process_rectangles(
        self,
        raw_rects: list[dict],
        region: Optional[BoundingBox],
    ) -> list[BoundingBox]:
        """
        Convert raw rectangle dictionaries to BoundingBox objects.

        Args:
            raw_rects: Rectangles from pdfplumber
            region: Optional region to filter rectangles

        Returns:
            List of BoundingBox objects
        """
        processed: list[BoundingBox] = []

        for raw in raw_rects:
            rect = BoundingBox.from_dict(raw)

            # Filter very small rectangles (likely artifacts)
            if rect.width < 5 or rect.height < 5:
                continue

            # Filter by region if specified
            if region and not region.overlaps(rect):
                continue

            processed.append(rect)

        return processed

    def _analyze_lines(self, lines: list[Line]) -> LineAnalysis:
        """
        Analyze lines to detect patterns and structure.

        Args:
            lines: List of Line objects

        Returns:
            LineAnalysis with categorized lines and patterns
        """
        analysis = LineAnalysis()

        # Separate horizontal and vertical lines
        for line in lines:
            if line.is_horizontal:
                analysis.horizontal_lines.append(line)
            elif line.is_vertical:
                analysis.vertical_lines.append(line)

        # Cluster horizontal lines by y-coordinate
        if analysis.horizontal_lines:
            h_boxes = [
                BoundingBox(l.x0, l.y0, l.x1, l.y1)
                for l in analysis.horizontal_lines
            ]
            h_clusters = cluster_by_y_coordinate(h_boxes, tolerance=self.grid_tolerance)
            # Convert back to line clusters
            analysis.horizontal_clusters = [
                [Line(b.x0, b.y0, b.x1, b.y1) for b in cluster]
                for cluster in h_clusters
            ]

        # Cluster vertical lines by x-coordinate
        if analysis.vertical_lines:
            v_boxes = [
                BoundingBox(l.x0, l.y0, l.x1, l.y1)
                for l in analysis.vertical_lines
            ]
            v_clusters = cluster_by_x_coordinate(v_boxes, tolerance=self.grid_tolerance)
            # Convert back to line clusters
            analysis.vertical_clusters = [
                [Line(b.x0, b.y0, b.x1, b.y1) for b in cluster]
                for cluster in v_clusters
            ]

        # Detect grid intersections
        analysis.grid_intersections = self._find_grid_intersections(
            analysis.horizontal_lines,
            analysis.vertical_lines,
        )

        # Check for grid pattern
        analysis.has_grid_pattern = self._has_grid_pattern(analysis)

        return analysis

    def _find_grid_intersections(
        self,
        horizontal: list[Line],
        vertical: list[Line],
    ) -> list[Point]:
        """
        Find intersection points between horizontal and vertical lines.

        Args:
            horizontal: Horizontal lines
            vertical: Vertical lines

        Returns:
            List of intersection points
        """
        intersections: list[Point] = []

        for h_line in horizontal:
            for v_line in vertical:
                # Check if lines can intersect
                # Horizontal line: y is constant (approximately)
                # Vertical line: x is constant (approximately)
                h_y = (h_line.y0 + h_line.y1) / 2
                v_x = (v_line.x0 + v_line.x1) / 2

                # Check if the intersection point is within both line segments
                h_x_min = min(h_line.x0, h_line.x1) - LINE_CONNECTION_TOLERANCE
                h_x_max = max(h_line.x0, h_line.x1) + LINE_CONNECTION_TOLERANCE
                v_y_min = min(v_line.y0, v_line.y1) - LINE_CONNECTION_TOLERANCE
                v_y_max = max(v_line.y0, v_line.y1) + LINE_CONNECTION_TOLERANCE

                if h_x_min <= v_x <= h_x_max and v_y_min <= h_y <= v_y_max:
                    intersections.append(Point(v_x, h_y))

        return intersections

    def _has_grid_pattern(self, analysis: LineAnalysis) -> bool:
        """
        Determine if lines form a regular grid pattern.

        A grid pattern requires:
        - Multiple horizontal lines at regular y-intervals
        - Multiple vertical lines at regular x-intervals
        - Significant intersections

        Args:
            analysis: Line analysis data

        Returns:
            True if a grid pattern is detected
        """
        # Need minimum lines
        if (len(analysis.horizontal_lines) < MIN_HORIZONTAL_LINES or
                len(analysis.vertical_lines) < MIN_VERTICAL_LINES):
            return False

        # Need significant intersections
        min_expected_intersections = (
            len(analysis.horizontal_clusters) * len(analysis.vertical_clusters) * 0.3
        )
        if len(analysis.grid_intersections) < min_expected_intersections:
            return False

        return True

    def _analyze_rectangles(self, rectangles: list[BoundingBox]) -> RectangleAnalysis:
        """
        Analyze rectangles for shading patterns.

        Args:
            rectangles: List of rectangle bounding boxes

        Returns:
            RectangleAnalysis with shading pattern detection
        """
        analysis = RectangleAnalysis()
        analysis.all_rectangles = rectangles

        # For now, treat all rectangles as potentially filled
        # In a full implementation, we'd check fill color from PDF
        analysis.filled_rectangles = rectangles

        if len(rectangles) >= MIN_SHADED_RECTANGLES:
            # Check for row shading pattern (rectangles with similar widths, different y)
            analysis.row_shading_pattern = self._detect_row_shading(rectangles)

            # Check for column shading pattern (similar heights, different x)
            analysis.column_shading_pattern = self._detect_column_shading(rectangles)

        return analysis

    def _detect_row_shading(self, rectangles: list[BoundingBox]) -> bool:
        """
        Detect if rectangles form a row shading pattern.

        Row shading: rectangles span full width and alternate rows.

        Args:
            rectangles: List of rectangles to analyze

        Returns:
            True if row shading pattern detected
        """
        if len(rectangles) < 2:
            return False

        # Cluster by y-coordinate
        y_clusters = cluster_by_y_coordinate(rectangles, tolerance=5.0)

        # Row shading typically has multiple rows with similar widths
        widths = [r.width for r in rectangles]
        if not widths:
            return False

        avg_width = sum(widths) / len(widths)

        # Check if most rectangles have similar width (spanning row)
        similar_width_count = sum(
            1 for w in widths
            if abs(w - avg_width) / avg_width < 0.2  # Within 20%
        )

        return (
            len(y_clusters) >= 2 and
            similar_width_count / len(widths) >= 0.7
        )

    def _detect_column_shading(self, rectangles: list[BoundingBox]) -> bool:
        """
        Detect if rectangles form a column shading pattern.

        Column shading: rectangles span full height and alternate columns.

        Args:
            rectangles: List of rectangles to analyze

        Returns:
            True if column shading pattern detected
        """
        if len(rectangles) < 2:
            return False

        # Cluster by x-coordinate
        x_clusters = cluster_by_x_coordinate(rectangles, tolerance=5.0)

        # Column shading has multiple columns with similar heights
        heights = [r.height for r in rectangles]
        if not heights:
            return False

        avg_height = sum(heights) / len(heights)

        # Check if most rectangles have similar height (spanning column)
        similar_height_count = sum(
            1 for h in heights
            if abs(h - avg_height) / avg_height < 0.2
        )

        return (
            len(x_clusters) >= 2 and
            similar_height_count / len(heights) >= 0.7
        )

    def _classify_structure(
        self,
        line_analysis: LineAnalysis,
        rect_analysis: RectangleAnalysis,
        region: Optional[BoundingBox],
    ) -> StructureAnalysis:
        """
        Determine structure type from analyzed elements.

        Classification priority:
        1. Bordered: Has grid pattern with many intersections
        2. Semi-bordered: Has horizontal lines but few/no verticals
        3. Shaded: Has row/column shading pattern
        4. Unbordered: No clear visual structure

        Args:
            line_analysis: Analyzed line data
            rect_analysis: Analyzed rectangle data
            region: Optional region being analyzed

        Returns:
            StructureAnalysis with final classification
        """
        analysis = StructureAnalysis(
            line_analysis=line_analysis,
            rectangle_analysis=rect_analysis,
            bounds=region,
        )

        h_count = len(line_analysis.horizontal_lines)
        v_count = len(line_analysis.vertical_lines)

        # Check for bordered (full grid)
        if line_analysis.has_grid_pattern:
            analysis.structure_type = StructureType.BORDERED
            analysis.confidence = min(0.9, 0.5 + len(line_analysis.grid_intersections) * 0.02)
            analysis.evidence.append(
                f"Grid pattern detected: {h_count} horizontal, {v_count} vertical lines, "
                f"{len(line_analysis.grid_intersections)} intersections"
            )
            return analysis

        # Check for semi-bordered (horizontal lines only)
        if h_count >= MIN_HORIZONTAL_LINES and v_count < MIN_VERTICAL_LINES:
            analysis.structure_type = StructureType.SEMI_BORDERED
            analysis.confidence = min(0.85, 0.5 + h_count * 0.05)
            analysis.evidence.append(
                f"Semi-bordered: {h_count} horizontal lines, {v_count} vertical lines"
            )
            return analysis

        # Check for shaded
        if rect_analysis.row_shading_pattern or rect_analysis.column_shading_pattern:
            analysis.structure_type = StructureType.SHADED
            analysis.confidence = 0.75
            if rect_analysis.row_shading_pattern:
                analysis.evidence.append("Row shading pattern detected")
            if rect_analysis.column_shading_pattern:
                analysis.evidence.append("Column shading pattern detected")
            return analysis

        # Check for weak bordered (some lines but no clear pattern)
        if h_count >= 1 or v_count >= 1:
            # Could be semi-bordered with minimal lines
            if h_count >= 1:
                analysis.structure_type = StructureType.SEMI_BORDERED
                analysis.confidence = 0.5
                analysis.evidence.append(
                    f"Weak structure: {h_count} horizontal, {v_count} vertical lines"
                )
                return analysis

        # Default to unbordered
        analysis.structure_type = StructureType.UNBORDERED
        analysis.confidence = 0.6
        analysis.evidence.append("No significant lines or shading detected")

        return analysis


# =============================================================================
# Convenience Functions
# =============================================================================

def classify_page_structure(
    pdf_document: Any,
    page_number: int,
    region: Optional[BoundingBox] = None,
) -> StructureType:
    """
    Convenience function to classify page structure.

    Args:
        pdf_document: PDFDocument instance
        page_number: Page number to analyze
        region: Optional region to analyze

    Returns:
        StructureType classification

    Example:
        >>> structure = classify_page_structure(pdf, 0)
        >>> print(structure.value)
        'bordered'
    """
    classifier = StructureClassifier()
    analysis = classifier.classify_page(pdf_document, page_number, region)
    return analysis.structure_type
