"""
Utility Functions and Helpers.

This package provides shared utilities used across the tables module:

- geometry: Bounding box, line, and point classes with spatial operations
- clustering: Coordinate-based clustering utilities (in geometry module)
- parsing: Date and amount parsing utilities for bank statements

Example Usage:
    >>> from tables.utils.geometry import BoundingBox, Point
    >>> box = BoundingBox(x0=0, y0=0, x1=100, y1=50)
    >>> box.contains_point(Point(50, 25))
    True

    >>> from tables.utils.parsing import parse_amount, parse_date
    >>> parse_amount("₹ 1,00,000.50")
    Decimal('100000.50')
"""

from tables.utils.geometry import (
    BoundingBox,
    Point,
    Line,
    cluster_by_x_coordinate,
    cluster_by_y_coordinate,
    find_gaps_in_range,
    merge_overlapping_boxes,
)
from tables.utils.parsing import (
    parse_amount,
    parse_amount_detailed,
    parse_date,
    parse_date_detailed,
    clean_text,
    clean_description,
    is_empty_cell,
    extract_reference,
    normalize_amount,
    AmountParseResult,
    DateParseResult,
)

__all__ = [
    # Geometry
    "BoundingBox",
    "Point",
    "Line",
    "cluster_by_x_coordinate",
    "cluster_by_y_coordinate",
    "find_gaps_in_range",
    "merge_overlapping_boxes",
    # Parsing
    "parse_amount",
    "parse_amount_detailed",
    "parse_date",
    "parse_date_detailed",
    "clean_text",
    "clean_description",
    "is_empty_cell",
    "extract_reference",
    "normalize_amount",
    "AmountParseResult",
    "DateParseResult",
]
