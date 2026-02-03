"""
Utility Functions and Helpers.

This package provides shared utilities used across the tables module:

- geometry: Bounding box, line, and point classes with spatial operations
- clustering: Coordinate-based clustering utilities (in geometry module)
- parsing: (Coming in Module 3) Date and amount parsing utilities

Example Usage:
    >>> from tables.utils.geometry import BoundingBox, Point
    >>> box = BoundingBox(x0=0, y0=0, x1=100, y1=50)
    >>> box.contains_point(Point(50, 25))
    True
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

__all__ = [
    "BoundingBox",
    "Point",
    "Line",
    "cluster_by_x_coordinate",
    "cluster_by_y_coordinate",
    "find_gaps_in_range",
    "merge_overlapping_boxes",
]
