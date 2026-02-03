"""
Geometry Utilities for PDF Coordinate Processing.

This module provides data structures and utilities for working with
PDF coordinates, bounding boxes, lines, and spatial relationships.

PDF Coordinate System:
    - Origin (0, 0) is at the bottom-left corner
    - X increases to the right
    - Y increases upward
    - All coordinates are in PDF points (72 points = 1 inch)

Main Components:
    - BoundingBox: Rectangle defined by (x0, y0, x1, y1)
    - Line: Line segment with start and end points
    - Point: 2D coordinate
    - Spatial relationship functions (overlap, containment, intersection)

Example Usage:
    >>> from tables.utils.geometry import BoundingBox
    >>>
    >>> box1 = BoundingBox(x0=0, y0=0, x1=100, y1=50)
    >>> box2 = BoundingBox(x0=50, y0=25, x1=150, y1=75)
    >>> box1.overlaps(box2)
    True
    >>> box1.intersection(box2)
    BoundingBox(x0=50, y0=25, x1=100, y1=50)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Iterator
import math


# =============================================================================
# Point Class
# =============================================================================

@dataclass(frozen=True)
class Point:
    """
    A 2D point in PDF coordinate space.

    Attributes:
        x: X-coordinate (horizontal position)
        y: Y-coordinate (vertical position)

    Example:
        >>> p1 = Point(10, 20)
        >>> p2 = Point(30, 40)
        >>> p1.distance_to(p2)
        28.284271247461902
    """

    x: float
    y: float

    def distance_to(self, other: Point) -> float:
        """
        Calculate Euclidean distance to another point.

        Args:
            other: Target point

        Returns:
            Distance in PDF points
        """
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)

    def manhattan_distance_to(self, other: Point) -> float:
        """
        Calculate Manhattan (taxicab) distance to another point.

        Useful for grid-aligned comparisons.

        Args:
            other: Target point

        Returns:
            Manhattan distance in PDF points
        """
        return abs(self.x - other.x) + abs(self.y - other.y)

    def __add__(self, other: Point) -> Point:
        """Add two points (vector addition)."""
        return Point(self.x + other.x, self.y + other.y)

    def __sub__(self, other: Point) -> Point:
        """Subtract two points (vector subtraction)."""
        return Point(self.x - other.x, self.y - other.y)


# =============================================================================
# Line Class
# =============================================================================

@dataclass(frozen=True)
class Line:
    """
    A line segment defined by start and end points.

    Attributes:
        x0: Start X-coordinate
        y0: Start Y-coordinate
        x1: End X-coordinate
        y1: End Y-coordinate

    Properties:
        is_horizontal: True if line is approximately horizontal
        is_vertical: True if line is approximately vertical
        length: Length of the line segment
        midpoint: Center point of the line

    Example:
        >>> line = Line(0, 0, 100, 0)
        >>> line.is_horizontal
        True
        >>> line.length
        100.0
    """

    x0: float
    y0: float
    x1: float
    y1: float

    # Tolerance for determining horizontal/vertical alignment (in PDF points)
    ALIGNMENT_TOLERANCE: float = field(default=2.0, repr=False, compare=False)

    def __post_init__(self):
        """Normalize line direction if needed (ensure x0 <= x1 or y0 <= y1)."""
        # We use frozen=True, so we can't modify after init
        # Normalization is handled in factory methods if needed
        pass

    @property
    def start(self) -> Point:
        """Get the start point of the line."""
        return Point(self.x0, self.y0)

    @property
    def end(self) -> Point:
        """Get the end point of the line."""
        return Point(self.x1, self.y1)

    @property
    def length(self) -> float:
        """Calculate the length of the line segment."""
        return math.sqrt((self.x1 - self.x0) ** 2 + (self.y1 - self.y0) ** 2)

    @property
    def midpoint(self) -> Point:
        """Get the midpoint of the line."""
        return Point((self.x0 + self.x1) / 2, (self.y0 + self.y1) / 2)

    @property
    def is_horizontal(self) -> bool:
        """
        Check if line is approximately horizontal.

        Uses ALIGNMENT_TOLERANCE for fuzzy comparison.
        """
        return abs(self.y1 - self.y0) <= 2.0  # Use literal to avoid dataclass issues

    @property
    def is_vertical(self) -> bool:
        """
        Check if line is approximately vertical.

        Uses ALIGNMENT_TOLERANCE for fuzzy comparison.
        """
        return abs(self.x1 - self.x0) <= 2.0  # Use literal to avoid dataclass issues

    def overlaps_horizontally(self, other: Line, tolerance: float = 2.0) -> bool:
        """
        Check if two horizontal lines overlap in x-range.

        Args:
            other: Another line to compare
            tolerance: Tolerance for y-coordinate matching

        Returns:
            True if lines overlap horizontally
        """
        if not (self.is_horizontal and other.is_horizontal):
            return False

        # Check if y-coordinates are close
        if abs(self.y0 - other.y0) > tolerance:
            return False

        # Check x-range overlap
        self_min_x = min(self.x0, self.x1)
        self_max_x = max(self.x0, self.x1)
        other_min_x = min(other.x0, other.x1)
        other_max_x = max(other.x0, other.x1)

        return self_min_x <= other_max_x and other_min_x <= self_max_x

    def overlaps_vertically(self, other: Line, tolerance: float = 2.0) -> bool:
        """
        Check if two vertical lines overlap in y-range.

        Args:
            other: Another line to compare
            tolerance: Tolerance for x-coordinate matching

        Returns:
            True if lines overlap vertically
        """
        if not (self.is_vertical and other.is_vertical):
            return False

        # Check if x-coordinates are close
        if abs(self.x0 - other.x0) > tolerance:
            return False

        # Check y-range overlap
        self_min_y = min(self.y0, self.y1)
        self_max_y = max(self.y0, self.y1)
        other_min_y = min(other.y0, other.y1)
        other_max_y = max(other.y0, other.y1)

        return self_min_y <= other_max_y and other_min_y <= self_max_y


# =============================================================================
# BoundingBox Class
# =============================================================================

@dataclass
class BoundingBox:
    """
    A rectangular bounding box in PDF coordinate space.

    The box is defined by its bottom-left (x0, y0) and top-right (x1, y1) corners.
    In PDF coordinates, y increases upward, so y0 < y1.

    Attributes:
        x0: Left edge X-coordinate
        y0: Bottom edge Y-coordinate
        x1: Right edge X-coordinate
        y1: Top edge Y-coordinate

    Properties:
        width: Width of the box
        height: Height of the box
        area: Area of the box
        center: Center point of the box

    Example:
        >>> box = BoundingBox(x0=0, y0=0, x1=100, y1=50)
        >>> box.width
        100
        >>> box.height
        50
        >>> box.area
        5000
        >>> box.contains_point(Point(50, 25))
        True
    """

    x0: float
    y0: float
    x1: float
    y1: float

    def __post_init__(self):
        """Ensure coordinates are normalized (x0 <= x1, y0 <= y1)."""
        if self.x0 > self.x1:
            self.x0, self.x1 = self.x1, self.x0
        if self.y0 > self.y1:
            self.y0, self.y1 = self.y1, self.y0

    @classmethod
    def from_points(cls, points: list[Point]) -> BoundingBox:
        """
        Create a bounding box that encloses all given points.

        Args:
            points: List of points to enclose

        Returns:
            BoundingBox enclosing all points

        Raises:
            ValueError: If points list is empty

        Example:
            >>> points = [Point(10, 20), Point(50, 60), Point(30, 40)]
            >>> box = BoundingBox.from_points(points)
            >>> box
            BoundingBox(x0=10, y0=20, x1=50, y1=60)
        """
        if not points:
            raise ValueError("Cannot create bounding box from empty point list")

        x_coords = [p.x for p in points]
        y_coords = [p.y for p in points]

        return cls(
            x0=min(x_coords),
            y0=min(y_coords),
            x1=max(x_coords),
            y1=max(y_coords),
        )

    @classmethod
    def from_dict(cls, d: dict) -> BoundingBox:
        """
        Create a BoundingBox from a dictionary.

        Supports both (x0, y0, x1, y1) and (x0, top, x1, bottom) formats
        commonly found in PDF libraries.

        Args:
            d: Dictionary with coordinate keys

        Returns:
            BoundingBox instance

        Example:
            >>> box = BoundingBox.from_dict({"x0": 0, "y0": 0, "x1": 100, "y1": 50})
        """
        # Handle pdfplumber's "top" and "bottom" naming
        y0 = d.get("y0", d.get("top", 0))
        y1 = d.get("y1", d.get("bottom", 0))

        return cls(
            x0=d.get("x0", 0),
            y0=y0,
            x1=d.get("x1", 0),
            y1=y1,
        )

    @classmethod
    def union(cls, boxes: list[BoundingBox]) -> Optional[BoundingBox]:
        """
        Create a bounding box that encloses all given boxes.

        Args:
            boxes: List of bounding boxes

        Returns:
            BoundingBox enclosing all boxes, or None if list is empty

        Example:
            >>> box1 = BoundingBox(0, 0, 50, 50)
            >>> box2 = BoundingBox(25, 25, 100, 100)
            >>> BoundingBox.union([box1, box2])
            BoundingBox(x0=0, y0=0, x1=100, y1=100)
        """
        if not boxes:
            return None

        return cls(
            x0=min(b.x0 for b in boxes),
            y0=min(b.y0 for b in boxes),
            x1=max(b.x1 for b in boxes),
            y1=max(b.y1 for b in boxes),
        )

    @property
    def width(self) -> float:
        """Get the width of the bounding box."""
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        """Get the height of the bounding box."""
        return self.y1 - self.y0

    @property
    def area(self) -> float:
        """Get the area of the bounding box."""
        return self.width * self.height

    @property
    def center(self) -> Point:
        """Get the center point of the bounding box."""
        return Point(
            (self.x0 + self.x1) / 2,
            (self.y0 + self.y1) / 2,
        )

    @property
    def corners(self) -> tuple[Point, Point, Point, Point]:
        """
        Get all four corners of the bounding box.

        Returns:
            Tuple of (bottom_left, bottom_right, top_right, top_left)
        """
        return (
            Point(self.x0, self.y0),  # bottom-left
            Point(self.x1, self.y0),  # bottom-right
            Point(self.x1, self.y1),  # top-right
            Point(self.x0, self.y1),  # top-left
        )

    def contains_point(self, point: Point, tolerance: float = 0.0) -> bool:
        """
        Check if a point is inside (or on the edge of) this bounding box.

        Args:
            point: Point to check
            tolerance: Expand box by this amount for fuzzy containment

        Returns:
            True if point is inside the box
        """
        return (
            self.x0 - tolerance <= point.x <= self.x1 + tolerance
            and self.y0 - tolerance <= point.y <= self.y1 + tolerance
        )

    def contains_box(self, other: BoundingBox, tolerance: float = 0.0) -> bool:
        """
        Check if another bounding box is completely inside this one.

        Args:
            other: Box to check for containment
            tolerance: Expand this box by this amount for fuzzy containment

        Returns:
            True if other is completely inside this box
        """
        return (
            self.x0 - tolerance <= other.x0
            and other.x1 <= self.x1 + tolerance
            and self.y0 - tolerance <= other.y0
            and other.y1 <= self.y1 + tolerance
        )

    def overlaps(self, other: BoundingBox) -> bool:
        """
        Check if this bounding box overlaps with another.

        Two boxes overlap if they share any interior points (not just edges).

        Args:
            other: Box to check for overlap

        Returns:
            True if boxes overlap
        """
        # Boxes don't overlap if one is completely to the left/right/above/below
        if self.x1 <= other.x0 or other.x1 <= self.x0:
            return False
        if self.y1 <= other.y0 or other.y1 <= self.y0:
            return False
        return True

    def intersection(self, other: BoundingBox) -> Optional[BoundingBox]:
        """
        Get the intersection of this bounding box with another.

        Args:
            other: Box to intersect with

        Returns:
            BoundingBox representing the intersection, or None if no overlap
        """
        if not self.overlaps(other):
            return None

        return BoundingBox(
            x0=max(self.x0, other.x0),
            y0=max(self.y0, other.y0),
            x1=min(self.x1, other.x1),
            y1=min(self.y1, other.y1),
        )

    def intersection_over_union(self, other: BoundingBox) -> float:
        """
        Calculate the Intersection over Union (IoU) with another box.

        IoU is a measure of overlap commonly used in object detection.
        Range is [0, 1] where 1 means perfect overlap.

        Args:
            other: Box to compare

        Returns:
            IoU score between 0 and 1
        """
        intersection = self.intersection(other)
        if intersection is None:
            return 0.0

        intersection_area = intersection.area
        union_area = self.area + other.area - intersection_area

        if union_area == 0:
            return 0.0

        return intersection_area / union_area

    def expand(self, margin: float) -> BoundingBox:
        """
        Create a new bounding box expanded by the given margin.

        Args:
            margin: Amount to expand in all directions (can be negative to shrink)

        Returns:
            New expanded BoundingBox
        """
        return BoundingBox(
            x0=self.x0 - margin,
            y0=self.y0 - margin,
            x1=self.x1 + margin,
            y1=self.y1 + margin,
        )

    def expand_to_contain(self, point: Point) -> BoundingBox:
        """
        Create a new bounding box expanded to contain the given point.

        Args:
            point: Point that must be contained

        Returns:
            New BoundingBox that contains both original box and point
        """
        return BoundingBox(
            x0=min(self.x0, point.x),
            y0=min(self.y0, point.y),
            x1=max(self.x1, point.x),
            y1=max(self.y1, point.y),
        )

    def vertical_overlap(self, other: BoundingBox) -> float:
        """
        Calculate the vertical overlap between two boxes.

        Args:
            other: Box to compare

        Returns:
            Overlap amount in PDF points (0 if no overlap)
        """
        overlap_y0 = max(self.y0, other.y0)
        overlap_y1 = min(self.y1, other.y1)
        return max(0, overlap_y1 - overlap_y0)

    def horizontal_overlap(self, other: BoundingBox) -> float:
        """
        Calculate the horizontal overlap between two boxes.

        Args:
            other: Box to compare

        Returns:
            Overlap amount in PDF points (0 if no overlap)
        """
        overlap_x0 = max(self.x0, other.x0)
        overlap_x1 = min(self.x1, other.x1)
        return max(0, overlap_x1 - overlap_x0)

    def is_horizontally_aligned(self, other: BoundingBox, tolerance: float = 5.0) -> bool:
        """
        Check if two boxes are horizontally aligned (same row).

        Boxes are considered aligned if their y-coordinates overlap significantly.

        Args:
            other: Box to compare
            tolerance: Maximum allowed gap in y-coordinates

        Returns:
            True if boxes appear to be in the same row
        """
        # Check if vertical centers are within tolerance
        self_center_y = (self.y0 + self.y1) / 2
        other_center_y = (other.y0 + other.y1) / 2

        # Also check for significant vertical overlap
        overlap = self.vertical_overlap(other)
        min_height = min(self.height, other.height)

        return (
            abs(self_center_y - other_center_y) <= tolerance
            or (min_height > 0 and overlap / min_height >= 0.5)
        )

    def is_vertically_aligned(self, other: BoundingBox, tolerance: float = 5.0) -> bool:
        """
        Check if two boxes are vertically aligned (same column).

        Boxes are considered aligned if their x-coordinates overlap significantly.

        Args:
            other: Box to compare
            tolerance: Maximum allowed gap in x-coordinates

        Returns:
            True if boxes appear to be in the same column
        """
        # Check if horizontal centers are within tolerance
        self_center_x = (self.x0 + self.x1) / 2
        other_center_x = (other.x0 + other.x1) / 2

        # Also check for significant horizontal overlap
        overlap = self.horizontal_overlap(other)
        min_width = min(self.width, other.width)

        return (
            abs(self_center_x - other_center_x) <= tolerance
            or (min_width > 0 and overlap / min_width >= 0.5)
        )

    def to_dict(self) -> dict[str, float]:
        """
        Convert to dictionary.

        Returns:
            Dictionary with x0, y0, x1, y1 keys
        """
        return {
            "x0": self.x0,
            "y0": self.y0,
            "x1": self.x1,
            "y1": self.y1,
        }

    def to_tuple(self) -> tuple[float, float, float, float]:
        """
        Convert to tuple.

        Returns:
            Tuple of (x0, y0, x1, y1)
        """
        return (self.x0, self.y0, self.x1, self.y1)


# =============================================================================
# Spatial Clustering Utilities
# =============================================================================

def cluster_by_y_coordinate(
    boxes: list[BoundingBox],
    tolerance: float = 5.0,
) -> list[list[BoundingBox]]:
    """
    Cluster bounding boxes by their y-coordinate (row clustering).

    Groups boxes that appear to be on the same horizontal row.

    Args:
        boxes: List of bounding boxes to cluster
        tolerance: Maximum y-difference to consider same row

    Returns:
        List of clusters, each cluster is a list of boxes on the same row
        Clusters are sorted by y-coordinate (top to bottom in PDF coordinates)

    Example:
        >>> boxes = [BoundingBox(0, 100, 50, 110), BoundingBox(60, 102, 100, 112),
        ...          BoundingBox(0, 50, 50, 60)]
        >>> clusters = cluster_by_y_coordinate(boxes)
        >>> len(clusters)
        2  # Two rows
    """
    if not boxes:
        return []

    # Sort by y-coordinate (descending - top to bottom in PDF)
    sorted_boxes = sorted(boxes, key=lambda b: -(b.y0 + b.y1) / 2)

    clusters: list[list[BoundingBox]] = []
    current_cluster: list[BoundingBox] = []
    current_y: Optional[float] = None

    for box in sorted_boxes:
        box_center_y = (box.y0 + box.y1) / 2

        if current_y is None or abs(box_center_y - current_y) <= tolerance:
            current_cluster.append(box)
            if current_y is None:
                current_y = box_center_y
            else:
                # Update running average of y-coordinate
                current_y = (current_y * (len(current_cluster) - 1) + box_center_y) / len(current_cluster)
        else:
            if current_cluster:
                clusters.append(current_cluster)
            current_cluster = [box]
            current_y = box_center_y

    if current_cluster:
        clusters.append(current_cluster)

    # Sort boxes within each cluster by x-coordinate (left to right)
    for cluster in clusters:
        cluster.sort(key=lambda b: b.x0)

    return clusters


def cluster_by_x_coordinate(
    boxes: list[BoundingBox],
    tolerance: float = 5.0,
) -> list[list[BoundingBox]]:
    """
    Cluster bounding boxes by their x-coordinate (column clustering).

    Groups boxes that appear to be in the same vertical column.

    Args:
        boxes: List of bounding boxes to cluster
        tolerance: Maximum x-difference to consider same column

    Returns:
        List of clusters, each cluster is a list of boxes in the same column
        Clusters are sorted by x-coordinate (left to right)

    Example:
        >>> boxes = [BoundingBox(0, 100, 50, 110), BoundingBox(2, 50, 48, 60),
        ...          BoundingBox(100, 100, 150, 110)]
        >>> clusters = cluster_by_x_coordinate(boxes)
        >>> len(clusters)
        2  # Two columns
    """
    if not boxes:
        return []

    # Sort by x-coordinate
    sorted_boxes = sorted(boxes, key=lambda b: (b.x0 + b.x1) / 2)

    clusters: list[list[BoundingBox]] = []
    current_cluster: list[BoundingBox] = []
    current_x: Optional[float] = None

    for box in sorted_boxes:
        box_center_x = (box.x0 + box.x1) / 2

        if current_x is None or abs(box_center_x - current_x) <= tolerance:
            current_cluster.append(box)
            if current_x is None:
                current_x = box_center_x
            else:
                # Update running average of x-coordinate
                current_x = (current_x * (len(current_cluster) - 1) + box_center_x) / len(current_cluster)
        else:
            if current_cluster:
                clusters.append(current_cluster)
            current_cluster = [box]
            current_x = box_center_x

    if current_cluster:
        clusters.append(current_cluster)

    # Sort boxes within each cluster by y-coordinate (top to bottom)
    for cluster in clusters:
        cluster.sort(key=lambda b: -(b.y0 + b.y1) / 2)

    return clusters


def find_gaps_in_range(
    values: list[float],
    min_gap: float,
) -> list[tuple[float, float]]:
    """
    Find gaps in a sorted list of values.

    Useful for detecting column separators from x-coordinates
    or row separators from y-coordinates.

    Args:
        values: Sorted list of coordinate values
        min_gap: Minimum gap size to report

    Returns:
        List of (start, end) tuples representing gaps

    Example:
        >>> values = [10, 15, 100, 105, 200]
        >>> find_gaps_in_range(values, min_gap=50)
        [(15, 100), (105, 200)]
    """
    if len(values) < 2:
        return []

    sorted_values = sorted(values)
    gaps: list[tuple[float, float]] = []

    for i in range(len(sorted_values) - 1):
        gap_size = sorted_values[i + 1] - sorted_values[i]
        if gap_size >= min_gap:
            gaps.append((sorted_values[i], sorted_values[i + 1]))

    return gaps


def merge_overlapping_boxes(
    boxes: list[BoundingBox],
    overlap_threshold: float = 0.5,
) -> list[BoundingBox]:
    """
    Merge bounding boxes that overlap significantly.

    Args:
        boxes: List of bounding boxes to merge
        overlap_threshold: Minimum IoU to trigger merge

    Returns:
        List of merged bounding boxes

    Example:
        >>> boxes = [BoundingBox(0, 0, 60, 50), BoundingBox(40, 0, 100, 50)]
        >>> merged = merge_overlapping_boxes(boxes, overlap_threshold=0.3)
        >>> len(merged)
        1
    """
    if not boxes:
        return []

    # Work with a copy
    remaining = list(boxes)
    merged: list[BoundingBox] = []

    while remaining:
        current = remaining.pop(0)
        merged_any = True

        while merged_any:
            merged_any = False
            new_remaining: list[BoundingBox] = []

            for other in remaining:
                if current.intersection_over_union(other) >= overlap_threshold:
                    # Merge boxes
                    current = BoundingBox(
                        x0=min(current.x0, other.x0),
                        y0=min(current.y0, other.y0),
                        x1=max(current.x1, other.x1),
                        y1=max(current.y1, other.y1),
                    )
                    merged_any = True
                else:
                    new_remaining.append(other)

            remaining = new_remaining

        merged.append(current)

    return merged
