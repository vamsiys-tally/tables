"""
Unit tests for geometry utilities.

Tests cover:
- Point operations
- Line segment classification and operations
- BoundingBox operations
- Spatial clustering functions
"""

import pytest
import math

from tables.utils.geometry import (
    Point,
    Line,
    BoundingBox,
    cluster_by_y_coordinate,
    cluster_by_x_coordinate,
    find_gaps_in_range,
    merge_overlapping_boxes,
)


class TestPoint:
    """Tests for the Point class."""

    def test_point_creation(self):
        """Test basic point creation."""
        p = Point(10, 20)
        assert p.x == 10
        assert p.y == 20

    def test_distance_to(self):
        """Test Euclidean distance calculation."""
        p1 = Point(0, 0)
        p2 = Point(3, 4)
        assert p1.distance_to(p2) == 5.0

    def test_manhattan_distance(self):
        """Test Manhattan distance calculation."""
        p1 = Point(0, 0)
        p2 = Point(3, 4)
        assert p1.manhattan_distance_to(p2) == 7.0

    def test_point_addition(self):
        """Test point addition (vector addition)."""
        p1 = Point(10, 20)
        p2 = Point(5, 10)
        result = p1 + p2
        assert result.x == 15
        assert result.y == 30

    def test_point_subtraction(self):
        """Test point subtraction."""
        p1 = Point(10, 20)
        p2 = Point(5, 10)
        result = p1 - p2
        assert result.x == 5
        assert result.y == 10

    def test_point_immutable(self):
        """Test that Point is immutable (frozen dataclass)."""
        p = Point(10, 20)
        with pytest.raises(Exception):  # FrozenInstanceError
            p.x = 100


class TestLine:
    """Tests for the Line class."""

    def test_horizontal_line(self):
        """Test horizontal line detection."""
        line = Line(x0=0, y0=100, x1=200, y1=100)
        assert line.is_horizontal is True
        assert line.is_vertical is False

    def test_vertical_line(self):
        """Test vertical line detection."""
        line = Line(x0=100, y0=0, x1=100, y1=200)
        assert line.is_horizontal is False
        assert line.is_vertical is True

    def test_diagonal_line(self):
        """Test diagonal line (neither horizontal nor vertical)."""
        line = Line(x0=0, y0=0, x1=100, y1=100)
        assert line.is_horizontal is False
        assert line.is_vertical is False

    def test_line_length(self):
        """Test line length calculation."""
        line = Line(x0=0, y0=0, x1=3, y1=4)
        assert line.length == 5.0

    def test_line_midpoint(self):
        """Test midpoint calculation."""
        line = Line(x0=0, y0=0, x1=100, y1=100)
        mid = line.midpoint
        assert mid.x == 50
        assert mid.y == 50

    def test_start_and_end_points(self):
        """Test start and end point properties."""
        line = Line(x0=10, y0=20, x1=30, y1=40)
        assert line.start == Point(10, 20)
        assert line.end == Point(30, 40)


class TestBoundingBox:
    """Tests for the BoundingBox class."""

    def test_bbox_creation(self):
        """Test basic bounding box creation."""
        box = BoundingBox(x0=0, y0=0, x1=100, y1=50)
        assert box.x0 == 0
        assert box.y0 == 0
        assert box.x1 == 100
        assert box.y1 == 50

    def test_bbox_normalization(self):
        """Test that coordinates are normalized (x0 <= x1, y0 <= y1)."""
        # Create with inverted coordinates
        box = BoundingBox(x0=100, y0=50, x1=0, y1=0)
        assert box.x0 == 0
        assert box.y0 == 0
        assert box.x1 == 100
        assert box.y1 == 50

    def test_width_height_area(self):
        """Test width, height, and area calculations."""
        box = BoundingBox(x0=0, y0=0, x1=100, y1=50)
        assert box.width == 100
        assert box.height == 50
        assert box.area == 5000

    def test_center(self):
        """Test center point calculation."""
        box = BoundingBox(x0=0, y0=0, x1=100, y1=50)
        center = box.center
        assert center.x == 50
        assert center.y == 25

    def test_contains_point(self):
        """Test point containment check."""
        box = BoundingBox(x0=0, y0=0, x1=100, y1=50)
        assert box.contains_point(Point(50, 25)) is True
        assert box.contains_point(Point(0, 0)) is True  # Edge
        assert box.contains_point(Point(100, 50)) is True  # Edge
        assert box.contains_point(Point(150, 25)) is False  # Outside

    def test_contains_box(self):
        """Test box containment check."""
        outer = BoundingBox(x0=0, y0=0, x1=100, y1=100)
        inner = BoundingBox(x0=25, y0=25, x1=75, y1=75)
        partial = BoundingBox(x0=50, y0=50, x1=150, y1=150)

        assert outer.contains_box(inner) is True
        assert outer.contains_box(partial) is False
        assert inner.contains_box(outer) is False

    def test_overlaps(self):
        """Test overlap detection."""
        box1 = BoundingBox(x0=0, y0=0, x1=100, y1=100)
        box2 = BoundingBox(x0=50, y0=50, x1=150, y1=150)  # Overlapping
        box3 = BoundingBox(x0=200, y0=0, x1=300, y1=100)  # No overlap

        assert box1.overlaps(box2) is True
        assert box1.overlaps(box3) is False
        assert box2.overlaps(box3) is False

    def test_intersection(self):
        """Test intersection calculation."""
        box1 = BoundingBox(x0=0, y0=0, x1=100, y1=100)
        box2 = BoundingBox(x0=50, y0=50, x1=150, y1=150)

        intersection = box1.intersection(box2)
        assert intersection is not None
        assert intersection.x0 == 50
        assert intersection.y0 == 50
        assert intersection.x1 == 100
        assert intersection.y1 == 100

    def test_intersection_no_overlap(self):
        """Test intersection when no overlap."""
        box1 = BoundingBox(x0=0, y0=0, x1=100, y1=100)
        box2 = BoundingBox(x0=200, y0=0, x1=300, y1=100)

        assert box1.intersection(box2) is None

    def test_iou(self):
        """Test Intersection over Union calculation."""
        box1 = BoundingBox(x0=0, y0=0, x1=100, y1=100)
        box2 = BoundingBox(x0=0, y0=0, x1=100, y1=100)  # Same box

        assert box1.intersection_over_union(box2) == 1.0

        box3 = BoundingBox(x0=50, y0=0, x1=150, y1=100)  # 50% overlap
        iou = box1.intersection_over_union(box3)
        # Intersection: 50*100=5000, Union: 100*100 + 100*100 - 5000 = 15000
        assert abs(iou - (5000 / 15000)) < 0.01

    def test_expand(self):
        """Test box expansion."""
        box = BoundingBox(x0=50, y0=50, x1=100, y1=100)
        expanded = box.expand(10)

        assert expanded.x0 == 40
        assert expanded.y0 == 40
        assert expanded.x1 == 110
        assert expanded.y1 == 110

    def test_from_points(self):
        """Test creation from point list."""
        points = [Point(10, 20), Point(50, 60), Point(30, 40)]
        box = BoundingBox.from_points(points)

        assert box.x0 == 10
        assert box.y0 == 20
        assert box.x1 == 50
        assert box.y1 == 60

    def test_from_dict(self):
        """Test creation from dictionary."""
        d = {"x0": 0, "y0": 10, "x1": 100, "y1": 50}
        box = BoundingBox.from_dict(d)

        assert box.x0 == 0
        assert box.y0 == 10
        assert box.x1 == 100
        assert box.y1 == 50

    def test_union(self):
        """Test union of multiple boxes."""
        boxes = [
            BoundingBox(x0=0, y0=0, x1=50, y1=50),
            BoundingBox(x0=25, y0=25, x1=100, y1=100),
        ]
        union = BoundingBox.union(boxes)

        assert union is not None
        assert union.x0 == 0
        assert union.y0 == 0
        assert union.x1 == 100
        assert union.y1 == 100

    def test_horizontal_alignment(self):
        """Test horizontal alignment detection."""
        box1 = BoundingBox(x0=0, y0=100, x1=50, y1=120)
        box2 = BoundingBox(x0=60, y0=102, x1=100, y1=118)  # Same row
        box3 = BoundingBox(x0=0, y0=200, x1=50, y1=220)  # Different row

        assert box1.is_horizontally_aligned(box2) is True
        assert box1.is_horizontally_aligned(box3) is False

    def test_vertical_alignment(self):
        """Test vertical alignment detection."""
        box1 = BoundingBox(x0=100, y0=0, x1=150, y1=50)
        box2 = BoundingBox(x0=102, y0=60, x1=148, y1=100)  # Same column
        box3 = BoundingBox(x0=200, y0=0, x1=250, y1=50)  # Different column

        assert box1.is_vertically_aligned(box2) is True
        assert box1.is_vertically_aligned(box3) is False


class TestClusterByYCoordinate:
    """Tests for the cluster_by_y_coordinate function."""

    def test_basic_clustering(self):
        """Test basic row clustering."""
        boxes = [
            BoundingBox(x0=0, y0=100, x1=50, y1=110),
            BoundingBox(x0=60, y0=102, x1=100, y1=112),  # Same row
            BoundingBox(x0=0, y0=50, x1=50, y1=60),  # Different row
        ]
        clusters = cluster_by_y_coordinate(boxes, tolerance=5.0)

        assert len(clusters) == 2
        # First cluster should have boxes with higher y (top row in PDF)
        assert len(clusters[0]) == 2 or len(clusters[1]) == 2

    def test_empty_input(self):
        """Test with empty input."""
        assert cluster_by_y_coordinate([]) == []

    def test_single_box(self):
        """Test with single box."""
        boxes = [BoundingBox(x0=0, y0=0, x1=100, y1=50)]
        clusters = cluster_by_y_coordinate(boxes)

        assert len(clusters) == 1
        assert len(clusters[0]) == 1


class TestClusterByXCoordinate:
    """Tests for the cluster_by_x_coordinate function."""

    def test_basic_clustering(self):
        """Test basic column clustering."""
        boxes = [
            BoundingBox(x0=0, y0=100, x1=50, y1=150),
            BoundingBox(x0=2, y0=50, x1=48, y1=90),  # Same column
            BoundingBox(x0=100, y0=100, x1=150, y1=150),  # Different column
        ]
        clusters = cluster_by_x_coordinate(boxes, tolerance=5.0)

        assert len(clusters) == 2


class TestFindGapsInRange:
    """Tests for the find_gaps_in_range function."""

    def test_basic_gaps(self):
        """Test basic gap detection."""
        values = [10, 15, 100, 105, 200]
        gaps = find_gaps_in_range(values, min_gap=50)

        assert len(gaps) == 2
        assert (15, 100) in gaps
        assert (105, 200) in gaps

    def test_no_gaps(self):
        """Test when no gaps exceed minimum."""
        values = [10, 15, 20, 25]
        gaps = find_gaps_in_range(values, min_gap=10)

        assert len(gaps) == 0

    def test_empty_input(self):
        """Test with empty input."""
        assert find_gaps_in_range([], min_gap=10) == []


class TestMergeOverlappingBoxes:
    """Tests for the merge_overlapping_boxes function."""

    def test_merge_overlapping(self):
        """Test merging overlapping boxes."""
        # Boxes with ~40% overlap (IoU = 2000/5000 = 0.4)
        boxes = [
            BoundingBox(x0=0, y0=0, x1=80, y1=50),
            BoundingBox(x0=40, y0=0, x1=100, y1=50),
        ]
        merged = merge_overlapping_boxes(boxes, overlap_threshold=0.3)

        assert len(merged) == 1
        assert merged[0].x0 == 0
        assert merged[0].x1 == 100

    def test_no_merge_separate_boxes(self):
        """Test that separate boxes are not merged."""
        boxes = [
            BoundingBox(x0=0, y0=0, x1=50, y1=50),
            BoundingBox(x0=100, y0=0, x1=150, y1=50),
        ]
        merged = merge_overlapping_boxes(boxes, overlap_threshold=0.3)

        assert len(merged) == 2

    def test_empty_input(self):
        """Test with empty input."""
        assert merge_overlapping_boxes([]) == []
