"""Pure geometry helpers for axis-aligned bounding boxes and points."""

from __future__ import annotations

BBox = tuple[float, float, float, float]
Point = tuple[float, float]


def get_center_of_bbox(bbox: BBox) -> Point:
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2, (y1 + y2) / 2


def get_bbox_width(bbox: BBox) -> float:
    x1, _, x2, _ = bbox
    return x2 - x1


def get_foot_position(bbox: BBox) -> Point:
    """The point where a player's feet meet the pitch: bbox bottom-center."""
    x1, _, x2, y2 = bbox
    return (x1 + x2) / 2, y2


def measure_distance(p1: Point, p2: Point) -> float:
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5


def measure_xy_distance(p1: Point, p2: Point) -> Point:
    return p1[0] - p2[0], p1[1] - p2[1]
