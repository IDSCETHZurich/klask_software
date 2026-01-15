"""Utility functions."""

from geometry_msgs.msg import Point


def create_point(x: float, y: float, z: float = 0.0) -> Point:
    """Create a Point message from coordinates."""
    point = Point()
    point.x = float(x)
    point.y = float(y)
    point.z = float(z)
    return point


def create_point_from_list(val: list[float]) -> Point:
    """Create a Point message from a list of coordinates."""
    if val is None or len(val) < 2:
        raise ValueError("Input list must have at least two elements.")

    point = Point()
    point.x = float(val[0])
    point.y = float(val[1])
    point.z = float(val[2]) if len(val) > 2 else 0.0
    return point
