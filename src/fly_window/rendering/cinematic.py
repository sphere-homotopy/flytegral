from __future__ import annotations

import math


ROOM_WIDTH = 10.0
ROOM_HEIGHT = 6.0
WORLD_WIDTH = 1.0
WORLD_HEIGHT = 0.6
FLIGHT_Z = 0.12


def env_to_world(x: float, y: float) -> tuple[float, float, float]:
    """Map the 10x6 simulation room into a compact MuJoCo world."""
    world_x = (float(x) / ROOM_WIDTH - 0.5) * WORLD_WIDTH
    world_y = (float(y) / ROOM_HEIGHT - 0.5) * WORLD_HEIGHT
    return (world_x, world_y, FLIGHT_Z)


def yaw_quaternion(heading: float) -> tuple[float, float, float, float]:
    """Return MuJoCo's wxyz quaternion for a yaw about the vertical z axis."""
    half = 0.5 * float(heading)
    return (math.cos(half), 0.0, 0.0, math.sin(half))
