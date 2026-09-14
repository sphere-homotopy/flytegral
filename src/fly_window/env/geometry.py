from __future__ import annotations

import math
from dataclasses import replace

from fly_window.env.config import RoomConfig
from fly_window.env.types import FlyState


_EPS = 1e-9
_COLLISION_SPEED_FACTOR = 0.2


def window_interval(room: RoomConfig) -> tuple[float, float]:
    half_width = room.window_width / 2.0
    return room.window_center_y - half_width, room.window_center_y + half_width


def ray_to_right_wall(
    x: float,
    y: float,
    angle: float,
    room: RoomConfig,
) -> tuple[float, float] | None:
    dx = math.cos(angle)
    if dx <= _EPS:
        return None

    distance = (room.width - x) / dx
    if distance < 0:
        return None

    y_hit = y + distance * math.sin(angle)
    return distance, y_hit


def crosses_window(
    prev_xy: tuple[float, float],
    next_xy: tuple[float, float],
    room: RoomConfig,
) -> bool:
    prev_x, prev_y = prev_xy
    next_x, next_y = next_xy
    dx = next_x - prev_x

    if dx <= 0 or prev_x >= room.width or next_x < room.width:
        return False

    fraction = (room.width - prev_x) / dx
    if fraction < 0 or fraction > 1:
        return False

    y_cross = prev_y + fraction * (next_y - prev_y)
    lower, upper = window_interval(room)
    return lower <= y_cross <= upper


def clamp_collision(
    prev: FlyState,
    candidate: FlyState,
    room: RoomConfig,
) -> tuple[FlyState, bool]:
    if crosses_window((prev.x, prev.y), (candidate.x, candidate.y), room):
        return candidate, False

    hit_left = candidate.x <= 0.0
    hit_right = candidate.x >= room.width
    hit_bottom = candidate.y <= 0.0
    hit_top = candidate.y >= room.height
    collision = hit_left or hit_right or hit_bottom or hit_top

    if not collision:
        return candidate, False

    return (
        replace(
            candidate,
            x=min(max(candidate.x, 0.0), room.width),
            y=min(max(candidate.y, 0.0), room.height),
            speed=candidate.speed * _COLLISION_SPEED_FACTOR,
        ),
        True,
    )
