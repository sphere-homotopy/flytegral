import math

import pytest

from fly_window.env.config import RoomConfig
from fly_window.env.geometry import clamp_collision, crosses_window, ray_to_right_wall, window_interval
from fly_window.env.types import FlyState


def _state(x: float, y: float, *, speed: float = 1.0, step_count: int = 1) -> FlyState:
    return FlyState(x=x, y=y, heading=0.0, speed=speed, step_count=step_count)


def test_default_window_interval_is_centered_on_right_wall():
    assert window_interval(RoomConfig()) == pytest.approx((2.25, 3.75))


def test_ray_to_right_wall_returns_distance_and_hit_height():
    room = RoomConfig()

    assert ray_to_right_wall(5.0, 3.0, 0.0, room) == pytest.approx((5.0, 3.0))
    assert ray_to_right_wall(5.0, 0.5, 0.0, room) == pytest.approx((5.0, 0.5))
    assert ray_to_right_wall(5.0, 3.0, math.pi, room) is None


def test_crosses_window_only_when_right_boundary_crossing_is_inside_opening():
    room = RoomConfig()

    assert crosses_window((9.9, 3.0), (10.1, 3.0), room)
    assert crosses_window((9.9, 2.25), (10.1, 2.25), room)
    assert not crosses_window((9.9, 1.0), (10.1, 1.0), room)
    assert not crosses_window((8.0, 3.0), (9.0, 3.0), room)


def test_clamp_collision_clamps_solid_walls_and_damps_speed():
    room = RoomConfig()

    left, hit_left = clamp_collision(_state(0.1, 3.0), _state(-0.1, 3.0), room)
    bottom, hit_bottom = clamp_collision(_state(4.0, 0.1), _state(4.0, -0.2), room)
    top, hit_top = clamp_collision(_state(4.0, 5.9), _state(4.0, 6.2), room)
    right, hit_right = clamp_collision(_state(9.9, 1.0), _state(10.2, 1.0), room)

    assert hit_left and left.x == 0.0 and left.speed == pytest.approx(0.2)
    assert hit_bottom and bottom.y == 0.0 and bottom.speed == pytest.approx(0.2)
    assert hit_top and top.y == room.height and top.speed == pytest.approx(0.2)
    assert hit_right and right.x == room.width and right.speed == pytest.approx(0.2)


def test_clamp_collision_does_not_block_valid_window_exit():
    room = RoomConfig()
    candidate = _state(10.2, 3.0)

    result, collision = clamp_collision(_state(9.9, 3.0), candidate, room)

    assert not collision
    assert result == candidate
