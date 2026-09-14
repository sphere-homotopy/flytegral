import math

import numpy as np

from fly_window.env.config import RoomConfig, SensorConfig
from fly_window.env.sensor import observe_window
from fly_window.env.types import FlyState


def _state(x: float, y: float, heading: float) -> FlyState:
    return FlyState(x=x, y=y, heading=heading, speed=0.0, step_count=0)


def test_observation_has_fixed_shape_dtype_and_range():
    observation = observe_window(_state(5.0, 3.0, 0.0), RoomConfig(), SensorConfig())

    assert observation.shape == (16,)
    assert observation.dtype == np.float32
    assert np.all(observation >= 0.02)
    assert np.all(observation <= 1.0)


def test_center_rays_see_open_window_when_facing_right():
    observation = observe_window(_state(5.0, 3.0, 0.0), RoomConfig(), SensorConfig())

    assert observation[7] > 0.02
    assert observation[8] > 0.02
    assert observation.max() > observation.min()


def test_facing_left_sees_only_ambient():
    sensor = SensorConfig()
    observation = observe_window(_state(5.0, 3.0, math.pi), RoomConfig(), sensor)

    assert np.array_equal(observation, np.full(sensor.ray_count, sensor.ambient, dtype=np.float32))


def test_near_corner_ray_can_still_see_window_when_heading_toward_it():
    heading = math.atan2(3.0 - 0.5, 10.0 - 9.0)
    observation = observe_window(_state(9.0, 0.5, heading), RoomConfig(), SensorConfig())

    assert np.any(observation > 0.02)


def test_observation_is_exactly_deterministic():
    state = _state(4.25, 2.7, 0.2)
    first = observe_window(state, RoomConfig(), SensorConfig())
    second = observe_window(state, RoomConfig(), SensorConfig())

    assert np.array_equal(first, second)
