from __future__ import annotations

import numpy as np

from fly_window.env.config import RoomConfig, SensorConfig
from fly_window.env.geometry import ray_to_right_wall, window_interval
from fly_window.env.types import FlyState


def observe_window(
    state: FlyState,
    room: RoomConfig,
    sensor: SensorConfig,
) -> np.ndarray:
    offsets = np.linspace(
        -sensor.fov_radians / 2.0,
        sensor.fov_radians / 2.0,
        sensor.ray_count,
        dtype=np.float64,
    )
    observation = np.full(sensor.ray_count, sensor.ambient, dtype=np.float32)
    lower, upper = window_interval(room)

    for index, offset in enumerate(offsets):
        hit = ray_to_right_wall(state.x, state.y, state.heading + float(offset), room)
        if hit is None:
            continue

        distance, y_hit = hit
        if not lower <= y_hit <= upper:
            continue

        brightness = 1.0 / (1.0 + 0.05 * distance**2)
        observation[index] = np.float32(np.clip(brightness, sensor.ambient, 1.0))

    return observation
