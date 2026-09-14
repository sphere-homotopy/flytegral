from __future__ import annotations

import math

import numpy as np

from fly_window.env.config import RoomConfig
from fly_window.env.types import FlyState


def _wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def oracle_action(state: FlyState, room: RoomConfig) -> np.ndarray:
    """Privileged sanity controller; never use this in training."""
    desired = math.atan2(room.window_center_y - state.y, room.width - state.x)
    error = _wrap_angle(desired - state.heading)
    yaw = float(np.clip(error / (math.pi / 3.0), -1.0, 1.0))
    return np.asarray([yaw, 1.0], dtype=np.float32)
