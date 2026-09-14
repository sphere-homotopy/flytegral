import math

import pytest
from pydantic import ValidationError

from fly_window.env.config import DynamicsConfig, RewardConfig, RoomConfig, SensorConfig
from fly_window.env.types import FlyState


def test_environment_configs_have_exact_v1_defaults():
    room = RoomConfig()
    dynamics = DynamicsConfig()
    sensor = SensorConfig()
    reward = RewardConfig()

    assert room.width == 10.0
    assert room.height == 6.0
    assert room.window_center_y == 3.0
    assert room.window_width == 1.5

    assert dynamics.dt == 0.05
    assert dynamics.max_speed == 3.0
    assert dynamics.max_turn_rate == math.pi
    assert dynamics.acceleration == 2.0
    assert dynamics.drag == 0.6

    assert sensor.ray_count == 16
    assert sensor.fov_radians == math.pi
    assert sensor.ambient == 0.02

    assert reward.progress_scale == 0.5
    assert reward.collision_penalty == -0.1
    assert reward.step_penalty == -0.002
    assert reward.success_reward == 5.0
    assert reward.max_steps == 400


def test_room_rejects_nonpositive_dimensions_and_window_outside_wall():
    with pytest.raises(ValidationError):
        RoomConfig(width=0.0)
    with pytest.raises(ValidationError):
        RoomConfig(height=-1.0)
    with pytest.raises(ValidationError):
        RoomConfig(window_width=0.0)
    with pytest.raises(ValidationError):
        RoomConfig(window_center_y=0.4, window_width=1.0)
    with pytest.raises(ValidationError):
        RoomConfig(window_center_y=5.8, window_width=1.0)


def test_dynamics_sensor_and_reward_validate_required_positive_values():
    with pytest.raises(ValidationError):
        DynamicsConfig(dt=0.0)
    with pytest.raises(ValidationError):
        DynamicsConfig(max_speed=0.0)
    with pytest.raises(ValidationError):
        SensorConfig(ray_count=1)
    with pytest.raises(ValidationError):
        RewardConfig(max_steps=0)


def test_configs_and_state_are_immutable():
    room = RoomConfig()
    state = FlyState(x=1.0, y=2.0, heading=0.25, speed=0.5, step_count=3)

    with pytest.raises(ValidationError):
        room.width = 12.0
    with pytest.raises(Exception):
        state.x = 9.0

    assert state == FlyState(x=1.0, y=2.0, heading=0.25, speed=0.5, step_count=3)
