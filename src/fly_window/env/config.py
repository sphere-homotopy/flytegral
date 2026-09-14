from __future__ import annotations

import math

from pydantic import BaseModel, ConfigDict, Field, model_validator


class _FrozenConfig(BaseModel):
    model_config = ConfigDict(frozen=True)


class RoomConfig(_FrozenConfig):
    width: float = 10.0
    height: float = 6.0
    window_center_y: float = 3.0
    window_width: float = 1.5

    @model_validator(mode="after")
    def validate_geometry(self) -> "RoomConfig":
        if self.width <= 0:
            raise ValueError("room width must be positive")
        if self.height <= 0:
            raise ValueError("room height must be positive")
        if self.window_width <= 0:
            raise ValueError("window width must be positive")
        lower = self.window_center_y - self.window_width / 2
        upper = self.window_center_y + self.window_width / 2
        if lower < 0 or upper > self.height:
            raise ValueError("window must lie entirely within the right wall")
        return self


class DynamicsConfig(_FrozenConfig):
    dt: float = 0.05
    max_speed: float = 3.0
    max_turn_rate: float = math.pi
    acceleration: float = 2.0
    drag: float = 0.6

    @model_validator(mode="after")
    def validate_dynamics(self) -> "DynamicsConfig":
        if self.dt <= 0:
            raise ValueError("dt must be positive")
        if self.max_speed <= 0:
            raise ValueError("max_speed must be positive")
        if self.max_turn_rate <= 0:
            raise ValueError("max_turn_rate must be positive")
        if self.acceleration <= 0:
            raise ValueError("acceleration must be positive")
        if self.drag < 0:
            raise ValueError("drag must be nonnegative")
        return self


class SensorConfig(_FrozenConfig):
    ray_count: int = Field(default=16, ge=2)
    fov_radians: float = math.pi
    ambient: float = 0.02

    @model_validator(mode="after")
    def validate_sensor(self) -> "SensorConfig":
        if self.fov_radians <= 0:
            raise ValueError("fov_radians must be positive")
        if not 0 <= self.ambient <= 1:
            raise ValueError("ambient must lie in [0, 1]")
        return self


class RewardConfig(_FrozenConfig):
    progress_scale: float = 0.5
    collision_penalty: float = -0.1
    step_penalty: float = -0.002
    success_reward: float = 5.0
    max_steps: int = Field(default=400, gt=0)


class EnvironmentConfig(_FrozenConfig):
    room: RoomConfig = Field(default_factory=RoomConfig)
    dynamics: DynamicsConfig = Field(default_factory=DynamicsConfig)
    sensor: SensorConfig = Field(default_factory=SensorConfig)
    reward: RewardConfig = Field(default_factory=RewardConfig)
