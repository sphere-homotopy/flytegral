from __future__ import annotations

import math

import numpy as np

from fly_window.env.config import EnvironmentConfig
from fly_window.env.geometry import clamp_collision, crosses_window
from fly_window.env.sensor import observe_window
from fly_window.env.types import FlyState, StepResult


class WindowExitEnv:
    def __init__(self, config: EnvironmentConfig):
        self.config = config
        self._state: FlyState | None = None

    @property
    def state(self) -> FlyState | None:
        return self._state

    def reset(self, seed: int) -> tuple[np.ndarray, FlyState]:
        rng = np.random.default_rng(seed)
        state = FlyState(
            x=float(rng.uniform(1.0, 8.5)),
            y=float(rng.uniform(0.75, 5.25)),
            heading=float(rng.uniform(-math.pi, math.pi)),
            speed=0.0,
            step_count=0,
        )
        self._state = state
        observation = observe_window(state, self.config.room, self.config.sensor)
        return observation, state

    def step(self, action: np.ndarray) -> StepResult:
        if self._state is None:
            raise RuntimeError("reset() must be called before step()")

        values = np.asarray(action, dtype=np.float64).reshape(-1)
        if values.shape != (2,):
            raise ValueError("action must contain exactly [yaw, thrust]")

        yaw = float(np.clip(values[0], -1.0, 1.0))
        thrust = float(np.clip(values[1], 0.0, 1.0))
        previous = self._state
        dynamics = self.config.dynamics
        room = self.config.room
        reward_config = self.config.reward

        heading = previous.heading + yaw * dynamics.max_turn_rate * dynamics.dt
        speed = previous.speed + (
            thrust * dynamics.acceleration - dynamics.drag * previous.speed
        ) * dynamics.dt
        speed = float(np.clip(speed, 0.0, dynamics.max_speed))

        candidate = FlyState(
            x=previous.x + math.cos(heading) * speed * dynamics.dt,
            y=previous.y + math.sin(heading) * speed * dynamics.dt,
            heading=heading,
            speed=speed,
            step_count=previous.step_count + 1,
        )

        success = crosses_window(
            (previous.x, previous.y),
            (candidate.x, candidate.y),
            room,
        )
        resolved, collision = clamp_collision(previous, candidate, room)

        target_x = room.width
        target_y = room.window_center_y
        previous_distance = math.hypot(target_x - previous.x, target_y - previous.y)
        current_distance = math.hypot(target_x - resolved.x, target_y - resolved.y)

        reward = reward_config.progress_scale * (previous_distance - current_distance)
        reward += reward_config.step_penalty
        if collision:
            reward += reward_config.collision_penalty
        if success:
            reward += reward_config.success_reward

        terminated = success
        truncated = (not success) and resolved.step_count >= reward_config.max_steps
        self._state = resolved
        observation = observe_window(resolved, room, self.config.sensor)

        return StepResult(
            observation=observation,
            state=resolved,
            reward=float(reward),
            terminated=terminated,
            truncated=truncated,
            success=success,
            collision=collision,
        )
