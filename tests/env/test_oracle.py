import numpy as np

from fly_window.env.config import EnvironmentConfig, RoomConfig
from fly_window.env.core import WindowExitEnv
from fly_window.env.oracle import oracle_action
from fly_window.env.types import FlyState


def test_oracle_action_is_bounded_two_channel_control():
    action = oracle_action(
        FlyState(x=4.0, y=1.0, heading=2.0, speed=0.5, step_count=12),
        RoomConfig(),
    )

    assert action.shape == (2,)
    assert action.dtype == np.float32
    assert -1.0 <= action[0] <= 1.0
    assert action[1] == 1.0


def test_oracle_solves_at_least_95_of_100_seeded_rooms():
    successes = 0

    for seed in range(100):
        env = WindowExitEnv(EnvironmentConfig())
        _, state = env.reset(seed=seed)

        for _ in range(env.config.reward.max_steps):
            result = env.step(oracle_action(state, env.config.room))
            state = result.state
            if result.success:
                successes += 1
                break
            if result.truncated:
                break

    assert successes >= 95, f"oracle succeeded on only {successes}/100 seeds"
