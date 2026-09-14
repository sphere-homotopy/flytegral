import math

import numpy as np
import pytest

from fly_window.env.config import EnvironmentConfig, RewardConfig
from fly_window.env.core import WindowExitEnv
from fly_window.env.types import FlyState


def _place(env: WindowExitEnv, state: FlyState) -> None:
    env._state = state


def test_reset_is_reproducible_and_stays_inside_spawn_bounds():
    env = WindowExitEnv(EnvironmentConfig())

    first_observation, first_state = env.reset(seed=123)
    second_observation, second_state = env.reset(seed=123)
    _, different_state = env.reset(seed=124)

    assert first_state == second_state
    assert np.array_equal(first_observation, second_observation)
    assert first_state != different_state
    assert 1.0 <= first_state.x <= 8.5
    assert 0.75 <= first_state.y <= 5.25
    assert -math.pi <= first_state.heading <= math.pi
    assert first_state.speed == 0.0
    assert first_state.step_count == 0
    assert first_observation.shape == (16,)


def test_step_implements_pinned_v1_dynamics():
    env = WindowExitEnv(EnvironmentConfig())
    env.reset(seed=0)
    _place(env, FlyState(x=5.0, y=3.0, heading=0.0, speed=0.0, step_count=0))

    result = env.step(np.array([0.0, 1.0], dtype=np.float32))

    assert result.state.heading == pytest.approx(0.0)
    assert result.state.speed == pytest.approx(0.1)
    assert result.state.x == pytest.approx(5.005)
    assert result.state.y == pytest.approx(3.0)
    assert result.state.step_count == 1
    assert not result.terminated
    assert not result.truncated


def test_actions_are_clipped_before_dynamics():
    config = EnvironmentConfig()
    extreme = WindowExitEnv(config)
    clipped = WindowExitEnv(config)
    initial = FlyState(x=5.0, y=3.0, heading=0.0, speed=0.5, step_count=0)

    extreme.reset(seed=1)
    clipped.reset(seed=1)
    _place(extreme, initial)
    _place(clipped, initial)

    extreme_result = extreme.step(np.array([9.0, 9.0]))
    clipped_result = clipped.step(np.array([1.0, 1.0]))

    assert extreme_result.state == clipped_result.state


def test_moving_toward_window_has_more_reward_than_moving_away():
    toward = WindowExitEnv(EnvironmentConfig())
    away = WindowExitEnv(EnvironmentConfig())
    toward.reset(seed=0)
    away.reset(seed=0)
    _place(toward, FlyState(x=5.0, y=3.0, heading=0.0, speed=1.0, step_count=0))
    _place(away, FlyState(x=5.0, y=3.0, heading=math.pi, speed=1.0, step_count=0))

    toward_result = toward.step(np.array([0.0, 0.0]))
    away_result = away.step(np.array([0.0, 0.0]))

    assert toward_result.reward > away_result.reward


def test_collision_penalty_contributes_exactly_minus_point_one():
    no_penalty = EnvironmentConfig(
        reward=RewardConfig(
            progress_scale=0.0,
            collision_penalty=0.0,
            step_penalty=0.0,
            success_reward=0.0,
        )
    )
    with_penalty = EnvironmentConfig(
        reward=RewardConfig(
            progress_scale=0.0,
            collision_penalty=-0.1,
            step_penalty=0.0,
            success_reward=0.0,
        )
    )
    baseline = WindowExitEnv(no_penalty)
    penalized = WindowExitEnv(with_penalty)
    initial = FlyState(x=0.01, y=3.0, heading=math.pi, speed=1.0, step_count=0)
    baseline.reset(seed=0)
    penalized.reset(seed=0)
    _place(baseline, initial)
    _place(penalized, initial)

    baseline_result = baseline.step(np.array([0.0, 0.0]))
    penalized_result = penalized.step(np.array([0.0, 0.0]))

    assert baseline_result.collision
    assert penalized_result.collision
    assert penalized_result.reward - baseline_result.reward == pytest.approx(-0.1)


def test_success_reward_contributes_exactly_plus_five_and_terminates():
    no_bonus = EnvironmentConfig(
        reward=RewardConfig(
            progress_scale=0.0,
            collision_penalty=0.0,
            step_penalty=0.0,
            success_reward=0.0,
        )
    )
    with_bonus = EnvironmentConfig(
        reward=RewardConfig(
            progress_scale=0.0,
            collision_penalty=0.0,
            step_penalty=0.0,
            success_reward=5.0,
        )
    )
    baseline = WindowExitEnv(no_bonus)
    rewarded = WindowExitEnv(with_bonus)
    initial = FlyState(x=9.99, y=3.0, heading=0.0, speed=1.0, step_count=0)
    baseline.reset(seed=0)
    rewarded.reset(seed=0)
    _place(baseline, initial)
    _place(rewarded, initial)

    baseline_result = baseline.step(np.array([0.0, 0.0]))
    rewarded_result = rewarded.step(np.array([0.0, 0.0]))

    assert rewarded_result.success
    assert rewarded_result.terminated
    assert not rewarded_result.truncated
    assert rewarded_result.reward - baseline_result.reward == pytest.approx(5.0)


def test_nonterminal_step_penalty_and_max_step_truncation():
    config = EnvironmentConfig(
        reward=RewardConfig(
            progress_scale=0.0,
            collision_penalty=0.0,
            step_penalty=-0.002,
            success_reward=0.0,
            max_steps=1,
        )
    )
    env = WindowExitEnv(config)
    env.reset(seed=0)
    _place(env, FlyState(x=5.0, y=3.0, heading=0.0, speed=0.0, step_count=0))

    result = env.step(np.array([0.0, 0.0]))

    assert result.reward == pytest.approx(-0.002)
    assert not result.terminated
    assert result.truncated
    assert not result.success
