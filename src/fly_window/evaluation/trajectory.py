from __future__ import annotations

from dataclasses import asdict

import numpy as np
import torch

from fly_window.env.config import EnvironmentConfig
from fly_window.env.core import WindowExitEnv


def _initial_frame(state) -> dict:
    return {
        "step": 0,
        **asdict(state),
        "action": None,
        "reward": 0.0,
        "collision": False,
        "done": False,
    }


def rollout_many_deterministic(
    policy,
    env_config: EnvironmentConfig,
    *,
    seeds: tuple[int, ...] | list[int],
) -> list[dict]:
    if not seeds:
        return []

    policy.eval()
    envs = [WindowExitEnv(env_config) for _ in seeds]
    observations: list[np.ndarray] = []
    frames: list[list[dict]] = []
    total_rewards = [0.0 for _ in seeds]
    results: list[dict | None] = [None for _ in seeds]
    active = list(range(len(seeds)))

    for env, seed in zip(envs, seeds, strict=True):
        observation, state = env.reset(int(seed))
        observations.append(observation)
        frames.append([_initial_frame(state)])

    with torch.no_grad():
        while active:
            observation_tensor = torch.as_tensor(
                np.stack([observations[index] for index in active]),
                dtype=torch.float32,
            )
            output = policy(observation_tensor)
            action_tensor = policy.action_from_latent(output.mean_action)
            actions = action_tensor.cpu().numpy()
            still_active: list[int] = []

            for batch_index, env_index in enumerate(active):
                action = actions[batch_index]
                result = envs[env_index].step(action)
                total_rewards[env_index] += result.reward
                done = result.terminated or result.truncated
                frames[env_index].append(
                    {
                        "step": result.state.step_count,
                        **asdict(result.state),
                        "action": [float(action[0]), float(action[1])],
                        "reward": float(result.reward),
                        "collision": bool(result.collision),
                        "done": bool(done),
                    }
                )
                observations[env_index] = result.observation
                if done:
                    results[env_index] = {
                        "seed": int(seeds[env_index]),
                        "success": bool(result.success),
                        "steps": int(result.state.step_count),
                        "total_reward": float(total_rewards[env_index]),
                        "frames": frames[env_index],
                    }
                else:
                    still_active.append(env_index)

            active = still_active

    return [result for result in results if result is not None]


def rollout_deterministic(policy, env_config: EnvironmentConfig, *, seed: int) -> dict:
    return rollout_many_deterministic(policy, env_config, seeds=(seed,))[0]
