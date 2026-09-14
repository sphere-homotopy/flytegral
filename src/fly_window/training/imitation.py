from __future__ import annotations

import numpy as np
import torch

from fly_window.env.config import EnvironmentConfig
from fly_window.env.core import WindowExitEnv
from fly_window.env.oracle import oracle_action


def collect_oracle_dataset(
    env_config: EnvironmentConfig,
    *,
    seeds: tuple[int, ...] | list[int],
    max_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    if max_samples <= 0:
        raise ValueError("max_samples must be positive")

    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    for seed in seeds:
        env = WindowExitEnv(env_config)
        observation, state = env.reset(int(seed))
        while len(observations) < max_samples:
            action = oracle_action(state, env_config.room)
            observations.append(np.asarray(observation, dtype=np.float32))
            actions.append(np.asarray(action, dtype=np.float32))
            result = env.step(action)
            if result.terminated or result.truncated:
                break
            observation = result.observation
            state = result.state
        if len(observations) >= max_samples:
            break

    if not observations:
        return (
            np.empty((0, env_config.sensor.ray_count), dtype=np.float32),
            np.empty((0, 2), dtype=np.float32),
        )
    return np.stack(observations), np.stack(actions)


def imitation_step(
    policy,
    optimizer: torch.optim.Optimizer,
    observations: torch.Tensor,
    target_actions: torch.Tensor,
) -> float:
    policy.train()
    optimizer.zero_grad(set_to_none=True)
    output = policy(observations)
    predicted_actions = policy.action_from_latent(output.mean_action)
    loss = torch.nn.functional.mse_loss(predicted_actions, target_actions)
    loss.backward()
    optimizer.step()
    return float(loss.detach().cpu())
