from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch

from fly_window.env.config import EnvironmentConfig
from fly_window.env.core import WindowExitEnv
from fly_window.env.oracle import oracle_action


@dataclass(frozen=True, slots=True)
class ImitationEpochResult:
    losses: tuple[float, ...]
    permutation: tuple[int, ...]
    batch_count: int
    midpoint_batch: int
    midpoint_samples_seen: int
    midpoint_policy_state: dict[str, torch.Tensor]


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


def _snapshot_policy_state(policy) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in policy.state_dict().items()
    }


def run_imitation_epoch(
    policy,
    optimizer: torch.optim.Optimizer,
    observations: torch.Tensor,
    target_actions: torch.Tensor,
    *,
    batch_size: int,
    seed: int,
) -> ImitationEpochResult:
    if observations.ndim != 2:
        raise ValueError("observations must be a 2D tensor")
    if target_actions.ndim != 2 or target_actions.shape[1] != 2:
        raise ValueError("target_actions must have shape (N, 2)")
    if len(observations) != len(target_actions):
        raise ValueError("observations and target_actions must contain the same number of rows")
    if len(observations) == 0:
        raise ValueError("imitation epoch requires at least one sample")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    permutation_tensor = torch.randperm(len(observations), generator=generator)
    permutation = tuple(int(index) for index in permutation_tensor.tolist())
    batch_count = math.ceil(len(observations) / batch_size)
    midpoint_batch = (batch_count + 1) // 2

    losses: list[float] = []
    midpoint_policy_state: dict[str, torch.Tensor] | None = None
    midpoint_samples_seen = 0
    for batch_number, start in enumerate(range(0, len(observations), batch_size), start=1):
        indices = permutation_tensor[start : start + batch_size]
        losses.append(
            imitation_step(
                policy,
                optimizer,
                observations[indices],
                target_actions[indices],
            )
        )
        if batch_number == midpoint_batch:
            midpoint_samples_seen = min(start + batch_size, len(observations))
            midpoint_policy_state = _snapshot_policy_state(policy)

    if midpoint_policy_state is None:
        raise RuntimeError("failed to capture imitation midpoint")

    return ImitationEpochResult(
        losses=tuple(losses),
        permutation=permutation,
        batch_count=batch_count,
        midpoint_batch=midpoint_batch,
        midpoint_samples_seen=midpoint_samples_seen,
        midpoint_policy_state=midpoint_policy_state,
    )
