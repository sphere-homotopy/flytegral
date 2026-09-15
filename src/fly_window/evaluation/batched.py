from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import torch
from torch import nn

from fly_window.evaluation.run import (
    EvaluationSummary,
    HELD_OUT_SEEDS,
    _assert_state_unchanged,
    _snapshot_state,
)


@dataclass(frozen=True, slots=True)
class EpisodeEvaluation:
    seed: int
    success: bool
    reward: float
    steps: int


EnvFactory = Callable[[], Any]
ActionFromOutput = Callable[[Any], torch.Tensor]


def evaluate_policy_batched(
    *,
    model_label: str,
    policy: nn.Module,
    env_factory: EnvFactory,
    action_from_output: ActionFromOutput,
    seeds: tuple[int, ...] = HELD_OUT_SEEDS,
    batch_size: int = 16,
) -> tuple[EvaluationSummary, list[EpisodeEvaluation]]:
    """Evaluate the pinned held-out seeds with batched policy forwards.

    Environments remain independent and are stepped only until their own terminal
    transition. Batching changes only how deterministic policy forward passes are
    grouped; it does not reset, replace, or reorder held-out episodes.
    """

    if tuple(seeds) != HELD_OUT_SEEDS:
        raise ValueError("evaluation must use the pinned held-out seed set 10000..10099")
    if len(seeds) != 100 or len(set(seeds)) != 100:
        raise ValueError("held-out evaluation requires exactly 100 unique seeds")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    before = _snapshot_state(policy)
    policy.eval()
    episodes: list[EpisodeEvaluation] = []

    with torch.no_grad():
        for batch_start in range(0, len(seeds), batch_size):
            batch_seeds = seeds[batch_start : batch_start + batch_size]
            envs = [env_factory() for _ in batch_seeds]
            observations = []
            for env, seed in zip(envs, batch_seeds, strict=True):
                observation, _ = env.reset(int(seed))
                observations.append(np.asarray(observation, dtype=np.float32))

            rewards = np.zeros(len(batch_seeds), dtype=np.float64)
            steps = np.zeros(len(batch_seeds), dtype=np.int64)
            active = list(range(len(batch_seeds)))
            completed: dict[int, EpisodeEvaluation] = {}

            while active:
                observation_tensor = torch.as_tensor(
                    np.stack([observations[index] for index in active]),
                    dtype=torch.float32,
                )
                output = policy(observation_tensor)
                actions = action_from_output(output)
                action_array = actions.detach().cpu().numpy()
                if action_array.shape != (len(active), 2):
                    raise ValueError(
                        f"actions must have shape ({len(active)}, 2), got {action_array.shape}"
                    )

                next_active: list[int] = []
                for row, index in enumerate(active):
                    result = envs[index].step(action_array[row])
                    rewards[index] += float(result.reward)
                    steps[index] += 1
                    observations[index] = np.asarray(result.observation, dtype=np.float32)
                    done = bool(result.terminated or result.truncated)
                    if done:
                        completed[index] = EpisodeEvaluation(
                            seed=int(batch_seeds[index]),
                            success=bool(result.success),
                            reward=float(rewards[index]),
                            steps=int(steps[index]),
                        )
                    else:
                        next_active.append(index)
                active = next_active

            episodes.extend(completed[index] for index in range(len(batch_seeds)))

    _assert_state_unchanged(before, policy)

    successes = sum(int(episode.success) for episode in episodes)
    summary = EvaluationSummary(
        model_label=model_label,
        episode_count=len(episodes),
        successes=successes,
        success_rate=successes / len(episodes),
        mean_reward=float(np.mean([episode.reward for episode in episodes])),
        mean_steps=float(np.mean([episode.steps for episode in episodes])),
        seed_start=episodes[0].seed,
        seed_end=episodes[-1].seed,
    )
    return summary, episodes
