from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
from torch import nn


HELD_OUT_SEEDS = tuple(range(10000, 10100))


@dataclass(frozen=True, slots=True)
class EvaluationSummary:
    model_label: str
    episode_count: int
    successes: int
    success_rate: float
    mean_reward: float
    mean_steps: float
    seed_start: int
    seed_end: int


EpisodeRunner = Callable[[nn.Module, int], tuple[bool, float, int]]


def _snapshot_state(policy: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in policy.state_dict().items()
    }


def _tensor_state_equal(expected: torch.Tensor, actual: torch.Tensor) -> bool:
    actual = actual.detach().cpu()
    if expected.layout != actual.layout or expected.shape != actual.shape:
        return False
    if expected.layout == torch.sparse_coo:
        expected_coalesced = expected.coalesce()
        actual_coalesced = actual.coalesce()
        return bool(
            torch.equal(expected_coalesced.indices(), actual_coalesced.indices())
            and torch.equal(expected_coalesced.values(), actual_coalesced.values())
        )
    return bool(torch.equal(expected, actual))


def _assert_state_unchanged(
    before: dict[str, torch.Tensor], policy: nn.Module
) -> None:
    after = policy.state_dict()
    if before.keys() != after.keys():
        raise RuntimeError("evaluation changed model state keys")
    for name, expected in before.items():
        if not _tensor_state_equal(expected, after[name]):
            raise RuntimeError(f"evaluation mutated model state: {name}")


def evaluate_policy(
    *,
    model_label: str,
    policy: nn.Module,
    episode_runner: EpisodeRunner,
    seeds: tuple[int, ...] = HELD_OUT_SEEDS,
) -> EvaluationSummary:
    """Evaluate one frozen model on the pinned 100-seed held-out set."""

    if tuple(seeds) != HELD_OUT_SEEDS:
        raise ValueError("evaluation must use the pinned held-out seed set 10000..10099")
    if len(seeds) != 100 or len(set(seeds)) != 100:
        raise ValueError("held-out evaluation requires exactly 100 unique seeds")

    before = _snapshot_state(policy)
    policy.eval()

    successes = 0
    rewards: list[float] = []
    steps: list[int] = []
    with torch.no_grad():
        for seed in seeds:
            success, reward, episode_steps = episode_runner(policy, int(seed))
            successes += int(bool(success))
            rewards.append(float(reward))
            steps.append(int(episode_steps))

    _assert_state_unchanged(before, policy)

    return EvaluationSummary(
        model_label=model_label,
        episode_count=len(seeds),
        successes=successes,
        success_rate=successes / len(seeds),
        mean_reward=float(np.mean(rewards)),
        mean_steps=float(np.mean(steps)),
        seed_start=seeds[0],
        seed_end=seeds[-1],
    )
