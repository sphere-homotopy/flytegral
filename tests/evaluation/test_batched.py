from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from fly_window.evaluation.batched import evaluate_policy_batched
from fly_window.evaluation.run import HELD_OUT_SEEDS


@dataclass
class _StepResult:
    observation: np.ndarray
    reward: float
    terminated: bool
    truncated: bool
    success: bool


class _ToyEnv:
    def __init__(self) -> None:
        self.seed = 0
        self.step_index = 0

    def reset(self, seed: int):
        self.seed = int(seed)
        self.step_index = 0
        return np.asarray([float(self.seed % 7), 0.0], dtype=np.float32), {}

    def step(self, action: np.ndarray) -> _StepResult:
        self.step_index += 1
        horizon = 1 + (self.seed % 3)
        done = self.step_index >= horizon
        success = done and (self.seed % 2 == 0)
        reward = float(action[0]) + (1.0 if success else 0.0)
        return _StepResult(
            observation=np.asarray(
                [float(self.seed % 7), float(self.step_index)], dtype=np.float32
            ),
            reward=reward,
            terminated=done,
            truncated=False,
            success=success,
        )


class _ToyPolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.25))

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        first = observation[:, 0] * self.scale
        second = observation[:, 1] * 0.0
        return torch.stack((first, second), dim=1)


def _action_from_output(output: torch.Tensor) -> torch.Tensor:
    return output


def test_batched_evaluation_preserves_seed_results_and_model_state() -> None:
    policy = _ToyPolicy()
    before = {name: value.detach().clone() for name, value in policy.state_dict().items()}

    summary, episodes = evaluate_policy_batched(
        model_label="toy",
        policy=policy,
        env_factory=_ToyEnv,
        action_from_output=_action_from_output,
        seeds=HELD_OUT_SEEDS,
        batch_size=17,
    )

    assert summary.episode_count == 100
    assert summary.successes == 50
    assert summary.success_rate == 0.5
    assert [episode.seed for episode in episodes] == list(HELD_OUT_SEEDS)
    assert [episode.steps for episode in episodes[:6]] == [2, 3, 1, 2, 3, 1]
    assert [episode.success for episode in episodes[:4]] == [True, False, True, False]

    after = policy.state_dict()
    assert before.keys() == after.keys()
    for name in before:
        assert torch.equal(before[name], after[name])
