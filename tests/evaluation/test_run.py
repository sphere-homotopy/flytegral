from __future__ import annotations

from dataclasses import asdict

import pytest
import torch
from torch import nn

from fly_window.evaluation.run import HELD_OUT_SEEDS, evaluate_policy


class RecordingPolicy(nn.Module):
    def __init__(self, success: bool):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([1.25]))
        self.success = success
        self.eval_called = False

    def eval(self):
        self.eval_called = True
        return super().eval()


def test_frozen_evaluation_uses_exact_held_out_seeds_without_optimizer(monkeypatch):
    policy = RecordingPolicy(success=True)
    seen_seeds: list[int] = []
    before = {name: tensor.detach().clone() for name, tensor in policy.state_dict().items()}

    def forbidden_optimizer(*args, **kwargs):
        raise AssertionError("evaluation must never instantiate an optimizer")

    monkeypatch.setattr(torch.optim, "Adam", forbidden_optimizer)

    def episode_runner(model: RecordingPolicy, seed: int):
        seen_seeds.append(seed)
        return model.success, 2.5, 7

    summary = evaluate_policy(
        model_label="trained",
        policy=policy,
        episode_runner=episode_runner,
    )

    assert policy.eval_called is True
    assert seen_seeds == list(range(10000, 10100))
    assert tuple(seen_seeds) == HELD_OUT_SEEDS
    assert len(seen_seeds) == len(set(seen_seeds)) == 100
    assert asdict(summary) == {
        "model_label": "trained",
        "episode_count": 100,
        "successes": 100,
        "success_rate": 1.0,
        "mean_reward": 2.5,
        "mean_steps": 7.0,
        "seed_start": 10000,
        "seed_end": 10099,
    }
    after = policy.state_dict()
    assert before.keys() == after.keys()
    for name in before:
        assert torch.equal(before[name], after[name])


def test_evaluation_rejects_any_seed_set_other_than_pinned_held_out_set():
    policy = RecordingPolicy(success=False)

    with pytest.raises(ValueError, match="held-out"):
        evaluate_policy(
            model_label="trained",
            policy=policy,
            episode_runner=lambda model, seed: (False, 0.0, 1),
            seeds=tuple(range(10001, 10101)),
        )
