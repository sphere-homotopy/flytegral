from types import SimpleNamespace

import torch

from fly_window.env.config import EnvironmentConfig, RewardConfig
from fly_window.evaluation.trajectory import rollout_deterministic, rollout_many_deterministic


class ConstantPolicy:
    def eval(self):
        return self

    def __call__(self, observation):
        batch = observation.shape[0]
        mean_action = torch.zeros((batch, 2), dtype=torch.float32)
        return SimpleNamespace(mean_action=mean_action)

    @staticmethod
    def action_from_latent(latent_action):
        yaw = torch.tanh(latent_action[..., 0])
        thrust = torch.sigmoid(latent_action[..., 1])
        return torch.stack((yaw, thrust), dim=-1)


class CountingPolicy(ConstantPolicy):
    def __init__(self):
        self.calls = 0
        self.batch_sizes = []

    def __call__(self, observation):
        self.calls += 1
        self.batch_sizes.append(int(observation.shape[0]))
        return super().__call__(observation)


def test_rollout_records_initial_and_terminal_frames():
    config = EnvironmentConfig(reward=RewardConfig(max_steps=3))
    trajectory = rollout_deterministic(ConstantPolicy(), config, seed=10000)

    assert trajectory["seed"] == 10000
    assert trajectory["steps"] == 3
    assert trajectory["success"] is False
    assert len(trajectory["frames"]) == 4
    assert trajectory["frames"][0]["step"] == 0
    assert trajectory["frames"][-1]["step"] == 3
    assert trajectory["frames"][-1]["done"] is True
    assert len(trajectory["frames"][1]["action"]) == 2


def test_many_rollout_batches_active_demo_seeds_in_one_policy_call_per_step():
    config = EnvironmentConfig(reward=RewardConfig(max_steps=3))
    policy = CountingPolicy()
    trajectories = rollout_many_deterministic(policy, config, seeds=(10000, 10001, 10002))

    assert [item["seed"] for item in trajectories] == [10000, 10001, 10002]
    assert all(item["steps"] == 3 for item in trajectories)
    assert policy.calls == 3
    assert policy.batch_sizes == [3, 3, 3]
