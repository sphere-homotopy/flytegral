from types import SimpleNamespace

import torch

from fly_window.env.config import EnvironmentConfig, RewardConfig
from fly_window.evaluation.trajectory import rollout_deterministic


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
