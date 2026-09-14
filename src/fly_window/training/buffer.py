from __future__ import annotations

import torch


class RolloutBuffer:
    """Minimal PPO rollout storage.

    Neural activity is intentionally not part of PPO state. It is recorded by
    higher-level diagnostics/rendering code when needed.
    """

    def __init__(self) -> None:
        self.observations: list[torch.Tensor] = []
        self.latent_actions: list[torch.Tensor] = []
        self.log_probs: list[torch.Tensor] = []
        self.rewards: list[torch.Tensor] = []
        self.dones: list[torch.Tensor] = []
        self.values: list[torch.Tensor] = []

    def add(
        self,
        *,
        observation: torch.Tensor,
        latent_action: torch.Tensor,
        log_prob: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        value: torch.Tensor,
    ) -> None:
        self.observations.append(torch.as_tensor(observation).detach())
        self.latent_actions.append(torch.as_tensor(latent_action).detach())
        self.log_probs.append(torch.as_tensor(log_prob).detach())
        self.rewards.append(torch.as_tensor(reward).detach())
        self.dones.append(torch.as_tensor(done).detach())
        self.values.append(torch.as_tensor(value).detach())

    def compute_gae(
        self,
        last_value: torch.Tensor,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.rewards:
            raise ValueError("cannot compute GAE for an empty rollout")

        values = torch.stack(self.values).to(dtype=torch.float32)
        rewards = torch.stack(self.rewards).to(device=values.device, dtype=values.dtype)
        dones = torch.stack(self.dones).to(device=values.device, dtype=torch.bool)
        next_value = torch.as_tensor(last_value, device=values.device, dtype=values.dtype)

        advantages = torch.zeros_like(values)
        gae = torch.zeros_like(values[-1])

        for step in range(len(self.rewards) - 1, -1, -1):
            nonterminal = (~dones[step]).to(dtype=values.dtype)
            successor_value = next_value if step == len(self.rewards) - 1 else values[step + 1]
            delta = rewards[step] + gamma * successor_value * nonterminal - values[step]
            gae = delta + gamma * gae_lambda * nonterminal * gae
            advantages[step] = gae

        returns = advantages + values
        normalized = (advantages - advantages.mean()) / (
            advantages.std(unbiased=False) + 1e-8
        )
        return normalized, returns
