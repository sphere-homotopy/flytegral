from __future__ import annotations

import torch


def ppo_loss(
    new_log_prob: torch.Tensor,
    old_log_prob: torch.Tensor,
    advantage: torch.Tensor,
    new_value: torch.Tensor,
    target_return: torch.Tensor,
    entropy: torch.Tensor,
    clip_ratio: float = 0.2,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the clipped PPO objective and its inspectable components."""

    ratio = torch.exp(new_log_prob - old_log_prob)
    unclipped_objective = ratio * advantage
    clipped_objective = torch.clamp(
        ratio,
        min=1.0 - clip_ratio,
        max=1.0 + clip_ratio,
    ) * advantage

    policy_loss = -torch.minimum(unclipped_objective, clipped_objective).mean()
    value_loss = torch.square(new_value - target_return).mean()
    entropy_term = entropy.mean()
    total_loss = policy_loss + value_coef * value_loss - entropy_coef * entropy_term

    return total_loss, policy_loss, value_loss, entropy_term
