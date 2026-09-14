import math

import pytest
import torch

from fly_window.training.ppo import ppo_loss


def test_ppo_loss_clips_ratios_and_applies_default_coefficients():
    ratios = torch.tensor([0.5, 1.0, 1.5])
    old_log_prob = torch.zeros(3)
    new_log_prob = ratios.log()
    advantage = torch.ones(3)
    new_value = torch.tensor([0.0, 1.0, 2.0])
    target_return = torch.ones(3)
    entropy = torch.tensor([0.5, 1.0, 1.5])

    total, policy, value, entropy_term = ppo_loss(
        new_log_prob=new_log_prob,
        old_log_prob=old_log_prob,
        advantage=advantage,
        new_value=new_value,
        target_return=target_return,
        entropy=entropy,
    )

    expected_policy = -(0.5 + 1.0 + 1.2) / 3.0
    expected_value = (1.0 + 0.0 + 1.0) / 3.0
    expected_entropy = 1.0
    expected_total = expected_policy + 0.5 * expected_value - 0.01 * expected_entropy

    assert policy.item() == pytest.approx(expected_policy, abs=1e-6)
    assert value.item() == pytest.approx(expected_value, abs=1e-6)
    assert entropy_term.item() == pytest.approx(expected_entropy, abs=1e-6)
    assert total.item() == pytest.approx(expected_total, abs=1e-6)
    assert math.isfinite(total.item())
