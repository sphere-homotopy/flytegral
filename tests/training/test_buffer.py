import torch

from fly_window.training.buffer import RolloutBuffer


def test_rollout_buffer_computes_hand_checked_gae_and_returns():
    buffer = RolloutBuffer()
    for reward, done, value in [
        (1.0, False, 0.5),
        (0.0, False, 0.25),
        (2.0, True, 1.0),
    ]:
        buffer.add(
            observation=torch.zeros(16),
            latent_action=torch.zeros(2),
            log_prob=torch.tensor(0.0),
            reward=torch.tensor(reward),
            done=torch.tensor(done),
            value=torch.tensor(value),
        )

    advantages, returns = buffer.compute_gae(
        last_value=torch.tensor(0.0),
        gamma=0.99,
        gae_lambda=0.95,
    )

    raw_advantages = torch.tensor([2.32801025, 1.6805, 1.0])
    expected_returns = torch.tensor([2.82801025, 1.9305, 2.0])
    expected_advantages = (raw_advantages - raw_advantages.mean()) / (
        raw_advantages.std(unbiased=False) + 1e-8
    )

    assert torch.allclose(advantages, expected_advantages, atol=1e-6, rtol=0.0)
    assert torch.allclose(returns, expected_returns, atol=1e-6, rtol=0.0)
    assert len(buffer.observations) == 3
    assert len(buffer.latent_actions) == 3
    assert len(buffer.log_probs) == 3
    assert len(buffer.rewards) == 3
    assert len(buffer.dones) == 3
    assert len(buffer.values) == 3
    assert not hasattr(buffer, "activity")
