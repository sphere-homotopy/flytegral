import numpy as np
import torch

from fly_window.env.config import EnvironmentConfig, RewardConfig
from fly_window.neural.graph import ConnectomeGraph, to_sparse_recurrent
from fly_window.neural.policy import ConnectomePolicy
from fly_window.training.imitation import collect_oracle_dataset, imitation_step


def _toy_graph() -> ConnectomeGraph:
    return ConnectomeGraph(
        body_ids=np.asarray([10, 20, 30], dtype=np.int64),
        src_idx=np.asarray([0, 1], dtype=np.int64),
        dst_idx=np.asarray([1, 2], dtype=np.int64),
        weights=np.asarray([3, 5], dtype=np.int64),
        input_mask=np.asarray([True, False, False], dtype=np.bool_),
        output_mask=np.asarray([False, False, True], dtype=np.bool_),
        transmitter_sign=np.asarray([1.0, 1.0, 1.0], dtype=np.float32),
    )


def test_collect_oracle_dataset_contains_only_visual_observations_and_actions():
    config = EnvironmentConfig(reward=RewardConfig(max_steps=20))
    observations, actions = collect_oracle_dataset(config, seeds=(1, 2), max_samples=12)

    assert observations.ndim == 2
    assert observations.shape[1] == 16
    assert actions.shape == (observations.shape[0], 2)
    assert 1 <= observations.shape[0] <= 12
    assert np.all(actions[:, 0] >= -1.0)
    assert np.all(actions[:, 0] <= 1.0)
    assert np.all(actions[:, 1] == 1.0)


def test_imitation_step_updates_trainable_surface_but_not_recurrent_graph():
    torch.manual_seed(7)
    graph = _toy_graph()
    recurrent = to_sparse_recurrent(graph)
    policy = ConnectomePolicy(graph, recurrent)
    optimizer = torch.optim.Adam(
        [parameter for parameter in policy.parameters() if parameter.requires_grad],
        lr=1e-2,
    )
    observations = torch.rand((4, 16), dtype=torch.float32)
    target_actions = torch.tensor(
        [[-0.8, 1.0], [-0.2, 1.0], [0.3, 1.0], [0.9, 1.0]], dtype=torch.float32
    )

    before = {name: parameter.detach().clone() for name, parameter in policy.named_parameters()}
    recurrent_before = policy.recurrent.detach().clone()
    loss = imitation_step(policy, optimizer, observations, target_actions)

    assert loss > 0.0
    assert torch.equal(policy.recurrent, recurrent_before)
    assert any(
        not torch.equal(before[name], parameter.detach())
        for name, parameter in policy.named_parameters()
        if parameter.requires_grad
    )
