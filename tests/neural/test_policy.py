import numpy as np
import torch

from fly_window.neural.graph import ConnectomeGraph, to_sparse_recurrent
from fly_window.neural.policy import ConnectomePolicy


def _toy_graph(*, with_edge: bool = True) -> ConnectomeGraph:
    if with_edge:
        src_idx = np.asarray([0, 1], dtype=np.int64)
        dst_idx = np.asarray([1, 2], dtype=np.int64)
        weights = np.asarray([3, 3], dtype=np.int64)
    else:
        src_idx = np.asarray([], dtype=np.int64)
        dst_idx = np.asarray([], dtype=np.int64)
        weights = np.asarray([], dtype=np.int64)

    return ConnectomeGraph(
        body_ids=np.asarray([10, 20, 30], dtype=np.int64),
        src_idx=src_idx,
        dst_idx=dst_idx,
        weights=weights,
        input_mask=np.asarray([True, False, False], dtype=np.bool_),
        output_mask=np.asarray([False, False, True], dtype=np.bool_),
        transmitter_sign=np.asarray([1.0, 1.0, 1.0], dtype=np.float32),
    )


def test_exact_trainable_parameter_surface_and_frozen_graph():
    graph = _toy_graph()
    recurrent = to_sparse_recurrent(graph)
    policy = ConnectomePolicy(graph, recurrent, visual_dim=16, microsteps=4, leak=0.35)

    trainable = {name for name, parameter in policy.named_parameters() if parameter.requires_grad}

    assert trainable == {
        "input_gain",
        "input_bias",
        "readout.weight",
        "readout.bias",
        "log_std",
    }
    assert policy.recurrent.requires_grad is False
    assert policy.recurrent.layout == torch.sparse_coo


def test_sensory_injection_reaches_only_input_mask_nodes():
    graph = _toy_graph(with_edge=False)
    recurrent = to_sparse_recurrent(graph)
    policy = ConnectomePolicy(graph, recurrent, visual_dim=16, microsteps=1, leak=1.0)

    with torch.no_grad():
        policy.input_gain.fill_(1.0)
        policy.input_bias.zero_()
        policy.readout.weight.zero_()
        policy.readout.bias.zero_()

    output = policy(torch.ones((2, 16), dtype=torch.float32))

    assert torch.all(output.activity[:, 0] > 0)
    assert torch.equal(output.activity[:, 1:], torch.zeros((2, 2)))


def test_backprop_updates_encoder_and_readout_but_not_recurrent_graph():
    graph = _toy_graph()
    recurrent = to_sparse_recurrent(graph)
    policy = ConnectomePolicy(graph, recurrent, visual_dim=16, microsteps=4, leak=0.35)
    observation = torch.linspace(0.0, 1.0, 32, dtype=torch.float32).reshape(2, 16)

    output = policy(observation)
    loss = output.mean_action.square().sum() + output.activity[:, graph.output_mask].square().sum()
    loss.backward()

    assert policy.input_gain.grad is not None
    assert policy.input_bias.grad is not None
    assert policy.readout.weight.grad is not None
    assert policy.readout.bias.grad is not None
    assert policy.recurrent.requires_grad is False


def test_distribution_and_latent_action_transform_have_expected_shapes_and_bounds():
    graph = _toy_graph()
    policy = ConnectomePolicy(graph, to_sparse_recurrent(graph))
    observation = torch.zeros((5, 16), dtype=torch.float32)

    output = policy(observation)
    distribution = policy.distribution(output.mean_action)
    latent = distribution.rsample()
    action = policy.action_from_latent(latent)

    assert output.mean_action.shape == (5, 2)
    assert output.activity.shape == (5, 3)
    assert distribution.event_shape == torch.Size([2])
    assert action.shape == (5, 2)
    assert torch.all(action[:, 0] >= -1.0)
    assert torch.all(action[:, 0] <= 1.0)
    assert torch.all(action[:, 1] >= 0.0)
    assert torch.all(action[:, 1] <= 1.0)
