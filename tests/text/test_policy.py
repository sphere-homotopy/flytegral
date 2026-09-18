import numpy as np
import torch

from fly_window.neural.graph import ConnectomeGraph, to_sparse_recurrent
from fly_window.text.policy import FlyTextPolicy
from fly_window.text.vocabulary import build_v1_vocabulary


def _toy_graph() -> ConnectomeGraph:
    return ConnectomeGraph(
        body_ids=np.asarray([10, 20, 30], dtype=np.int64),
        src_idx=np.asarray([0, 1], dtype=np.int64),
        dst_idx=np.asarray([1, 2], dtype=np.int64),
        weights=np.asarray([3, 3], dtype=np.int64),
        input_mask=np.asarray([True, False, False], dtype=np.bool_),
        output_mask=np.asarray([False, False, True], dtype=np.bool_),
        transmitter_sign=np.asarray([1.0, 1.0, 1.0], dtype=np.float32),
    )


def _policy() -> FlyTextPolicy:
    graph = _toy_graph()
    policy = FlyTextPolicy(
        graph,
        to_sparse_recurrent(graph),
        vocab_size=1024,
        sensory_dim=8,
        microsteps=3,
        leak=0.5,
    )
    with torch.no_grad():
        policy.token_embedding.weight.fill_(0.25)
        policy.input_gain.fill_(0.5)
        policy.input_bias.zero_()
        policy.readout.weight.fill_(0.2)
        policy.readout.bias.zero_()
    return policy


def test_initial_state_is_zero_and_step_returns_vocab_logits():
    policy = _policy()
    state = policy.initial_state(batch_size=2)

    assert state.shape == (2, 3)
    assert torch.equal(state, torch.zeros_like(state))

    output = policy.step(torch.tensor([1, 2]), state)

    assert output.logits.shape == (2, 1024)
    assert output.state.shape == (2, 3)
    assert output.activity.shape == (2, 3)
    assert not torch.equal(output.state, state)


def test_same_token_after_history_has_different_state_than_after_reset():
    vocabulary = build_v1_vocabulary()
    policy = _policy()
    zero = policy.initial_state(batch_size=1)

    first = policy.step(torch.tensor([vocabulary.id_for("<BOS>")]), zero)
    continued = policy.step(torch.tensor([vocabulary.id_for("theorem")]), first.state)
    reset = policy.step(torch.tensor([vocabulary.id_for("theorem")]), zero)

    assert not torch.allclose(continued.state, reset.state)


def test_recurrent_connectome_is_frozen_but_text_interface_is_trainable():
    policy = _policy()
    trainable = {name for name, parameter in policy.named_parameters() if parameter.requires_grad}

    assert policy.recurrent.requires_grad is False
    assert policy.recurrent.layout == torch.sparse_coo
    assert trainable == {
        "token_embedding.weight",
        "input_gain",
        "input_bias",
        "readout.weight",
        "readout.bias",
    }


def test_detach_state_breaks_autograd_history_without_changing_values():
    policy = _policy()
    state = policy.initial_state(batch_size=1)
    output = policy.step(torch.tensor([1]), state)

    detached = policy.detach_state(output.state)

    assert torch.equal(detached, output.state)
    assert detached.requires_grad is False
    assert detached.grad_fn is None
