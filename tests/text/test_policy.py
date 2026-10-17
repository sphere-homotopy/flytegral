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


def test_cached_sparse_transpose_matches_uncached_dynamics_and_gradients():
    policy = _policy()
    assert policy.recurrent_transpose.is_coalesced()
    assert "recurrent_transpose" not in policy.state_dict()
    assert torch.equal(
        policy.recurrent_transpose.to_dense(),
        policy.recurrent.transpose(0, 1).to_dense(),
    )
    ids = torch.tensor([11, 12])
    state = policy.initial_state(batch_size=2)
    output = policy.step(ids, state)
    output.logits.sum().backward()
    grads = {
        name: param.grad.detach().clone()
        for name, param in policy.named_parameters()
        if param.grad is not None
    }
    policy.zero_grad(set_to_none=True)
    activity = state
    sensory = policy.token_embedding(ids)
    drive = sensory @ policy.input_gain.transpose(0, 1) + policy.input_bias
    for _ in range(policy.microsteps):
        recurrent = torch.sparse.mm(
            policy.recurrent.transpose(0, 1), activity.transpose(0, 1)
        ).transpose(0, 1)
        injected = torch.zeros_like(activity)
        injected[:, policy.input_indices] = drive
        proposal = torch.tanh(recurrent + injected)
        activity = (1.0 - policy.leak) * activity + policy.leak * proposal
    reference_logits = policy.readout(activity[:, policy.output_indices])
    assert torch.allclose(output.logits, reference_logits, atol=1e-6)
    reference_logits.sum().backward()
    for name, parameter in policy.named_parameters():
        if name in grads:
            assert torch.allclose(parameter.grad, grads[name], atol=1e-6), name
