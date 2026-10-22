import numpy as np
import torch

from fly_window.neural.graph import ConnectomeGraph, to_sparse_recurrent
from fly_window.text.gate import GateConfig, evaluate_launch_gate
from fly_window.text.policy import FlyTextPolicy
from fly_window.text.vocabulary import build_v1_vocabulary


def _eos_policy() -> tuple[FlyTextPolicy, object]:
    vocabulary = build_v1_vocabulary()
    graph = ConnectomeGraph(
        body_ids=np.asarray([10, 20, 30], dtype=np.int64),
        src_idx=np.asarray([0, 1], dtype=np.int64),
        dst_idx=np.asarray([1, 2], dtype=np.int64),
        weights=np.asarray([3, 3], dtype=np.int64),
        input_mask=np.asarray([True, False, False], dtype=np.bool_),
        output_mask=np.asarray([False, False, True], dtype=np.bool_),
        transmitter_sign=np.asarray([1.0, 1.0, 1.0], dtype=np.float32),
    )
    policy = FlyTextPolicy(
        graph,
        to_sparse_recurrent(graph),
        vocab_size=len(vocabulary),
        sensory_dim=8,
        microsteps=1,
        leak=1.0,
    )
    with torch.no_grad():
        policy.token_embedding.weight.zero_()
        policy.input_gain.zero_()
        policy.input_bias.zero_()
        policy.readout.weight.zero_()
        policy.readout.bias.fill_(-100.0)
        policy.readout.bias[vocabulary.id_for("<EOS>")] = 100.0
    return policy, vocabulary


def _config() -> GateConfig:
    return GateConfig(
        max_heldout_loss=0.01,
        min_eos_rate=1.0,
        max_repeat_run=2,
        min_non_degenerate_rate=1.0,
        min_median_chars=0,
        max_median_chars=10,
        probe_max_tokens=8,
        temperature=1.0,
    )


def test_launch_gate_passes_deterministic_eos_policy_and_records_replay():
    policy, vocabulary = _eos_policy()
    heldout = [
        (vocabulary.id_for("<BOS>"), vocabulary.id_for("<EOS>")),
    ]

    report = evaluate_launch_gate(
        policy,
        vocabulary,
        heldout=heldout,
        probe_seeds=(11, 12, 13),
        config=_config(),
    )

    assert report.passed is True
    assert report.heldout_loss <= 0.01
    assert report.eos_rate == 1.0
    assert report.non_degenerate_rate == 1.0
    assert report.finite is True
    assert report.replay_exact is True
    assert report.probe_count == 3


def test_launch_gate_fails_closed_on_non_finite_logits():
    policy, vocabulary = _eos_policy()
    with torch.no_grad():
        policy.readout.bias[vocabulary.id_for("<EOS>")] = float("nan")
    heldout = [
        (vocabulary.id_for("<BOS>"), vocabulary.id_for("<EOS>")),
    ]

    report = evaluate_launch_gate(
        policy,
        vocabulary,
        heldout=heldout,
        probe_seeds=(21,),
        config=_config(),
    )

    assert report.passed is False
    assert report.finite is False


def test_gate_config_rejects_invalid_thresholds():
    try:
        GateConfig(
            max_heldout_loss=-1.0,
            min_eos_rate=1.0,
            max_repeat_run=2,
            min_non_degenerate_rate=1.0,
            min_median_chars=0,
            max_median_chars=10,
            probe_max_tokens=8,
            temperature=1.0,
        )
    except ValueError as error:
        assert "max_heldout_loss" in str(error)
    else:
        raise AssertionError("negative held-out loss ceiling must be rejected")
