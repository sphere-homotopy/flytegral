from __future__ import annotations

import math

import numpy as np
import torch

from fly_window.neural.benchmark import (
    benchmark_policy_backward,
    benchmark_policy_forward,
    estimate_training_seconds,
)
from fly_window.neural.graph import ConnectomeGraph, to_sparse_recurrent
from fly_window.neural.policy import ConnectomePolicy


def _tiny_policy() -> ConnectomePolicy:
    graph = ConnectomeGraph(
        body_ids=np.array([10, 11, 12], dtype=np.int64),
        src_idx=np.array([0, 1], dtype=np.int64),
        dst_idx=np.array([1, 2], dtype=np.int64),
        weights=np.array([3, 5], dtype=np.int64),
        input_mask=np.array([True, False, False]),
        output_mask=np.array([False, False, True]),
        transmitter_sign=np.ones(3, dtype=np.float32),
    )
    return ConnectomePolicy(graph, to_sparse_recurrent(graph), microsteps=2)


def test_forward_benchmark_reports_environment_throughput() -> None:
    policy = _tiny_policy()
    result = benchmark_policy_forward(policy, batch_size=4, iterations=3, warmup=1)

    assert result.batch_size == 4
    assert result.iterations == 3
    assert result.samples == 12
    assert result.elapsed_seconds > 0
    assert result.samples_per_second > 0


def test_backward_benchmark_runs_real_autograd_without_mutating_parameters() -> None:
    policy = _tiny_policy()
    before = {name: value.detach().clone() for name, value in policy.named_parameters()}

    result = benchmark_policy_backward(policy, batch_size=2, iterations=2, warmup=1)

    assert result.samples == 4
    assert result.samples_per_second > 0
    for name, value in policy.named_parameters():
        assert torch.equal(value.detach(), before[name])


def test_training_time_estimate_accounts_for_ppo_reprocessing() -> None:
    seconds = estimate_training_seconds(
        max_environment_steps=2_000_000,
        ppo_epochs=4,
        rollout_samples_per_second=1000.0,
        training_samples_per_second=250.0,
    )

    assert math.isclose(seconds.rollout_seconds, 2_000.0)
    assert math.isclose(seconds.ppo_seconds, 32_000.0)
    assert math.isclose(seconds.total_seconds, 34_000.0)
