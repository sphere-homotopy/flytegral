from copy import deepcopy
import json

import numpy as np
import torch

from fly_window.neural.graph import ConnectomeGraph, to_sparse_recurrent
from fly_window.publishing.cadence import FlyCadencePolicy
from fly_window.publishing.cadence_training import (
    CadenceTrainingConfig,
    apply_cadence_policy_update,
    cadence_training_examples_from_rows,
)


def _policy() -> FlyCadencePolicy:
    graph = ConnectomeGraph(
        body_ids=np.asarray([10, 20, 30, 40], dtype=np.int64),
        src_idx=np.asarray([0, 1, 2], dtype=np.int64),
        dst_idx=np.asarray([1, 2, 3], dtype=np.int64),
        weights=np.asarray([3, 2, 4], dtype=np.int64),
        input_mask=np.asarray([True, False, False, False], dtype=np.bool_),
        output_mask=np.asarray([False, False, False, True], dtype=np.bool_),
        transmitter_sign=np.asarray([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
    )
    torch.manual_seed(123)
    return FlyCadencePolicy(
        graph,
        to_sparse_recurrent(graph),
        wait_minutes=(20, 60, 180),
        context_dim=4,
        microsteps=1,
        leak=1.0,
    )


def _config() -> CadenceTrainingConfig:
    return CadenceTrainingConfig(
        learning_rate=1e-3,
        weight_decay=0.0,
        entropy_coefficient=0.01,
        gradient_clip_norm=1.0,
        max_update_steps=1,
    )


def test_rewarded_rows_become_cadence_training_examples():
    rows = [
        {
            "idempotency_key": "batch-a:0",
            "cadence_context": json.dumps([0.0, 1.0, 0.25, 0.75]),
            "cadence_action": 2,
            "reward": 0.8,
            "training_consumed_at": "",
        },
        {
            "idempotency_key": "batch-a:1",
            "cadence_context": json.dumps([1.0, 0.0, 0.5, 0.5]),
            "cadence_action": 1,
            "reward": -0.2,
            "training_consumed_at": "",
        },
    ]

    examples = cadence_training_examples_from_rows(rows, action_count=4)

    assert [example.idempotency_key for example in examples] == ["batch-a:0", "batch-a:1"]
    assert examples[0].action == 2
    assert examples[0].context == (0.0, 1.0, 0.25, 0.75)


def test_engagement_update_changes_cadence_head_but_not_recurrent_connectome():
    policy = _policy()
    recurrent_before = policy.recurrent.coalesce()
    state_before = deepcopy(policy.state_dict())
    rows = [
        {
            "idempotency_key": "batch-a:0",
            "cadence_context": json.dumps([0.0, 1.0, 0.25, 0.75]),
            "cadence_action": 2,
            "reward": 1.0,
            "training_consumed_at": "",
        }
    ]
    examples = cadence_training_examples_from_rows(rows, action_count=policy.action_count)

    stats = apply_cadence_policy_update(policy, examples=examples, config=_config())

    assert stats.example_count == 1
    assert policy.recurrent.requires_grad is False
    recurrent_after = policy.recurrent.coalesce()
    assert torch.equal(recurrent_before.indices(), recurrent_after.indices())
    assert torch.equal(recurrent_before.values(), recurrent_after.values())
    changed = [
        name
        for name, value in policy.state_dict().items()
        if value.layout != torch.sparse_coo and not torch.equal(value, state_before[name])
    ]
    assert any(name.startswith("action_readout") or name.startswith("input_") for name in changed)
