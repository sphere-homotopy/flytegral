import json

import numpy as np
import pytest
import torch

from fly_window.neural.graph import ConnectomeGraph, to_sparse_recurrent
from fly_window.text.daily_training import (
    DailyTrainingConfig,
    apply_daily_policy_update,
    daily_policy_objective,
    training_examples_from_rows,
)
from fly_window.text.policy import FlyTextPolicy
from fly_window.text.vocabulary import build_v1_vocabulary


def _toy_graph() -> ConnectomeGraph:
    return ConnectomeGraph(
        body_ids=np.asarray([10, 20, 30, 40], dtype=np.int64),
        src_idx=np.asarray([0, 1, 2], dtype=np.int64),
        dst_idx=np.asarray([1, 2, 3], dtype=np.int64),
        weights=np.asarray([3, 2, 4], dtype=np.int64),
        input_mask=np.asarray([True, False, False, False], dtype=np.bool_),
        output_mask=np.asarray([False, False, False, True], dtype=np.bool_),
        transmitter_sign=np.asarray([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
    )


def _policy(seed: int = 7) -> FlyTextPolicy:
    torch.manual_seed(seed)
    graph = _toy_graph()
    return FlyTextPolicy(
        graph,
        to_sparse_recurrent(graph),
        vocab_size=1024,
        sensory_dim=8,
        microsteps=2,
        leak=0.5,
    )


def _config(**overrides) -> DailyTrainingConfig:
    values = {
        "learning_rate": 1e-3,
        "weight_decay": 0.0,
        "entropy_coefficient": 0.01,
        "anchor_kl_coefficient": 0.05,
        "gradient_clip_norm": 1.0,
        "max_update_steps": 1,
    }
    values.update(overrides)
    return DailyTrainingConfig(**values)


def _row(key: str, tokens: list[int], reward: float, consumed: str = "") -> dict[str, object]:
    return {
        "idempotency_key": key,
        "token_ids": json.dumps(tokens),
        "reward": reward,
        "training_consumed_at": consumed,
    }


def test_training_examples_skip_consumed_rows_and_parse_json_trajectories():
    vocabulary = build_v1_vocabulary()
    theorem = vocabulary.id_for("theorem")
    proof = vocabulary.id_for("proof")

    examples = training_examples_from_rows(
        [
            _row("batch:0", [theorem, proof], 1.0),
            _row("batch:1", [proof], -1.0, "2026-09-17T10:00:00+00:00"),
        ],
        vocabulary=vocabulary,
    )

    assert [(example.idempotency_key, example.token_ids, example.reward) for example in examples] == [
        ("batch:0", (theorem, proof), 1.0)
    ]


def test_training_examples_reject_duplicate_unconsumed_keys():
    vocabulary = build_v1_vocabulary()
    token = vocabulary.id_for("theorem")

    with pytest.raises(ValueError, match="duplicate idempotency_key"):
        training_examples_from_rows(
            [_row("same", [token], 1.0), _row("same", [token], -1.0)],
            vocabulary=vocabulary,
        )


def test_daily_objective_is_finite_and_anchor_kl_is_zero_for_identical_policies():
    vocabulary = build_v1_vocabulary()
    policy = _policy()
    anchor = _policy()
    tokens = (vocabulary.id_for("theorem"), vocabulary.id_for("proof"))
    examples = training_examples_from_rows(
        [_row("batch:0", list(tokens), 0.75)], vocabulary=vocabulary
    )

    objective = daily_policy_objective(
        policy,
        anchor,
        vocabulary=vocabulary,
        examples=examples,
        config=_config(),
    )

    assert torch.isfinite(objective.loss)
    assert objective.anchor_kl.item() == pytest.approx(0.0, abs=1e-7)
    assert objective.entropy.item() > 0.0
    objective.loss.backward()
    assert policy.token_embedding.weight.grad is not None
    assert policy.recurrent.requires_grad is False
    assert anchor.token_embedding.weight.grad is None


def test_daily_update_changes_text_interface_but_never_connectome_weights():
    vocabulary = build_v1_vocabulary()
    policy = _policy(seed=11)
    anchor = _policy(seed=11)
    recurrent_before = policy.recurrent.coalesce().values().clone()
    parameter_before = policy.readout.weight.detach().clone()
    examples = training_examples_from_rows(
        [
            _row(
                "batch:0",
                [vocabulary.id_for("theorem"), vocabulary.id_for("proof")],
                1.0,
            ),
            _row(
                "batch:1",
                [vocabulary.id_for("banana"), vocabulary.id_for("bzz")],
                -1.0,
            ),
        ],
        vocabulary=vocabulary,
    )

    stats = apply_daily_policy_update(
        policy,
        anchor,
        vocabulary=vocabulary,
        examples=examples,
        config=_config(),
    )

    assert stats.update_steps == 1
    assert stats.example_count == 2
    assert torch.equal(policy.recurrent.coalesce().values(), recurrent_before)
    assert not torch.equal(policy.readout.weight.detach(), parameter_before)
    assert all(torch.isfinite(torch.tensor(value)) for value in stats.losses)


def test_daily_update_refuses_empty_or_already_consumed_work():
    vocabulary = build_v1_vocabulary()
    policy = _policy()
    anchor = _policy()

    with pytest.raises(ValueError, match="no unconsumed training examples"):
        apply_daily_policy_update(
            policy,
            anchor,
            vocabulary=vocabulary,
            examples=[],
            config=_config(),
        )
