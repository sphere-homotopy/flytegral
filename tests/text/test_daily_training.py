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


def test_streamed_policy_update_matches_full_batch_objective_for_unequal_lengths():
    vocabulary = build_v1_vocabulary()
    examples = training_examples_from_rows(
        [
            _row("mixed:0", [vocabulary.id_for("theorem")], 1.0),
            _row(
                "mixed:1",
                [vocabulary.id_for("proof"), vocabulary.id_for("banana"), vocabulary.id_for("bzz")],
                -0.35,
            ),
        ],
        vocabulary=vocabulary,
    )
    reference_policy = _policy(seed=25)
    anchor = _policy(seed=25)
    reference = daily_policy_objective(
        reference_policy,
        anchor,
        vocabulary=vocabulary,
        examples=examples,
        config=_config(),
    )
    reference.loss.backward()
    expected_grads = {
        name: param.grad.detach().clone()
        for name, param in reference_policy.named_parameters()
        if param.grad is not None
    }

    streamed_policy = _policy(seed=25)
    stats = apply_daily_policy_update(
        streamed_policy,
        _policy(seed=25),
        vocabulary=vocabulary,
        examples=examples,
        config=_config(),
    )
    assert stats.final_policy_loss == pytest.approx(float(reference.policy_loss.detach()), abs=1e-5)
    assert stats.final_entropy == pytest.approx(float(reference.entropy.detach()), abs=1e-5)
    assert stats.final_anchor_kl == pytest.approx(float(reference.anchor_kl.detach()), abs=1e-5)
    assert stats.losses[0] == pytest.approx(float(reference.loss.detach()), abs=1e-5)
    # Clip norm derives from the accumulated per-example gradient, which should
    # be numerically the same as one global backward() with all tweet graphs.
    expected_norm = torch.linalg.vector_norm(
        torch.cat([grad.reshape(-1) for grad in expected_grads.values()])
    )
    assert stats.gradient_norms[0] == pytest.approx(float(expected_norm), rel=1e-4)


def test_readout_only_matches_full_objective_gradient_for_readout():
    vocabulary = build_v1_vocabulary()
    examples = training_examples_from_rows(
        [
            _row("a", [vocabulary.id_for("theorem")], 0.8),
            _row("b", [vocabulary.id_for("proof"), vocabulary.id_for("banana")], -0.4),
        ],
        vocabulary=vocabulary,
    )
    reference = _policy(seed=31)
    anchor = _policy(seed=31)
    objective = daily_policy_objective(
        reference, anchor, vocabulary=vocabulary,
        examples=examples, config=_config(),
    )
    objective.loss.backward()
    expected_readout_gradient_norm = torch.linalg.vector_norm(
        torch.cat([
            reference.readout.weight.grad.reshape(-1),
            reference.readout.bias.grad.reshape(-1),
        ])
    )
    fast = _policy(seed=31)
    encoder_before = {
        name: param.detach().clone()
        for name, param in fast.named_parameters()
        if not name.startswith("readout.")
    }
    stats = apply_daily_policy_update(
        fast, _policy(seed=31), vocabulary=vocabulary,
        examples=examples, config=_config(training_mode="readout_only"),
    )
    assert stats.final_policy_loss == pytest.approx(float(objective.policy_loss.detach()), abs=1e-5)
    assert stats.final_entropy == pytest.approx(float(objective.entropy.detach()), abs=1e-5)
    assert stats.final_anchor_kl == pytest.approx(float(objective.anchor_kl.detach()), abs=1e-5)
    assert stats.gradient_norms[0] == pytest.approx(float(expected_readout_gradient_norm), rel=1e-4)
    assert all(torch.equal(dict(fast.named_parameters())[n], p) for n, p in encoder_before.items())
    assert all(param.grad is None for name, param in fast.named_parameters() if not name.startswith("readout."))
    assert any(not torch.equal(getattr(fast.readout, n), getattr(reference.readout, n)) for n in ("weight", "bias"))


def test_readout_only_reuses_cached_features_across_optimizer_steps(monkeypatch):
    vocabulary = build_v1_vocabulary()
    tokens = [vocabulary.id_for("theorem"), vocabulary.id_for("proof")]
    examples = training_examples_from_rows(
        [_row("cached", tokens, 1.0)], vocabulary=vocabulary
    )
    policy = _policy(seed=43)
    anchor = _policy(seed=43)
    count = {"forward": 0}
    original_step = policy.step

    def counted_step(token_ids, state):
        count["forward"] += 1
        return original_step(token_ids, state)

    monkeypatch.setattr(policy, "step", counted_step)
    stats = apply_daily_policy_update(
        policy, anchor, vocabulary=vocabulary, examples=examples,
        config=_config(training_mode="readout_only", max_update_steps=3),
    )
    assert stats.update_steps == 3
    assert count["forward"] == len(tokens)
    assert all(torch.isfinite(torch.tensor(value)) for value in stats.losses)


def test_production_daily_config_defaults_to_readout_only():
    from pathlib import Path

    config_path = Path(__file__).resolve().parents[2] / "configs" / "daily_training_v1.json"
    assert DailyTrainingConfig.from_json(config_path).training_mode == "readout_only"


def test_invalid_readout_training_mode_rejected():
    with pytest.raises(ValueError, match="training_mode"):
        _config(training_mode="invalid")
