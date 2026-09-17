from datetime import UTC, datetime

import numpy as np
import pytest
import torch

from fly_window.neural.graph import ConnectomeGraph, to_sparse_recurrent
from fly_window.text.daily_training import (
    DailyTrainingConfig,
    run_daily_update,
)
from fly_window.text.generation import GenerationConfig
from fly_window.text.policy import FlyTextPolicy
from fly_window.text.vocabulary import build_v1_vocabulary


def _policy(seed: int = 17) -> FlyTextPolicy:
    torch.manual_seed(seed)
    graph = ConnectomeGraph(
        body_ids=np.asarray([10, 20, 30, 40], dtype=np.int64),
        src_idx=np.asarray([0, 1, 2], dtype=np.int64),
        dst_idx=np.asarray([1, 2, 3], dtype=np.int64),
        weights=np.asarray([3, 2, 4], dtype=np.int64),
        input_mask=np.asarray([True, False, False, False], dtype=np.bool_),
        output_mask=np.asarray([False, False, False, True], dtype=np.bool_),
        transmitter_sign=np.asarray([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
    )
    return FlyTextPolicy(
        graph,
        to_sparse_recurrent(graph),
        vocab_size=1024,
        sensory_dim=8,
        microsteps=2,
        leak=0.5,
    )


def _config() -> DailyTrainingConfig:
    return DailyTrainingConfig(
        learning_rate=1e-3,
        weight_decay=0.0,
        entropy_coefficient=0.01,
        anchor_kl_coefficient=0.05,
        gradient_clip_norm=1.0,
        max_update_steps=1,
    )


def _rows():
    vocabulary = build_v1_vocabulary()
    return [
        {
            "idempotency_key": "old-batch:0",
            "token_ids": f"[{vocabulary.id_for('theorem')}, {vocabulary.id_for('proof')}]",
            "reward": 1.0,
            "training_consumed_at": "",
        },
        {
            "idempotency_key": "old-batch:1",
            "token_ids": f"[{vocabulary.id_for('banana')}, {vocabulary.id_for('bzz')}]",
            "reward": -1.0,
            "training_consumed_at": "",
        },
    ]


def _state(policy: FlyTextPolicy):
    return {key: value.detach().clone() for key, value in policy.state_dict().items()}


def test_daily_transaction_marks_training_rows_only_after_checkpoint_and_append_succeed():
    vocabulary = build_v1_vocabulary()
    policy = _policy()
    anchor = _policy()
    rows = _rows()
    calls = []
    appended = []

    def write_checkpoint(updated_policy, checkpoint_id, manifest):
        calls.append(("checkpoint", checkpoint_id, manifest["example_count"]))
        assert updated_policy is policy

    def append_rows(new_rows):
        calls.append(("append", len(new_rows)))
        appended.extend(new_rows)

    result = run_daily_update(
        policy,
        anchor,
        vocabulary=vocabulary,
        rows=rows,
        config=_config(),
        generation_config=GenerationConfig(max_tokens=1),
        checkpoint_id="daily-2026-09-17-a",
        batch_id="batch-2026-09-18-a",
        git_sha="abc123",
        base_seed=9000,
        now=datetime(2026, 9, 17, 12, 0, tzinfo=UTC),
        write_checkpoint=write_checkpoint,
        append_rows=append_rows,
    )

    assert calls[0][0] == "checkpoint"
    assert calls[1] == ("append", 10)
    assert len(appended) == 10
    assert result.checkpoint_id == "daily-2026-09-17-a"
    assert result.batch_id == "batch-2026-09-18-a"
    assert result.consumed_keys == ("old-batch:0", "old-batch:1")
    assert all(row["training_consumed_at"] == "2026-09-17T12:00:00+00:00" for row in rows)
    assert [row["idempotency_key"] for row in appended] == [
        f"batch-2026-09-18-a:{index}" for index in range(10)
    ]


def test_daily_transaction_rolls_back_model_and_consumption_if_append_fails():
    vocabulary = build_v1_vocabulary()
    policy = _policy()
    anchor = _policy()
    rows = _rows()
    before = _state(policy)

    def write_checkpoint(updated_policy, checkpoint_id, manifest):
        assert updated_policy is policy

    def fail_append(new_rows):
        assert len(new_rows) == 10
        raise RuntimeError("sheet unavailable")

    with pytest.raises(RuntimeError, match="sheet unavailable"):
        run_daily_update(
            policy,
            anchor,
            vocabulary=vocabulary,
            rows=rows,
            config=_config(),
            generation_config=GenerationConfig(max_tokens=1),
            checkpoint_id="daily-2026-09-17-a",
            batch_id="batch-2026-09-18-a",
            git_sha="abc123",
            base_seed=9000,
            now=datetime(2026, 9, 17, 12, 0, tzinfo=UTC),
            write_checkpoint=write_checkpoint,
            append_rows=fail_append,
        )

    assert all(row["training_consumed_at"] == "" for row in rows)
    after = policy.state_dict()
    assert all(torch.equal(after[key], value) for key, value in before.items())
