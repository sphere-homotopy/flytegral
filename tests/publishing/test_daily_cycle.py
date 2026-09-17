from copy import deepcopy
from datetime import UTC, datetime, timedelta
import json

import numpy as np
import pytest
import torch

from fly_window.neural.graph import ConnectomeGraph, to_sparse_recurrent
from fly_window.publishing.cadence import FlyCadencePolicy
from fly_window.publishing.daily_cycle import run_scheduled_daily_cycle
from fly_window.text.daily_training import DailyTrainingConfig
from fly_window.text.generation import GenerationConfig
from fly_window.text.policy import FlyTextPolicy
from fly_window.text.vocabulary import build_v1_vocabulary


def _graph() -> ConnectomeGraph:
    return ConnectomeGraph(
        body_ids=np.asarray([10, 20, 30, 40], dtype=np.int64),
        src_idx=np.asarray([0, 1, 2], dtype=np.int64),
        dst_idx=np.asarray([1, 2, 3], dtype=np.int64),
        weights=np.asarray([3, 2, 4], dtype=np.int64),
        input_mask=np.asarray([True, False, False, False], dtype=np.bool_),
        output_mask=np.asarray([False, False, False, True], dtype=np.bool_),
        transmitter_sign=np.asarray([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
    )


def _policies():
    vocabulary = build_v1_vocabulary()
    graph = _graph()
    recurrent = to_sparse_recurrent(graph)
    torch.manual_seed(123)
    policy = FlyTextPolicy(
        graph,
        recurrent,
        vocab_size=len(vocabulary),
        sensory_dim=8,
        microsteps=1,
        leak=1.0,
    )
    anchor = deepcopy(policy)
    cadence = FlyCadencePolicy(
        graph,
        recurrent,
        wait_minutes=(20, 30, 60),
        context_dim=4,
        microsteps=1,
        leak=1.0,
    )
    return policy, anchor, cadence, vocabulary


def _training_config() -> DailyTrainingConfig:
    return DailyTrainingConfig(
        learning_rate=1e-3,
        weight_decay=0.0,
        entropy_coefficient=0.01,
        anchor_kl_coefficient=0.05,
        gradient_clip_norm=1.0,
        max_update_steps=1,
    )


def _rewarded_row(vocabulary):
    return {
        "idempotency_key": "old:0",
        "token_ids": json.dumps([vocabulary.id_for("math")]),
        "reward": 1.0,
        "training_consumed_at": "",
    }


def _window():
    start = datetime(2026, 9, 18, 0, 0, tzinfo=UTC)
    return start, start + timedelta(hours=1)


def _force_wait_20(cadence):
    with torch.no_grad():
        cadence.action_readout.weight.zero_()
        cadence.action_readout.bias.fill_(-100.0)
        cadence.action_readout.bias[1] = 100.0


def _force_stop(cadence):
    with torch.no_grad():
        cadence.action_readout.weight.zero_()
        cadence.action_readout.bias.fill_(-100.0)
        cadence.action_readout.bias[0] = 100.0


def test_cycle_trains_when_stats_exist_and_uses_fly_selected_times():
    policy, anchor, cadence, vocabulary = _policies()
    _force_wait_20(cadence)
    rows = [_rewarded_row(vocabulary)]
    start, end = _window()
    written = []
    appended = []

    result = run_scheduled_daily_cycle(
        policy,
        anchor,
        cadence,
        vocabulary=vocabulary,
        rows=rows,
        training_config=_training_config(),
        generation_config=GenerationConfig(max_tokens=1),
        current_checkpoint_id="current-a",
        next_checkpoint_id="daily-a",
        batch_id="batch-a",
        git_sha="abc123",
        text_seed=900,
        cadence_seed=901,
        window_start=start,
        window_end=end,
        now=datetime(2026, 9, 17, 15, 0, tzinfo=UTC),
        write_checkpoint=lambda *args: written.append(args),
        append_rows=lambda new_rows: appended.extend(new_rows),
    )

    assert result.training_applied is True
    assert result.checkpoint_id == "daily-a"
    assert result.publish_times == (
        start + timedelta(minutes=20),
        start + timedelta(minutes=40),
    )
    assert [row["scheduled_at"] for row in appended] == [
        "2026-09-18T00:20:00+00:00",
        "2026-09-18T00:40:00+00:00",
    ]
    assert len(written) == 1
    assert rows[0]["training_consumed_at"] != ""


def test_cycle_without_new_stats_still_generates_from_current_checkpoint():
    policy, anchor, cadence, vocabulary = _policies()
    _force_wait_20(cadence)
    start, end = _window()
    written = []
    appended = []

    result = run_scheduled_daily_cycle(
        policy,
        anchor,
        cadence,
        vocabulary=vocabulary,
        rows=[],
        training_config=_training_config(),
        generation_config=GenerationConfig(max_tokens=1),
        current_checkpoint_id="current-a",
        next_checkpoint_id="daily-unused",
        batch_id="batch-b",
        git_sha="abc123",
        text_seed=910,
        cadence_seed=911,
        window_start=start,
        window_end=end,
        now=datetime(2026, 9, 17, 15, 0, tzinfo=UTC),
        write_checkpoint=lambda *args: written.append(args),
        append_rows=lambda new_rows: appended.extend(new_rows),
    )

    assert result.training_applied is False
    assert result.checkpoint_id == "current-a"
    assert len(written) == 0
    assert len(appended) == 2


def test_cycle_allows_fly_to_schedule_zero_posts_and_still_commit_training():
    policy, anchor, cadence, vocabulary = _policies()
    _force_stop(cadence)
    rows = [_rewarded_row(vocabulary)]
    start, end = _window()
    written = []
    appended = []

    result = run_scheduled_daily_cycle(
        policy,
        anchor,
        cadence,
        vocabulary=vocabulary,
        rows=rows,
        training_config=_training_config(),
        generation_config=GenerationConfig(max_tokens=1),
        current_checkpoint_id="current-a",
        next_checkpoint_id="daily-zero",
        batch_id="batch-zero",
        git_sha="abc123",
        text_seed=920,
        cadence_seed=921,
        window_start=start,
        window_end=end,
        now=datetime(2026, 9, 17, 15, 0, tzinfo=UTC),
        write_checkpoint=lambda *args: written.append(args),
        append_rows=lambda new_rows: appended.extend(new_rows),
    )

    assert result.training_applied is True
    assert result.publish_times == ()
    assert result.generated_rows == ()
    assert len(written) == 1
    assert appended == []
    assert rows[0]["training_consumed_at"] != ""


def test_cycle_rolls_back_text_update_and_consumption_if_queue_append_fails():
    policy, anchor, cadence, vocabulary = _policies()
    _force_wait_20(cadence)
    rows = [_rewarded_row(vocabulary)]
    start, end = _window()
    before = {name: tensor.detach().clone() for name, tensor in policy.state_dict().items()}

    with pytest.raises(RuntimeError, match="queue unavailable"):
        run_scheduled_daily_cycle(
            policy,
            anchor,
            cadence,
            vocabulary=vocabulary,
            rows=rows,
            training_config=_training_config(),
            generation_config=GenerationConfig(max_tokens=1),
            current_checkpoint_id="current-a",
            next_checkpoint_id="daily-fail",
            batch_id="batch-fail",
            git_sha="abc123",
            text_seed=930,
            cadence_seed=931,
            window_start=start,
            window_end=end,
            now=datetime(2026, 9, 17, 15, 0, tzinfo=UTC),
            write_checkpoint=lambda *args: None,
            append_rows=lambda new_rows: (_ for _ in ()).throw(RuntimeError("queue unavailable")),
        )

    assert rows[0]["training_consumed_at"] == ""
    for name, expected in before.items():
        actual = policy.state_dict()[name]
        if expected.layout == torch.sparse_coo:
            expected = expected.coalesce()
            actual = actual.coalesce()
            assert torch.equal(actual.indices(), expected.indices())
            assert torch.equal(actual.values(), expected.values())
        else:
            assert torch.equal(actual, expected)
