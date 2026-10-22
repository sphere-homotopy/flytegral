from datetime import UTC, datetime, timedelta

import numpy as np
import torch

from fly_window.neural.graph import ConnectomeGraph, to_sparse_recurrent
from fly_window.publishing.cadence import FlyCadencePolicy, sample_publish_times


WAIT_MINUTES = (20, 30, 45, 60, 90, 120, 180, 240, 360)


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


def _policy() -> FlyCadencePolicy:
    graph = _graph()
    return FlyCadencePolicy(
        graph,
        to_sparse_recurrent(graph),
        wait_minutes=WAIT_MINUTES,
        context_dim=4,
        microsteps=1,
        leak=1.0,
    )


def test_cadence_is_seed_replayable_and_connectome_stays_frozen():
    policy = _policy()
    start = datetime(2026, 9, 18, 0, 0, tzinfo=UTC)
    end = start + timedelta(days=1)

    first = sample_publish_times(policy, start=start, end=end, seed=1709)
    second = sample_publish_times(policy, start=start, end=end, seed=1709)

    assert first == second
    assert policy.recurrent.requires_grad is False


def test_stop_action_allows_fly_to_choose_zero_posts_for_a_day():
    policy = _policy()
    with torch.no_grad():
        policy.action_readout.weight.zero_()
        policy.action_readout.bias.fill_(-100.0)
        policy.action_readout.bias[0] = 100.0  # STOP

    start = datetime(2026, 9, 18, 0, 0, tzinfo=UTC)
    assert sample_publish_times(
        policy,
        start=start,
        end=start + timedelta(days=1),
        seed=1,
    ) == ()


def test_wait_actions_choose_count_and_absolute_publish_times():
    policy = _policy()
    with torch.no_grad():
        policy.action_readout.weight.zero_()
        policy.action_readout.bias.fill_(-100.0)
        policy.action_readout.bias[1] = 100.0  # wait 20 minutes, then post

    start = datetime(2026, 9, 18, 0, 0, tzinfo=UTC)
    times = sample_publish_times(
        policy,
        start=start,
        end=start + timedelta(hours=3),
        seed=2,
        max_posts_per_24h=30,
    )

    assert times[:3] == (
        start + timedelta(minutes=20),
        start + timedelta(minutes=40),
        start + timedelta(minutes=60),
    )
    assert all(left < right for left, right in zip(times, times[1:]))
    assert all(
        right - left >= timedelta(minutes=20)
        for left, right in zip(times, times[1:])
    )


def test_five_day_bootstrap_resets_daily_decision_process_but_keeps_one_schedule():
    policy = _policy()
    with torch.no_grad():
        policy.action_readout.weight.zero_()
        policy.action_readout.bias.fill_(-100.0)
        policy.action_readout.bias[0] = 100.0  # each day may independently choose zero

    start = datetime(2026, 9, 18, 0, 0, tzinfo=UTC)
    end = start + timedelta(days=5)

    assert sample_publish_times(policy, start=start, end=end, seed=3) == ()
