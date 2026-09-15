from __future__ import annotations

import pytest

from fly_window.rendering.release import build_release_timeline, heldout_metric


def test_release_timeline_is_exactly_32_seconds_and_pinned() -> None:
    timeline = build_release_timeline(fps=30)

    assert timeline.total_frames == 960
    assert timeline.duration_seconds == 32.0
    assert [(segment.kind, segment.seed, segment.start_frame, segment.end_frame) for segment in timeline.segments] == [
        ("hook", None, 0, 90),
        ("development-untrained", 9000, 90, 270),
        ("development-midpoint", 9000, 270, 450),
        ("development-final", 9000, 450, 630),
        ("heldout", 10000, 630, 720),
        ("heldout", 10001, 720, 810),
        ("heldout", 10002, 810, 900),
        ("end-card", None, 900, 960),
    ]


def test_heldout_metric_requires_exact_100_seed_summary() -> None:
    evaluation = {
        "held_out_seed_start": 10000,
        "held_out_seed_end": 10099,
        "models": [
            {
                "model_label": "oracle-imitation-biological",
                "episode_count": 100,
                "successes": 87,
                "success_rate": 0.87,
            }
        ],
    }

    metric = heldout_metric(evaluation, "oracle-imitation-biological")
    assert metric.successes == 87
    assert metric.episode_count == 100
    assert metric.success_rate == 0.87
    assert metric.display == "87/100 held-out successes (87%)"

    evaluation["models"][0]["episode_count"] = 99
    with pytest.raises(ValueError, match="exactly 100"):
        heldout_metric(evaluation, "oracle-imitation-biological")
