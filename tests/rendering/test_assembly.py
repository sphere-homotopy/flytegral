import pytest

from fly_window.rendering.assembly import (
    build_release_assembly_plan,
    validated_release_metric,
)


def _evaluation(*, passed: bool = True) -> dict:
    return {
        "held_out_seed_start": 10000,
        "held_out_seed_end": 10099,
        "release_gate": {"passed": passed},
        "models": [
            {
                "model_label": "oracle-imitation-biological",
                "episode_count": 100,
                "successes": 86,
                "success_rate": 0.86,
                "mean_reward": 1.0,
                "mean_steps": 120.0,
                "seed_start": 10000,
                "seed_end": 10099,
            }
        ],
    }


def test_release_assembly_plan_has_exact_eight_slots_and_32_seconds():
    plan = build_release_assembly_plan()

    assert [(item.kind, item.seed, item.source_name, item.duration_seconds) for item in plan] == [
        ("hook", None, "hook.png", 3.0),
        ("development-untrained", 9000, "development-untrained.mp4", 6.0),
        ("development-midpoint", 9000, "development-midpoint.mp4", 6.0),
        ("development-final", 9000, "development-final.mp4", 6.0),
        ("heldout", 10000, "heldout-10000.mp4", 3.0),
        ("heldout", 10001, "heldout-10001.mp4", 3.0),
        ("heldout", 10002, "heldout-10002.mp4", 3.0),
        ("end-card", None, "end-card.png", 2.0),
    ]
    assert sum(item.duration_seconds for item in plan) == 32.0


def test_release_metric_requires_passing_gate_and_uses_100_seed_result():
    metric = validated_release_metric(_evaluation())
    assert metric.successes == 86
    assert metric.episode_count == 100
    assert metric.display == "86/100 held-out successes (86%)"

    with pytest.raises(ValueError, match="release gate"):
        validated_release_metric(_evaluation(passed=False))


def test_demo_release_can_report_real_metric_without_claiming_gate_passed():
    evaluation = _evaluation(passed=False)
    evaluation["models"][0]["successes"] = 71
    evaluation["models"][0]["success_rate"] = 0.71

    metric = validated_release_metric(evaluation, require_gate=False)

    assert metric.successes == 71
    assert metric.episode_count == 100
    assert metric.display == "71/100 held-out successes (71%)"
    assert evaluation["release_gate"]["passed"] is False
