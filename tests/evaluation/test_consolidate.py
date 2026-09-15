import pytest

from fly_window.evaluation.consolidate import consolidate_heldout_results


def _payload(label: str, successes: int) -> dict:
    return {
        "summary": {
            "model_label": label,
            "episode_count": 100,
            "successes": successes,
            "success_rate": successes / 100,
            "mean_reward": 1.25,
            "mean_steps": 123.0,
            "seed_start": 10000,
            "seed_end": 10099,
        },
        "batch_size": 32,
        "elapsed_seconds": 10.0,
    }


def test_consolidation_preserves_expected_models_and_computes_release_gate():
    result = consolidate_heldout_results(
        biological=_payload("oracle-imitation-biological", 86),
        shuffled=_payload("oracle-imitation-shuffled", 71),
        untrained=_payload("untrained-biological", 12),
    )

    assert result["held_out_seed_start"] == 10000
    assert result["held_out_seed_end"] == 10099
    assert result["initialization_seed"] == 20260914
    assert result["shuffled_control_seed"] == 314159
    assert [model["model_label"] for model in result["models"]] == [
        "oracle-imitation-biological",
        "untrained-biological",
        "oracle-imitation-shuffled",
    ]
    assert result["release_gate"] == {
        "minimum_biological_successes": 80,
        "minimum_margin_percentage_points": 10,
        "biological_successes": 86,
        "untrained_successes": 12,
        "shuffled_successes": 71,
        "margin_over_untrained_percentage_points": 74,
        "margin_over_shuffled_percentage_points": 15,
        "passed": True,
    }


def test_release_gate_fails_when_either_margin_is_below_ten_points():
    result = consolidate_heldout_results(
        biological=_payload("oracle-imitation-biological", 90),
        shuffled=_payload("oracle-imitation-shuffled", 83),
        untrained=_payload("untrained-biological", 10),
    )

    assert result["release_gate"]["passed"] is False
    assert result["release_gate"]["margin_over_shuffled_percentage_points"] == 7


def test_consolidation_rejects_non_pinned_or_inconsistent_summary():
    biological = _payload("oracle-imitation-biological", 86)
    biological["summary"]["seed_start"] = 10001
    with pytest.raises(ValueError, match="10000..10099"):
        consolidate_heldout_results(
            biological=biological,
            shuffled=_payload("oracle-imitation-shuffled", 71),
            untrained=_payload("untrained-biological", 12),
        )

    inconsistent = _payload("oracle-imitation-biological", 86)
    inconsistent["summary"]["success_rate"] = 0.99
    with pytest.raises(ValueError, match="success_rate"):
        consolidate_heldout_results(
            biological=inconsistent,
            shuffled=_payload("oracle-imitation-shuffled", 71),
            untrained=_payload("untrained-biological", 12),
        )
