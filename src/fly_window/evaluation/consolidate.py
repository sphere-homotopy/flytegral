from __future__ import annotations

import math
from typing import Any, Mapping


INITIALIZATION_SEED = 20260914
SHUFFLED_CONTROL_SEED = 314159
HELD_OUT_SEED_START = 10000
HELD_OUT_SEED_END = 10099
EXPECTED_EPISODES = 100
MINIMUM_BIOLOGICAL_SUCCESSES = 80
MINIMUM_MARGIN_PERCENTAGE_POINTS = 10

BIOLOGICAL_LABEL = "oracle-imitation-biological"
UNTRAINED_LABEL = "untrained-biological"
SHUFFLED_LABEL = "oracle-imitation-shuffled"


def _validated_summary(payload: Mapping[str, Any], expected_label: str) -> dict[str, Any]:
    summary = payload.get("summary")
    if not isinstance(summary, Mapping):
        raise ValueError(f"{expected_label} payload must contain a summary object")

    label = summary.get("model_label")
    if label != expected_label:
        raise ValueError(
            f"expected model_label {expected_label!r}, got {label!r}"
        )
    episode_count = int(summary.get("episode_count", -1))
    seed_start = int(summary.get("seed_start", -1))
    seed_end = int(summary.get("seed_end", -1))
    if (
        episode_count != EXPECTED_EPISODES
        or seed_start != HELD_OUT_SEED_START
        or seed_end != HELD_OUT_SEED_END
    ):
        raise ValueError(
            "held-out summary must contain exactly seeds 10000..10099"
        )

    successes = int(summary.get("successes", -1))
    if not 0 <= successes <= EXPECTED_EPISODES:
        raise ValueError("successes must lie in [0, 100]")
    success_rate = float(summary.get("success_rate", math.nan))
    expected_rate = successes / EXPECTED_EPISODES
    if not math.isfinite(success_rate) or not math.isclose(
        success_rate, expected_rate, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError("success_rate must equal successes / episode_count")

    return dict(summary)


def consolidate_heldout_results(
    *,
    biological: Mapping[str, Any],
    shuffled: Mapping[str, Any],
    untrained: Mapping[str, Any],
) -> dict[str, Any]:
    biological_summary = _validated_summary(biological, BIOLOGICAL_LABEL)
    shuffled_summary = _validated_summary(shuffled, SHUFFLED_LABEL)
    untrained_summary = _validated_summary(untrained, UNTRAINED_LABEL)

    biological_successes = int(biological_summary["successes"])
    untrained_successes = int(untrained_summary["successes"])
    shuffled_successes = int(shuffled_summary["successes"])
    margin_untrained = biological_successes - untrained_successes
    margin_shuffled = biological_successes - shuffled_successes
    passed = (
        biological_successes >= MINIMUM_BIOLOGICAL_SUCCESSES
        and margin_untrained >= MINIMUM_MARGIN_PERCENTAGE_POINTS
        and margin_shuffled >= MINIMUM_MARGIN_PERCENTAGE_POINTS
    )

    return {
        "held_out_seed_start": HELD_OUT_SEED_START,
        "held_out_seed_end": HELD_OUT_SEED_END,
        "initialization_seed": INITIALIZATION_SEED,
        "shuffled_control_seed": SHUFFLED_CONTROL_SEED,
        "models": [
            biological_summary,
            untrained_summary,
            shuffled_summary,
        ],
        "release_gate": {
            "minimum_biological_successes": MINIMUM_BIOLOGICAL_SUCCESSES,
            "minimum_margin_percentage_points": MINIMUM_MARGIN_PERCENTAGE_POINTS,
            "biological_successes": biological_successes,
            "untrained_successes": untrained_successes,
            "shuffled_successes": shuffled_successes,
            "margin_over_untrained_percentage_points": margin_untrained,
            "margin_over_shuffled_percentage_points": margin_shuffled,
            "passed": passed,
        },
    }
