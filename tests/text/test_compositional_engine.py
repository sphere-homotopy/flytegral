import json
import sys
from collections import Counter
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from fly_window.text.compositional_engine import (
    CONCEPTS, MOVES, SCENES, choose_parts, compose_fly_trace,
    render_parts, select_compositions,
)


def row(*tokens, text="", source="test"):
    return {"token_ids": list(tokens), "text": text, "idempotency_key": source}


def test_actual_fly_token_stream_controls_every_choice_and_not_queue_seed():
    original = row(*range(1, 31))
    a = choose_parts(original)
    b = choose_parts({**original, "rng_seed": 99999999, "idempotency_key": "another-day"})
    assert a == b
    assert compose_fly_trace(original)["decoder"] == "fly_compositional_v1"
    assert compose_fly_trace(original)["publishable"] is False


def test_three_disjoint_streams_and_constrained_scene():
    data = row(*range(1, 34))
    base = choose_parts(data)
    changed_scene = dict(data)
    changed_scene["token_ids"] = data["token_ids"][:]
    for i in range(1, len(data["token_ids"]), 3):
        changed_scene["token_ids"][i] = 900 + i
    next_parts = choose_parts(changed_scene)
    assert next_parts[0] == base[0]
    assert next_parts[2] == base[2]
    assert set(next_parts[0].axes).intersection(next_parts[1].axes)


def test_token_keyword_is_salient_but_no_fallback_to_queue_id():
    data = row(10, 20, 30, 40, 50, 60, text="look at this compact noise")
    concept, _, _, trigger = choose_parts(data)
    assert concept.key == "compact"
    assert trigger == "compact"


def test_all_rhetorical_moves_have_complete_renderings():
    for concept in CONCEPTS:
        scenes = [scene for scene in SCENES if set(concept.axes).intersection(scene.axes)]
        assert scenes
        for scene in scenes:
            for move in MOVES:
                out = render_parts(concept, scene, move)
                assert out.endswith((".", "?"))
                assert ".." not in out
                assert len(out) < 300


def test_invalid_traces_fail_closed():
    for invalid in ([], None, "[", [1, -1], ["1"], [1.5]):
        with pytest.raises((ValueError, TypeError, json.JSONDecodeError)):
            compose_fly_trace({"token_ids": invalid})


def test_replay_and_no_truncation_with_realistic_cap():
    data = row(*range(1, 50), text="measure inverse")
    output = compose_fly_trace(data, max_chars=240)
    assert output == compose_fly_trace(data, max_chars=240)
    assert len(output["text"]) <= 240
    assert output["source_idempotency_key"] == "test"


def test_semantic_bridge_excludes_bad_math_scene_pairs():
    by_math = {c.key: c for c in CONCEPTS}
    by_scene = {s.key: s for s in SCENES}
    for math_key, scene_key in (
        ("compact", "paired_jars"), ("compact", "map"),
        ("measure_zero", "shadow"), ("expectation", "hands"),
        ("counting", "crumbs"), ("counting", "two_bananas"),
        ("boundary", "banana"), ("convergence", "oscillation"),
    ):
        assert not set(by_math[math_key].axes).intersection(by_scene[scene_key].axes)


def test_candidate_selection_deduplicates_word_reorderings():
    inputs = [row(*range(i, i + 33), source=str(i)) for i in range(100)]
    out = select_compositions(inputs, count=40)
    assert 1 < len(out) <= 40
    assert len({o["text"] for o in out}) == len(out)
    assert len({o["source_idempotency_key"] for o in out}) == len(out)
    assert len(Counter(o["fly_topic"] for o in out)) > 6
