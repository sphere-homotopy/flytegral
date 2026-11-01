import csv
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from fly_window.text.preference_lab import (
    PreferenceModel, make_blind_trials, rank_candidates, read_reviews,
    write_trials,
)


def _row(n: int):
    return {"token_ids": list(range(n + 1, n + 41)), "text": "math banana theorem",
            "idempotency_key": f"batch:{n}", "rng_seed": n + 100}


def test_blind_trials_share_real_source_and_hide_generator_names(tmp_path):
    trials, metadata = make_blind_trials([_row(i) for i in range(35)], count=20)
    assert len(trials) >= 10
    assert len(trials) == len(metadata)
    assert all(set(t) == {"pair_id", "text_a", "text_b", "preferred", "review_notes"} for t in trials)
    assert all(t["preferred"] == "" for t in trials)
    for public, private in zip(trials, metadata):
        assert public["text_a"] == private["a"]["text"]
        assert public["text_b"] == private["b"]["text"]
        assert private["a"]["source_idempotency_key"] == private["b"]["source_idempotency_key"]
        assert private["approved_for_publishing"] is False
    sidecar = write_trials(tmp_path / "review.csv", trials, metadata)
    assert sidecar.exists()
    assert read_reviews(tmp_path / "review.csv", metadata) == []


def test_labels_required_and_explicit_preferences_are_honored(tmp_path):
    trials, meta = make_blind_trials([_row(i) for i in range(35)], count=10)
    path = tmp_path / "review.csv"
    write_trials(path, trials, meta)
    trials[0]["preferred"] = "A"
    trials[1]["preferred"] = "B"
    trials[2]["preferred"] = "tie"
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=trials[0].keys())
        writer.writeheader()
        writer.writerows(trials)
    decisions = read_reviews(path, meta)
    assert decisions == [(meta[0]["a"], meta[0]["b"]), (meta[1]["b"], meta[1]["a"])]


def test_bad_and_duplicate_trials_rejected(tmp_path):
    trials, meta = make_blind_trials([_row(i) for i in range(35)], count=8)
    path = tmp_path / "review.csv"
    write_trials(path, trials, meta)
    trials[0]["preferred"] = "yes"
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=trials[0].keys())
        writer.writeheader()
        writer.writerows(trials)
    with pytest.raises(ValueError, match="invalid"):
        read_reviews(path, meta)


def test_preference_model_only_learns_from_real_inputs():
    win = {"decoder": "fly_compositional_v1", "fly_topic": "recurrence",
           "fly_scene": "lamp", "fly_move": "objection", "text": "i fly. why?"}
    lose = {"decoder": "fly_semantic_contrast_v0", "fly_topic": "recurrence",
            "fly_voice": "theorem", "text": "theorem. i fly."}
    model = PreferenceModel()
    model.train([])
    assert model.as_dict()["human_comparisons"] == 0
    assert model.weights == {}
    model.train([(win, lose)], epochs=60)
    assert model.comparisons == 1
    assert model.score(win) > model.score(lose)
    assert rank_candidates([lose, win], model)[0] == win
    assert model.as_dict()["algorithm"] == "pairwise_logistic_linear_v1"


def test_invalid_count_and_bad_token_trace():
    with pytest.raises(ValueError):
        make_blind_trials([_row(1)], count=0)
    trials, _ = make_blind_trials([{"token_ids": "garbage", "text": "proof"}], count=10)
    assert trials == []
