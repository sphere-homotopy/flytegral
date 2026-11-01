import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from fly_window.text.meme_engine import decode_fly_trace, preview_candidates, SITUATIONS


def test_topic_and_provenance_from_fly_trace():
    row = {"text": "open topology window maths", "token_ids": json.dumps([3, 2, 71]), "rng_seed": 19, "idempotency_key": "b:0"}
    candidate = decode_fly_trace(row)
    assert candidate["fly_topic"] == "open_set"
    assert candidate["fly_trigger"] == "open"
    assert candidate["source_idempotency_key"] == "b:0"
    assert candidate["publishable"] is False
    assert candidate["text"].endswith(".")
    assert len(candidate["text"]) <= 280
    assert candidate == decode_fly_trace(row)


def test_repeated_seeds_do_not_publish_duplicates():
    rows = [{"text": "fixed point", "token_ids": [4, 7], "rng_seed": 1, "idempotency_key": "x"}] * 7
    assert len(preview_candidates(rows, count=5)) == 1


def test_deduplication_rejects_reordered_same_concept():
    from fly_window.text.meme_engine import lexical_similarity
    original = "theorem: the window is glass. i feel betrayed by topology."
    reordered = "i feel betrayed by topology. the window is glass. theorem."
    assert lexical_similarity(original, reordered) == 1.0
    rows = [{"text": "open", "token_ids": [1, 2, i], "rng_seed": i,
             "idempotency_key": str(i)} for i in range(8)]
    assert len(preview_candidates(rows, count=8)) == 1


def test_every_fact_is_renderable_without_truncation():
    from fly_window.text.meme_engine import render_situation, VOICES
    for card in SITUATIONS:
        for voice in VOICES:
            result = render_situation(card, voice)
            assert result.endswith(".")
            assert not result.endswith("..")
            assert len(result) <= 280
