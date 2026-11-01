import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from fly_window.text.style_profile import (
    StyleProfile, learn_style, select_styled, sentence_case, styled_fly_trace,
)
from fly_window.text.compositional_engine import choose_parts


def corpus():
    return [
        {"tweet_id": str(i), "translation_status": "original_en" if i < 10 else "editor_draft",
         "text_en": f"Is theorem {i} actually true?" if i % 2 else f"I doubt the theorem {i}."}
        for i in range(16)
    ]


def trace(i=0):
    return {"token_ids": list(range(i + 10, i + 51)),
            "text": "proof math topology and banana", "idempotency_key": f"old:{i}"}


def test_style_profile_uses_only_trusted_english_with_no_duplicate_or_urls():
    examples = corpus()
    examples.extend([
        {"translation_status": "machine_draft", "text_en": "x" * 300},
        {"translation_status": "original_en", "text_en": "https://t.co/test"},
        examples[0],
    ])
    profile = learn_style(examples)
    assert profile.examples == 16
    assert profile.question_fraction == .5
    assert 15 < profile.median_chars < 45
    assert profile.lowercase_start_fraction == 0


def test_minimum_trustworthy_corpus_needed():
    with pytest.raises(ValueError, match="at least eight"):
        learn_style(corpus()[:7])


def test_capitalization_conforms_to_owner_style():
    assert sentence_case("i fell. i woke up in a jar. is this topology?") == (
        "I fell. I woke up in a jar. Is this topology?"
    )


def test_fly_still_controls_semantic_components_not_queue_seed():
    profile = learn_style(corpus())
    item = trace(1)
    output = styled_fly_trace(item, profile)
    concept, scene, _, trigger = choose_parts(item)
    assert output["fly_topic"] == concept.key
    assert output["fly_scene"] == scene.key
    assert output["fly_trigger"] == trigger
    assert output["text"].startswith(output["text"][:1].upper())
    assert output["publishable"] is False
    assert output == styled_fly_trace({**item, "rng_seed": 999, "idempotency_key": "different"}, profile) | {
        "source_idempotency_key": "old:1"
    }


def test_select_styled_uses_actual_traces_and_no_copying():
    profile = learn_style(corpus())
    inputs = [trace(i * 7) for i in range(70)]
    result = select_styled(inputs, profile, count=30)
    assert 10 < len(result) <= 30
    assert len({r["text"] for r in result}) == len(result)
    assert all(r["decoder"] == "fly_compositional_style_v2" for r in result)
    assert all(r["publishable"] is False for r in result)
    corpus_texts = {r["text_en"] for r in corpus()}
    assert all(r["text"] not in corpus_texts for r in result)
