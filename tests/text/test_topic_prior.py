import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from fly_window.text.topic_prior import learn_topic_prior, weighted_index
from fly_window.text.compositional_engine import choose_parts
from fly_window.text.style_profile import learn_style, styled_fly_trace


def owner_posts():
    ru = [
        {"tweet_id": str(i), "text_original": f"Группа {i} и фундаментальная группа",
         "translation_status": "machine_draft", "text_en": "Machine garbage"}
        for i in range(32)
    ]
    ru.extend([
        {"tweet_id": str(100 + i), "text_original": f"Гомеоморфизм {i}",
         "translation_status": "machine_draft", "text_en": "machine"}
        for i in range(7)
    ])
    ru.extend([
        {"tweet_id": str(200 + i), "text_original": f"Question {i}: mathematics?",
         "translation_status": "original_en", "text_en": f"Question {i}: mathematics?"}
        for i in range(10)
    ])
    return ru


def test_prior_learns_ru_math_without_relying_on_machine_translation():
    prior = learn_topic_prior(owner_posts())
    assert prior.source_posts == 49
    assert prior.counts["symmetry"] == 32
    assert prior.counts["equivalence"] == 7
    assert prior.counts["compact"] == 0
    assert prior.matched_posts == 39
    assert prior.weight("symmetry") > prior.weight("equivalence") > prior.weight("compact")
    assert len(prior.fingerprint) == 12


def test_smoothed_topics_remain_possible_and_prior_is_valid():
    prior = learn_topic_prior(owner_posts())
    keys = ["symmetry", "equivalence", "compact"]
    counts = [0, 0, 0]
    for step in range(1000):
        counts[weighted_index(keys, prior, step / 1000)] += 1
    assert counts[0] > counts[1] > counts[2] > 0
    with pytest.raises(ValueError):
        weighted_index(keys, prior, 1.)
    with pytest.raises(ValueError):
        learn_topic_prior([])


def test_unchanged_salient_trigger_takes_precedence_over_prior():
    prior = learn_topic_prior(owner_posts())
    row = {"text": "this is compact", "token_ids": list(range(1, 50))}
    assert choose_parts(row, topic_prior=prior)[0].key == "compact"
    assert choose_parts(row, topic_prior=prior)[3] == "compact"


def test_prior_only_changes_fly_controlled_fallback_distribution():
    prior = learn_topic_prior(owner_posts())
    observed_diff = False
    for n in range(10, 70):
        row = {"text": "hello fly", "token_ids": list(range(n, n + 42))}
        unchanged = choose_parts(row)[0].key
        personalized = choose_parts(row, topic_prior=prior)[0].key
        if unchanged != personalized:
            observed_diff = True
            break
    assert observed_diff


def test_style_and_topic_prior_can_be_combined_without_copying():
    source = owner_posts()
    prior = learn_topic_prior(source)
    style = learn_style(source)
    row = {"text": "hello fly", "token_ids": list(range(1, 41)), "idempotency_key": "x"}
    result = styled_fly_trace(row, style, topic_prior=prior)
    assert result["topic_prior_fingerprint"] == prior.fingerprint
    assert result["style_fingerprint"]
    assert result["publishable"] is False
