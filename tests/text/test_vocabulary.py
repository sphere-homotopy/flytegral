import re

from fly_window.text.vocabulary import build_v1_vocabulary


def test_v1_vocabulary_has_exact_size_unique_tokens_and_required_voice():
    vocabulary = build_v1_vocabulary()

    assert len(vocabulary) == 1024
    assert len(set(vocabulary.tokens)) == 1024

    required = {
        "<PAD>",
        "<BOS>",
        "<EOS>",
        "<UNK>",
        "theorem",
        "proof",
        "lemma",
        "topology",
        "integral",
        "category",
        "probability",
        "sigma",
        "epsilon",
        "buzz",
        "bzz",
        "bzzz",
        "fly",
        "banana",
        "neuron",
        "lol",
        "wtf",
    }
    assert required <= set(vocabulary.tokens)


def test_v1_vocabulary_has_stable_round_trip_ids():
    vocabulary = build_v1_vocabulary()

    for token in ("<BOS>", "integral", "bzz", "theorem", "?"):
        token_id = vocabulary.id_for(token)
        assert vocabulary.token_for(token_id) == token


def test_v1_vocabulary_keeps_fly_meme_layer_small_and_contains_no_cyrillic():
    vocabulary = build_v1_vocabulary()

    assert 0 < len(vocabulary.meme_tokens) < 0.05 * len(vocabulary)
    assert set(vocabulary.meme_tokens) <= set(vocabulary.tokens)

    cyrillic = re.compile(r"[\u0400-\u04ff]")
    assert not any(cyrillic.search(token) for token in vocabulary.tokens)
