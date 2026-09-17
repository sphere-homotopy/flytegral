from __future__ import annotations

import random
from collections.abc import Sequence

from fly_window.text.vocabulary import FlyVocabulary


_NOUN_CANDIDATES = (
    "theorem",
    "proof",
    "lemma",
    "function",
    "sequence",
    "matrix",
    "vector",
    "group",
    "ring",
    "field",
    "module",
    "topology",
    "integral",
    "category",
    "probability",
    "manifold",
    "measure",
    "graph",
    "space",
    "operator",
)

_PROPERTY_CANDIDATES = (
    "finite",
    "infinite",
    "linear",
    "random",
    "canonical",
    "natural",
    "stable",
    "simple",
    "global",
    "local",
    "compact",
    "connected",
    "smooth",
    "analytic",
    "symmetric",
    "convex",
)

_BINARY_RELATIONS = (
    "is",
    "contains",
    "implies",
    "preserves",
)


def _available(vocabulary: FlyVocabulary, candidates: Sequence[str]) -> tuple[str, ...]:
    token_set = set(vocabulary.tokens)
    available = tuple(token for token in candidates if token in token_set)
    if not available:
        raise ValueError("curriculum lexical class has no vocabulary tokens")
    return available


def _encode(vocabulary: FlyVocabulary, tokens: Sequence[str]) -> tuple[int, ...]:
    return tuple(vocabulary.id_for(token) for token in tokens)


def generate_curriculum(
    vocabulary: FlyVocabulary,
    *,
    seed: int,
    count: int,
) -> list[tuple[int, ...]]:
    if count < 0:
        raise ValueError("count must be non-negative")

    rng = random.Random(seed)
    nouns = _available(vocabulary, _NOUN_CANDIDATES)
    properties = _available(vocabulary, _PROPERTY_CANDIDATES)
    relations = _available(vocabulary, _BINARY_RELATIONS)

    required = {
        "<BOS>",
        "<EOS>",
        ".",
        "?",
        ":",
        "every",
        "if",
        "then",
        "there",
        "exists",
        "such",
        "that",
        "why",
        "not",
        "is",
        "bzz",
        "buzz",
    }
    missing = required - set(vocabulary.tokens)
    if missing:
        raise ValueError(f"vocabulary lacks curriculum tokens: {sorted(missing)}")

    def noun() -> str:
        return rng.choice(nouns)

    def prop() -> str:
        return rng.choice(properties)

    def statement() -> list[str]:
        return [noun(), rng.choice(relations), rng.choice((noun(), prop()))]

    def ordinary_template() -> list[str]:
        choice = rng.randrange(7)
        if choice == 0:
            return ["every", noun(), "is", prop(), "."]
        if choice == 1:
            return ["if", *statement(), "then", *statement(), "."]
        if choice == 2:
            return ["there", "exists", noun(), "such", "that", *statement(), "."]
        if choice == 3:
            return ["why", "is", noun(), prop(), "?"]
        if choice == 4:
            return [noun(), "is", "not", noun(), "."]
        if choice == 5:
            return ["theorem", ":", *statement(), "."]
        return ["proof", ":", *statement(), "."]

    sequences: list[tuple[int, ...]] = []
    for index in range(count):
        if (index + seed) % 11 == 0:
            body = [rng.choice(("bzz", "buzz")), *statement(), "."]
        else:
            body = ordinary_template()
        tokens = ["<BOS>", *body, "<EOS>"]
        sequences.append(_encode(vocabulary, tokens))

    return sequences
