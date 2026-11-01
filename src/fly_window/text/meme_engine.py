"""Offline semantic decoder for *actual* FlyTextPolicy token trajectories.

A constrained, explicit creative baseline: the fly selects a concept and voice
from its sampled trajectory. Human-authored mathematical/world contrasts are
rendered grammatically, with provenance to the triggering fly token.
This is not autonomous joke understanding or an LLM.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class Situation:
    topic: str
    triggers: tuple[str, ...]
    premise: str
    fly_observation: str
    ending: str


SITUATIONS = (
    Situation("open_set", ("open", "topology", "topological", "set", "empty"),
              "the empty set is open", "there are zero exits from this jar", "i would like to see the definition of open again"),
    Situation("fixed_point", ("fixed-point", "fixed", "dynamics", "orbit", "recurrence"),
              "i have found a fixed point", "it's the exact spot on the window i hit yesterday", "the humans seem delighted"),
    Situation("compactness", ("compact", "compactness", "bounded", "closed"),
              "closed and bounded is compact in euclidean space", "my jar meets both conditions", "the mathematician calls this a feature"),
    Situation("hausdorff", ("separate", "hausdorff", "neighborhood", "topology"),
              "in a hausdorff space, two different points can be separated by neighborhoods", "the human and i remain in the same laboratory", "i would like a neighborhood somewhere else"),
    Situation("measure_zero", ("measure", "lebesgue", "zero", "almost-surely", "probability-one"),
              "a single point has lebesgue measure zero", "the swatter still managed to hit exactly mine", "statistics is personal now"),
    Situation("convergence", ("limit", "converge", "convergence", "cauchy"),
              "my trajectory seems to converge", "unfortunately its limit is a window", "i have filed an appeal"),
    Situation("ergodicity", ("ergodic", "ergodicity", "markov", "random-walk"),
              "the humans say my wandering is ergodic", "i have inspected this same lamp fourteen times", "please show me the invariant measure"),
    Situation("pigeonhole", ("combinatorics", "pigeonhole", "matching", "count"),
              "if there are more flies than jars, one jar gets at least two", "the scientist has obtained another jar", "this is not the application i wanted"),
    Situation("continuity", ("continuous", "continuity", "smooth", "differentiable"),
              "the experimenter says my flight path is continuous", "his hand and the swatter are not", "i have questions about the boundary conditions"),
    Situation("proof", ("proof", "theorem", "lemma", "counterexample", "contradiction"),
              "i have a proof that the window is an exit", "it fails at the final step", "the final step is glass"),
    Situation("covering", ("covering", "cover", "fiber", "bundle", "sheaf"),
              "the topologist says every point has a neighborhood", "my favorite point is under the flyswatter", "i am reconsidering local existence"),
    Situation("isomorphism", ("isomorphic", "isomorphism", "equivalent", "homeomorphic"),
              "the humans say two spaces can be homeomorphic without looking alike", "they have confused my jar with a home", "the home has no door"),
    Situation("independence", ("independent", "independence", "conditional", "probability"),
              "they say the experiments are independent", "the scientist uses the same swatter every day", "i suspect a hidden variable"),
    Situation("infinity", ("infinite", "infinity", "countable", "uncountable", "cardinal"),
              "the mathematician assures me there are infinitely many possible trajectories", "i have personally tested the same five", "the window is very persuasive"),
    Situation("optimization", ("optimize", "optimization", "gradient", "minimum", "maximum"),
              "i am minimizing my distance to the banana", "the human keeps moving the banana", "my loss function has become emotional"),
    Situation("symmetry", ("symmetric", "symmetry", "group", "transformation", "rotation"),
              "i discovered a symmetry of the laboratory", "the left and right windows are equally hard", "the group action is hostile"),
)
VOICES = ("deadpan", "lab_note", "theorem", "complaint")
TOKEN_PATTERN = re.compile(r"[\w-]+", re.UNICODE)


def _hash_index(data: object, label: str, modulus: int) -> int:
    encoded = json.dumps([label, data], ensure_ascii=True, sort_keys=True).encode("utf-8")
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big") % modulus


def choose_situation(raw_text: str, token_ids: Iterable[int]) -> tuple[Situation, str]:
    """First math-specific token in Fly text wins; otherwise deterministic trace hash."""
    words = TOKEN_PATTERN.findall(raw_text.lower())
    # The fly's first salient token controls the semantic theme.
    for word in words:
        matches = [card for card in SITUATIONS if word in card.triggers]
        if matches:
            return matches[0], word
    trace = list(token_ids)
    return SITUATIONS[_hash_index(trace, "topic", len(SITUATIONS))], "trace_hash"


def render_situation(card: Situation, voice: str) -> str:
    premise, observation, ending = card.premise, card.fly_observation, card.ending
    if voice == "deadpan":
        return f"{premise}. {observation}. {ending}."
    if voice == "lab_note":
        return f"lab notes: {observation}. {premise}. {ending}."
    if voice == "theorem":
        return f"theorem: {premise}. proof: {observation}. {ending}."
    if voice == "complaint":
        return f"{observation}. {premise}. {ending}."
    raise ValueError(f"unsupported voice: {voice}")


def decode_fly_trace(row: dict, *, max_chars: int = 280) -> dict:
    if max_chars < 40:
        raise ValueError("max_chars must be >= 40")
    token_ids = row.get("token_ids", [])
    if isinstance(token_ids, str):
        token_ids = json.loads(token_ids)
    if not isinstance(token_ids, list):
        raise ValueError("Fly token_ids must be a JSON array")
    text = str(row.get("text", ""))
    card, trigger = choose_situation(text, token_ids)
    identity = [token_ids, row.get("rng_seed", row.get("seed")), row.get("idempotency_key", "")]
    start = _hash_index(identity, "voice", len(VOICES))
    # A long template must never be truncated mid-sentence.
    options = [VOICES[(start + i) % len(VOICES)] for i in range(len(VOICES))]
    for voice in options:
        candidate = render_situation(card, voice)
        if len(candidate) <= max_chars:
            return {
                "text": candidate,
                "source_idempotency_key": row.get("idempotency_key", ""),
                "fly_topic": card.topic,
                "fly_trigger": trigger,
                "fly_voice": voice,
                "decoder": "fly_semantic_contrast_v0",
                "publishable": False,
            }
    raise ValueError("No complete semantic candidate fits char cap")


def lexical_similarity(first: str, second: str) -> float:
    """Jaccard of word sets catches reordered sentences with identical punchlines."""
    a = set(TOKEN_PATTERN.findall(first.casefold()))
    b = set(TOKEN_PATTERN.findall(second.casefold()))
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def preview_candidates(rows: list[dict], *, count: int, max_chars: int = 280) -> list[dict]:
    if count <= 0:
        raise ValueError("count must be positive")
    output: list[dict] = []
    seen_texts: set[str] = set()
    seen_sources: set[str] = set()
    seen_topics: set[str] = set()
    # Pass 1: spread the fly's independent concepts across the feed.
    # Pass 2: allow different voices on a previously seen topic if needed.
    for fresh_topic_only in (True, False):
        for row in reversed(rows):
            candidate = decode_fly_trace(row, max_chars=max_chars)
            topic = candidate["fly_topic"]
            source = candidate["source_idempotency_key"]
            normalized = re.sub(r"\s+", " ", candidate["text"]).casefold()
            if normalized in seen_texts or source in seen_sources:
                continue
            if any(lexical_similarity(candidate["text"], selected["text"]) > 0.72 for selected in output):
                continue
            if fresh_topic_only and topic in seen_topics:
                continue
            seen_topics.add(topic)
            seen_texts.add(normalized)
            seen_sources.add(source)
            output.append(candidate)
            if len(output) == count:
                return output
    return output
