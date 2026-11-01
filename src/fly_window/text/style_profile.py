"""Learn only surface-style statistics from the owner's verified English posts.

The fly still selects the semantic tuple. This offline rerenderer calibrates
length, question frequency and capitalization; it does not learn humor.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from statistics import median

from fly_window.text.compositional_engine import (
    MOVES, MathConcept, FlyScene, _choice, _sampled_tokens, choose_parts,
)
from fly_window.text.meme_engine import lexical_similarity

URL = re.compile(r"https?://\S+", re.IGNORECASE)
PRONOUN_I = re.compile(r"\bi\b")
AFTER_STOP = re.compile(r"([.!?]\s+)([a-z])")


@dataclass(frozen=True)
class StyleProfile:
    examples: int
    median_chars: float
    question_fraction: float
    lowercase_start_fraction: float

    def as_dict(self) -> dict:
        return {
            "examples": self.examples,
            "median_chars": self.median_chars,
            "question_fraction": self.question_fraction,
            "lowercase_start_fraction": self.lowercase_start_fraction,
        }


def learn_style(rows: list[dict], *, max_chars: int = 280) -> StyleProfile:
    texts: list[str] = []
    seen: set[str] = set()
    for row in rows:
        # The Argos drafts are NOT clean language references.
        if row.get("translation_status") not in {"original_en", "editor_draft", "reviewed"}:
            continue
        text = str(row.get("text_en") or "").strip()
        if not text or len(text) > max_chars or URL.search(text):
            continue
        key = " ".join(text.casefold().split())
        if key in seen:
            continue
        seen.add(key)
        texts.append(text)
    if len(texts) < 8:
        raise ValueError("need at least eight nonduplicated trusted style examples")
    return StyleProfile(
        examples=len(texts),
        median_chars=float(median(map(len, texts))),
        question_fraction=sum("?" in text for text in texts) / len(texts),
        lowercase_start_fraction=sum(text[:1].islower() for text in texts) / len(texts),
    )


def sentence_case(raw: str) -> str:
    cleaned = PRONOUN_I.sub("I", raw)
    cleaned = cleaned[:1].upper() + cleaned[1:]
    return AFTER_STOP.sub(lambda match: match.group(1) + match.group(2).upper(), cleaned)


def render_short(concept: MathConcept, scene: FlyScene, move: str) -> str:
    """Composes semantic slots, not stored full jokes. All have math referents."""
    fact, observation, label = concept.fact, scene.observation, concept.label
    if move == "report":
        return f"field note on {label}: {observation}."
    if move == "objection":
        return f"{fact}. {observation}."
    if move == "label":
        return f"{observation}. the humans call this {label}."
    if move == "question":
        return f"{observation}. is this {label}?"
    if move == "misclassification":
        return f"is this what {label} means? {observation}."
    if move == "diagnosis":
        return f"{observation}. {scene.objection}. so much for {label}."
    raise ValueError(move)


def styled_fly_trace(row: dict, profile: StyleProfile, *, max_chars: int = 240, topic_prior=None) -> dict:
    tokens = _sampled_tokens(row)
    concept, scene, fly_move, trigger = choose_parts(row, topic_prior=topic_prior)
    move_stream = tokens[2::3]
    question_target = (_choice(move_stream, "question_intent", 10000) / 10000.) < profile.question_fraction
    # The fly's preferred rhetorical move is a tie-breaker, NOT overwritten by a fixed schedule.
    preferred_idx = MOVES.index(fly_move)
    alternatives: list[tuple[float, int, str, str]] = []
    for index, move in enumerate(MOVES):
        output = sentence_case(render_short(concept, scene, move))
        if len(output) > max_chars:
            continue
        is_question = "?" in output
        length_cost = abs(len(output) - profile.median_chars) / max(1., profile.median_chars)
        question_cost = 2.5 if is_question != question_target else 0.
        fly_cost = .15 * min((index - preferred_idx) % len(MOVES),
                             (preferred_idx - index) % len(MOVES))
        # A deterministic sample-derived tie-breaker, never idempotency or posting cadence.
        tie = _choice(move_stream + [index], "style_tie", 100000) / 10000000.
        alternatives.append((length_cost + question_cost + fly_cost + tie, index, move, output))
    if not alternatives:
        raise ValueError("all short formats exceed max_chars")
    _, _, move, result = min(alternatives)
    fingerprint = hashlib.sha256(json.dumps(profile.as_dict(), sort_keys=True).encode()).hexdigest()[:12]
    return {
        "text": result,
        "decoder": "fly_compositional_style_v2",
        "fly_topic": concept.key,
        "fly_scene": scene.key,
        "fly_move": move,
        "fly_trigger": trigger,
        "source_idempotency_key": row.get("idempotency_key", ""),
        "style_fingerprint": fingerprint,
        "topic_prior_fingerprint": topic_prior.fingerprint if topic_prior is not None else None,
        "publishable": False,
    }


def select_styled(rows: list[dict], profile: StyleProfile, count: int = 30,
                  *, max_chars: int = 240, topic_prior=None) -> list[dict]:
    if count < 1:
        raise ValueError("count must be positive")
    out: list[dict] = []
    seen_sources: set[str] = set()
    for row in reversed(rows):
        try:
            candidate = styled_fly_trace(row, profile, max_chars=max_chars, topic_prior=topic_prior)
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        source = str(candidate["source_idempotency_key"])
        if source in seen_sources:
            continue
        if any(lexical_similarity(candidate["text"], other["text"]) >= .73 for other in out):
            continue
        out.append(candidate)
        seen_sources.add(source)
        if len(out) >= count:
            break
    return out
