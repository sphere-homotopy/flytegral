"""Weak topic-frequency priors learned from owner's own original posts.

No joke text is copied; no machine translation is required. This only biases
topic selection when the fly has not already emitted a salient math token.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass

# Deliberately conservative: don't equate generic "math" / "topology" with
# one specific theorem. Original RU text avoids translation artifacts.
PATTERNS = {
    "boundary": r"границ|boundary|frontier",
    "open": r"открыт.{0,10}множеств|open set|open subset",
    "compact": r"компакт|compact",
    "convergence": r"предел|сходи|\bconverg|\blimit\b",
    "recurrence": r"динамич|эргодич|ergodic|recurren",
    "fixed_point": r"неподвижн.{0,10}точк|fixed.point|brouwer",
    "measure_zero": r"лебег|lebesgue|нулев.{0,12}мер",
    "independence": r"независим|independen",
    "optimization": r"оптимиз|градиент|optimis|optimiz|gradient",
    "symmetry": r"симметри|\bгрупп|symmet|group theory|group action",
    "equivalence": r"гомеоморф|изоморф|homeomorph|isomorph",
    "continuity": r"непрерыв|дифференц|continuity|continuous|differentiab",
    "counting": r"комбинатор|combinator|pigeonhole",
    "expectation": r"матожид|математическ.{0,12}ожидан|expected value",
    "locality": r"окрестн|neighborhood|пучк|sheaf",
    "entropy": r"энтроп|entropy|случайн|random",
}


@dataclass(frozen=True)
class TopicPrior:
    counts: dict[str, int]
    matched_posts: int
    source_posts: int
    exponent: float = .5
    pseudo_count: float = 1.

    def weight(self, key: str) -> float:
        return (self.counts.get(key, 0) + self.pseudo_count) ** self.exponent

    @property
    def fingerprint(self) -> str:
        data = json.dumps(self.as_dict(), sort_keys=True).encode()
        return hashlib.sha256(data).hexdigest()[:12]

    def as_dict(self) -> dict:
        return {
            "kind": "topic_prior_from_original_text_v1",
            "counts": self.counts,
            "source_posts": self.source_posts,
            "matched_posts": self.matched_posts,
            "exponent": self.exponent,
            "pseudo_count": self.pseudo_count,
        }


def learn_topic_prior(rows: list[dict], *, exponent: float = .5, pseudo_count: float = 1.) -> TopicPrior:
    if not 0 <= exponent <= 1 or pseudo_count <= 0:
        raise ValueError("invalid smoothing")
    compiled = {key: re.compile(expr, re.IGNORECASE) for key, expr in PATTERNS.items()}
    counts: dict[str, int] = {key: 0 for key in PATTERNS}
    total, matched = 0, 0
    seen: set[str] = set()
    for row in rows:
        original = str(row.get("text_original") or "").strip()
        identity = str(row.get("tweet_id") or row.get("url") or "")
        if not original or not identity or identity in seen or original.startswith("RT @"):
            continue
        seen.add(identity)
        total += 1
        hits = [key for key, pattern in compiled.items() if pattern.search(original)]
        for key in hits:
            counts[key] += 1
        if hits:
            matched += 1
    if not total:
        raise ValueError("no original tweets supplied")
    return TopicPrior(counts=counts, matched_posts=matched, source_posts=total,
                      exponent=exponent, pseudo_count=pseudo_count)


def weighted_index(keys: list[str], prior: TopicPrior, fraction: float) -> int:
    """Draw from a prior with a supplied fly-derived fraction in [0,1)."""
    if not keys or not 0 <= fraction < 1:
        raise ValueError("invalid weighted selection")
    weights = [prior.weight(k) for k in keys]
    threshold = fraction * sum(weights)
    for idx, weight in enumerate(weights):
        threshold -= weight
        if threshold < 0:
            return idx
    return len(keys) - 1
