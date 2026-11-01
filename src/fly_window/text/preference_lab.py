"""Offline blinded A/B review and pairwise preference fitting.

No implicit positive labels, no publishing, no connection to Fly weights.
Real human decisions are necessary before the preference model can learn.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

from fly_window.text.compositional_engine import compose_fly_trace
from fly_window.text.meme_engine import decode_fly_trace, lexical_similarity

HEADER = ("pair_id", "text_a", "text_b", "preferred", "review_notes")
PREFERENCES = {"A", "B", "tie", "skip"}


def _identity(obj: object) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=True).encode()).hexdigest()[:18]


def make_blind_trials(rows: list[dict], *, count: int = 30) -> tuple[list[dict], list[dict]]:
    """Both candidates originate from the SAME actual Fly token trajectory."""
    if count < 1:
        raise ValueError("count must be positive")
    public: list[dict] = []
    metadata: list[dict] = []
    compared: list[str] = []
    used_ids: set[str] = set()
    for row in reversed(rows):
        try:
            old = decode_fly_trace(row, max_chars=240)
            new = compose_fly_trace(row, max_chars=240)
        except (ValueError, KeyError, TypeError, json.JSONDecodeError):
            continue
        source = str(row.get("idempotency_key", ""))
        trial_id = _identity([source, old["text"], new["text"]])
        if trial_id in used_ids or old["text"] == new["text"]:
            continue
        if any(lexical_similarity(new["text"], existing) >= 0.73 for existing in compared):
            continue
        compared.append(new["text"])
        used_ids.add(trial_id)
        # Order is deterministic and balanced in expectation; hides the decoder names.
        new_goes_first = int(trial_id, 16) % 2 == 0
        a, b = (new, old) if new_goes_first else (old, new)
        public.append({"pair_id": trial_id, "text_a": a["text"], "text_b": b["text"],
                       "preferred": "", "review_notes": ""})
        metadata.append({"pair_id": trial_id, "a": a, "b": b,
                         "same_fly_source": source, "approved_for_publishing": False})
        if len(public) >= count:
            break
    return public, metadata


def write_trials(path: Path, public: list[dict], metadata: list[dict]) -> Path:
    if [row["pair_id"] for row in public] != [row["pair_id"] for row in metadata]:
        raise ValueError("review and metadata trial IDs mismatch")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=HEADER)
        writer.writeheader()
        writer.writerows(public)
    sidecar = path.with_suffix(".metadata.jsonl")
    sidecar.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in metadata),
                       encoding="utf-8")
    return sidecar


def read_reviews(path: Path, metadata: list[dict]) -> list[tuple[dict, dict]]:
    candidates = {row["pair_id"]: row for row in metadata}
    decisions: list[tuple[dict, dict]] = []
    seen: set[str] = set()
    with path.open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            pair_id, value = row["pair_id"], row["preferred"].strip()
            if pair_id in seen or pair_id not in candidates:
                raise ValueError(f"duplicate or unknown trial: {pair_id}")
            seen.add(pair_id)
            if value and value not in PREFERENCES:
                raise ValueError(f"invalid answer in {pair_id}: {value}")
            if value in ("A", "B"):
                match = candidates[pair_id]
                yes, no = (match["a"], match["b"]) if value == "A" else (match["b"], match["a"])
                decisions.append((yes, no))
    return decisions


def features(candidate: dict) -> dict[str, float]:
    """Interpretable, sparse features; never use raw tweet text in a model."""
    text = candidate["text"]
    parts = {
        f"decoder:{candidate['decoder']}": 1.,
        f"topic:{candidate['fly_topic']}": 1.,
        f"scene:{candidate.get('fly_scene', 'whole_joke')}": 1.,
        f"move:{candidate.get('fly_move', candidate.get('fly_voice', 'unknown'))}": 1.,
        "question": float(text.endswith("?")),
        "first_person": float(" i " in f" {text.lower()} " or text.lower().startswith("i ")),
        "length": min(len(text), 280) / 280.,
    }
    return parts


@dataclass
class PreferenceModel:
    weights: dict[str, float] = field(default_factory=dict)
    comparisons: int = 0

    def score(self, candidate: dict) -> float:
        return sum(v * self.weights.get(k, 0.) for k, v in features(candidate).items())

    def train(self, comparisons: list[tuple[dict, dict]], *, epochs: int = 50,
              learning_rate: float = .1, l2: float = .01) -> None:
        if epochs < 1 or not 0 < learning_rate <= 1 or not 0 <= l2 < 1:
            raise ValueError("invalid training hyperparameters")
        # Never invent labels: with zero human decisions the model stays empty.
        if not comparisons:
            return
        for _ in range(epochs):
            for preferred, rejected in comparisons:
                f_good, f_bad = features(preferred), features(rejected)
                delta = {k: f_good.get(k, 0.) - f_bad.get(k, 0.) for k in f_good.keys() | f_bad.keys()}
                margin = max(-30., min(30., sum(self.weights.get(k, 0.) * v for k, v in delta.items())))
                gradient_scale = 1. / (1. + math.exp(margin))
                for key, diff in delta.items():
                    self.weights[key] = (1. - learning_rate * l2) * self.weights.get(key, 0.) + learning_rate * gradient_scale * diff
        self.comparisons += len(comparisons)

    def as_dict(self) -> dict:
        return {"algorithm": "pairwise_logistic_linear_v1",
                "human_comparisons": self.comparisons,
                "weights": dict(sorted(self.weights.items()))}


def rank_candidates(rows: list[dict], model: PreferenceModel) -> list[dict]:
    """Rank, not rewrite; no model is promoted by this function."""
    return sorted(rows, key=lambda item: (-model.score(item), item["text"]))
