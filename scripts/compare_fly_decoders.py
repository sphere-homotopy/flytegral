"""Create blinded human A/B CSV or train an offline preference reranker."""
from __future__ import annotations

import argparse
import json
import sys
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fly_window.text.preference_lab import (
    PreferenceModel, make_blind_trials, read_reviews, write_trials,
)
from fly_window.text.compositional_engine import compose_fly_trace
from fly_window.text.style_profile import learn_style, styled_fly_trace
from fly_window.text.topic_prior import learn_topic_prior


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def main() -> None:
    p = argparse.ArgumentParser()
    modes = p.add_subparsers(dest="action", required=True)
    make = modes.add_parser("make")
    make.add_argument("--outbox", type=Path, required=True)
    make.add_argument("--review-csv", type=Path, required=True)
    make.add_argument("--count", type=int, default=30)
    make.add_argument("--owner-corpus", type=Path, help="Compare v1 against corpus-styled v2")
    make.add_argument("--topic-prior-from-owner", action="store_true")
    train = modes.add_parser("train")
    train.add_argument("--review-csv", type=Path, required=True)
    train.add_argument("--model-json", type=Path, required=True)
    args = p.parse_args()
    if args.action == "make":
        baseline, challenger = None, None
        if args.topic_prior_from_owner and not args.owner_corpus:
            p.error("--topic-prior-from-owner requires --owner-corpus")
        if args.owner_corpus:
            owner = read_jsonl(args.owner_corpus)
            style = learn_style(owner)
            prior = learn_topic_prior(owner) if args.topic_prior_from_owner else None
            baseline = compose_fly_trace
            challenger = partial(styled_fly_trace, profile=style, topic_prior=prior)
        public, metadata = make_blind_trials(read_jsonl(args.outbox), count=args.count,
                                              baseline=baseline, challenger=challenger)
        sidecar = write_trials(args.review_csv, public, metadata)
        print(json.dumps({"trials": len(public), "labels": 0, "review": str(args.review_csv),
                          "metadata": str(sidecar), "publication_enabled": False}))
        return
    metadata = read_jsonl(args.review_csv.with_suffix(".metadata.jsonl"))
    decisions = read_reviews(args.review_csv, metadata)
    if not decisions:
        print(json.dumps({"human_preferences": 0, "trained": False, "reason": "no explicit A/B choices"}))
        return
    model = PreferenceModel()
    model.train(decisions)
    args.model_json.parent.mkdir(parents=True, exist_ok=True)
    args.model_json.write_text(json.dumps(model.as_dict(), indent=2), encoding="utf-8")
    print(json.dumps({"human_preferences": len(decisions), "trained": True,
                      "model": str(args.model_json), "publication_enabled": False}))


if __name__ == "__main__":
    main()
