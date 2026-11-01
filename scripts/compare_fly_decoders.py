"""Create blinded human A/B CSV or train an offline preference reranker."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fly_window.text.preference_lab import (
    PreferenceModel, make_blind_trials, read_reviews, write_trials,
)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def main() -> None:
    p = argparse.ArgumentParser()
    modes = p.add_subparsers(dest="action", required=True)
    make = modes.add_parser("make")
    make.add_argument("--outbox", type=Path, required=True)
    make.add_argument("--review-csv", type=Path, required=True)
    make.add_argument("--count", type=int, default=30)
    train = modes.add_parser("train")
    train.add_argument("--review-csv", type=Path, required=True)
    train.add_argument("--model-json", type=Path, required=True)
    args = p.parse_args()
    if args.action == "make":
        public, metadata = make_blind_trials(read_jsonl(args.outbox), count=args.count)
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
