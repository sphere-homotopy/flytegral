"""Compare v1 vs owner-style-calibrated v2 from the SAME real Fly token traces."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fly_window.text.compositional_engine import select_compositions
from fly_window.text.style_profile import learn_style, select_styled
from fly_window.text.topic_prior import learn_topic_prior


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def summary(rows: list[dict]) -> dict:
    return {
        "n": len(rows),
        "mean_chars": round(sum(len(x["text"]) for x in rows) / len(rows), 1) if rows else None,
        "question_fraction": round(sum("?" in x["text"] for x in rows) / len(rows), 3) if rows else None,
        "unique_texts": len({x["text"] for x in rows}),
        "topics": len({x["fly_topic"] for x in rows}),
        "scenes": len({x.get("fly_scene") for x in rows}),
        "moves": len({x.get("fly_move") for x in rows}),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--outbox", type=Path, required=True)
    p.add_argument("--owner-corpus", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--count", type=int, default=30)
    p.add_argument("--topic-prior-from-owner", action="store_true", help="Bias topics using math terms in owner originals")
    args = p.parse_args()
    owner = read_jsonl(args.owner_corpus)
    profile = learn_style(owner)
    topic_prior = learn_topic_prior(owner) if args.topic_prior_from_owner else None
    source = read_jsonl(args.outbox)
    v1 = select_compositions(source, count=args.count)
    v2 = select_styled(source, profile, count=args.count, topic_prior=topic_prior)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in v2), encoding="utf-8")
    report = {"owner_profile": profile.as_dict(), "topic_prior": topic_prior.as_dict() if topic_prior is not None else None, "v1": summary(v1), "v2_styled": summary(v2),
              "samples_from_real_fly": len(source), "human_funniness_labels": 0}
    args.output.with_suffix(".report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
