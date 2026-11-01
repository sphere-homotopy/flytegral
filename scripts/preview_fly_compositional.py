"""Offline A/B surface comparison for v0 full-joke and v1 compositional decoders."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fly_window.text.compositional_engine import select_compositions
from fly_window.text.meme_engine import preview_candidates


def load_lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def save_lines(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n" for item in rows), encoding="utf-8")


def stats(rows: list[dict]) -> dict:
    return {
        "count": len(rows),
        "unique_topics": len({row["fly_topic"] for row in rows}),
        "unique_texts": len({row["text"] for row in rows}),
        "max_chars": max((len(row["text"]) for row in rows), default=0),
        "mean_chars": round(sum(len(row["text"]) for row in rows) / max(1, len(rows)), 1),
        "publishable": sum(bool(row.get("publishable")) for row in rows),
        "unique_scenes": len({row.get("fly_scene") for row in rows if row.get("fly_scene")}),
        "unique_moves": len({row.get("fly_move") for row in rows if row.get("fly_move")}),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--outbox", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--count", type=int, default=30)
    p.add_argument("--max-chars", type=int, default=240)
    args = p.parse_args()
    rows = load_lines(args.outbox)
    baseline = preview_candidates(rows, count=args.count, max_chars=args.max_chars)
    # Same source rows, no publication and no newly generated neural activity.
    composed = select_compositions(rows, count=args.count, max_chars=args.max_chars)
    save_lines(args.output, composed)
    report = {"outbox_rows": len(rows), "baseline_v0": stats(baseline), "compositional_v1": stats(composed)}
    report_path = args.output.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
