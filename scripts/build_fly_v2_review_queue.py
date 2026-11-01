"""Prepare a local CSV review sheet. Nothing is accepted into training automatically."""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

REVIEW_COLUMNS = (
    "tweet_id", "url", "language", "original_text", "draft_en", "translation_status",
    "review_decision", "review_notes",
)
URL = re.compile(r"https?://\S+", re.IGNORECASE)


def choose_review_rows(rows: list[dict], *, max_chars: int, count: int) -> list[dict]:
    if count < 1 or max_chars < 20:
        raise ValueError("count/max_chars invalid")
    candidates: list[tuple[int, dict]] = []
    seen: set[str] = set()
    for row in rows:
        en = str(row.get("text_en") or "").strip()
        if not en or len(en) > max_chars:
            continue
        key = " ".join(en.lower().split())
        if key in seen:
            continue
        seen.add(key)
        original = str(row.get("text_original") or "")
        status = str(row.get("translation_status") or "")
        if status not in {"machine_draft", "editor_draft", "original_en", "reviewed"}:
            continue
        priority = (0 if status == "editor_draft" else 1 if status == "original_en" else 2)
        priority += 3 if URL.search(en) else 0
        priority += 2 if len(en) > 200 else 0
        candidates.append((priority, {
            "tweet_id": str(row["tweet_id"]), "url": str(row.get("url") or ""),
            "language": str(row["language"]), "original_text": original, "draft_en": en,
            "translation_status": status, "review_decision": "",
            "review_notes": "",
        }))
    return [entry for _, entry in sorted(candidates, key=lambda item: item[0])[:count]]


def write_review_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=REVIEW_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--max-chars", type=int, default=280)
    args = parser.parse_args()
    rows = [json.loads(x) for x in args.input.read_text(encoding="utf-8").splitlines() if x.strip()]
    selected = choose_review_rows(rows, max_chars=args.max_chars, count=args.count)
    write_review_csv(args.output, selected)
    print(json.dumps({"total": len(rows), "review_rows": len(selected), "output": str(args.output),
                      "approved": 0}))


if __name__ == "__main__":
    main()
