"""Merge locally reviewed translation drafts back onto canonical X-export rows.

Edits JSONL contains only tweet_id + text_en. Never overwrites originals.
All outputs remain local and are not training-approved.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from translate_owner_math_corpus import load_jsonl, save_jsonl_atomic


def apply_edits(source: list[dict], edits: list[dict]) -> list[dict]:
    by_id = {str(row["tweet_id"]): dict(row) for row in source}
    if len(by_id) != len(source):
        raise ValueError("Duplicate source tweet IDs")
    edited: set[str] = set()
    for entry in edits:
        key = str(entry["tweet_id"])
        if key in edited or key not in by_id:
            raise ValueError(f"Duplicate or unknown editor tweet_id: {key}")
        edited.add(key)
        text = str(entry["text_en"]).strip()
        if not text:
            raise ValueError(f"Blank translation: {key}")
        original = by_id[key]
        if original["language"] != "ru":
            raise ValueError(f"Only Russian originals can be edited: {key}")
        original["text_en"] = text
        original["translation_status"] = "editor_draft"
        original["training_approved"] = False
    return [by_id[str(row["tweet_id"])] for row in source]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--edits", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    source = load_jsonl(args.input)
    edits = load_jsonl(args.edits)
    merged = apply_edits(source, edits)
    save_jsonl_atomic(args.output, merged)
    print(json.dumps({"total": len(merged), "editor_drafts": len(edits), "output": str(args.output)}))


if __name__ == "__main__":
    main()
