"""Normalize third-party X collector rows as review-only mathematical references."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from urllib.parse import urlparse

from prepare_owner_math_corpus import MATH_MARKERS, URL


def original_author_matches(handle: str, tweet_url: str) -> bool:
    try:
        url = urlparse(tweet_url)
    except ValueError:
        return False
    path = [p for p in url.path.split("/") if p]
    return (url.netloc.lower() in {"x.com", "www.x.com", "twitter.com", "www.twitter.com"}
            and len(path) == 3 and path[0].lower() == handle.lower()
            and path[1].lower() == "status" and path[2].isdecimal())


def filter_external_rows(records: list[dict], *, handles: set[str]) -> list[dict]:
    accepted: list[dict] = []
    seen: set[str] = set()
    for row in records:
        handle = str(row.get("account", "")).strip().lstrip("@").lower()
        url = str(row.get("tweet_url", "")).split("?")[0]
        text = str(row.get("text", "")).strip()
        if handle not in handles or not original_author_matches(handle, url):
            continue
        # Scrapers lacking reply/quote evidence must not silently treat it as safe.
        if row.get("is_reply") is not False or row.get("is_quote") is not False:
            continue
        if row.get("is_repost") or text.startswith("RT @"):
            continue
        if not MATH_MARKERS.search(URL.sub("", text)):
            continue
        key = re.sub(r"\s+", " ", text).casefold()
        if key in seen:
            continue
        seen.add(key)
        accepted.append({
            "source": "third_party_candidate",
            "handle": handle,
            "url": url,
            "text_original": text,
            "language": "en",
            "translation_status": "original_en",
            "created_at": str(row.get("tweet_datetime", "")),
            "training_approved": False,
        })
    return accepted


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--handles", default="alcuintweets,mouse_math")
    args = p.parse_args()
    handles = {h.strip().lstrip("@").lower() for h in args.handles.split(",") if h.strip()}
    if not handles:
        p.error("--handles cannot be empty")
    records = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    output = filter_external_rows(records, handles=handles)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in output), encoding="utf-8")
    print(json.dumps({"source_rows": len(records), "candidate_math_rows": len(output), "training_approved": 0}))


if __name__ == "__main__":
    main()
