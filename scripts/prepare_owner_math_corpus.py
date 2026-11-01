"""Build a private, provenance-preserving math corpus from an owned X export.

This tool never unpacks media or uploads archive contents.
"""
from __future__ import annotations

import argparse
import json
import re
import zipfile
from collections import Counter
from pathlib import Path

MATH_MARKERS = re.compile(
    r"матем|теорем|доказат|интеграл|тополог|геометр|функци|"
    r"пространств|гомеомор|проектив|сходим|групп|изоморф|"
    r"множест|дифференц|аксиом|алгебр|фрактал|"
    r"math|theorem|proof|topolog|integral|geometry|homotop|"
    r"manifold|function|homology|category theory|"
    r"\\b(?:lean|latex|banach|cantor|riemann|hilbert|yoneda)\\b",
    flags=re.IGNORECASE,
)
CYRILLIC = re.compile(r"[А-Яа-яЁё]")
URL = re.compile(r"https?://\\S+")


def parse_archive_records(archive: Path) -> list[dict]:
    """Read only the official data/tweets.js, never any sensitive archive sections."""
    with zipfile.ZipFile(archive) as zf:
        names = [name for name in zf.namelist() if name.endswith("/tweets.js")]
        if len(names) != 1:
            raise ValueError("Expected exactly one data/tweets.js in X archive")
        raw = zf.read(names[0]).decode("utf-8-sig")
    marker = raw.find("=")
    if marker == -1:
        raise ValueError("Invalid X export: no JS assignment")
    payload = json.loads(raw[marker + 1:].strip().rstrip(";"))
    if not isinstance(payload, list):
        raise ValueError("Expected an array of tweet objects")
    return payload


def select_math_tweets(records: list[dict], *, handle: str, include_replies: bool = False) -> list[dict]:
    selected: list[dict] = []
    seen: set[str] = set()
    handle = handle.strip().lstrip("@")
    if not handle or not re.fullmatch(r"[A-Za-z0-9_]{1,15}", handle):
        raise ValueError("Invalid X handle")
    for entry in records:
        tweet = entry.get("tweet", {})
        if not isinstance(tweet, dict):
            continue
        text = str(tweet.get("full_text", "")).strip()
        tweet_id = str(tweet.get("id_str", "")).strip()
        if not tweet_id.isdecimal() or not text:
            continue
        if text.startswith("RT @"):
            continue
        if not include_replies and (tweet.get("in_reply_to_status_id_str") or tweet.get("in_reply_to_status_id")):
            continue
        # Require a recognizable mathematical topic, not merely generic smart-sounding words.
        if not MATH_MARKERS.search(URL.sub("", text)):
            continue
        clean = re.sub(r"\\s+", " ", text).casefold()
        if clean in seen:
            continue
        seen.add(clean)
        lang = "ru" if CYRILLIC.search(text) else "en"
        selected.append({
            "source": "owner_archive",
            "handle": handle,
            "tweet_id": tweet_id,
            "url": f"https://x.com/{handle}/status/{tweet_id}",
            "created_at": str(tweet.get("created_at", "")),
            "text_original": text,
            "language": lang,
            "text_en": text if lang == "en" else None,
            "translation_status": "original_en" if lang == "en" else "pending",
        })
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--handle", default="sphere_homotopy")
    parser.add_argument("--include-replies", action="store_true")
    args = parser.parse_args()

    records = parse_archive_records(args.archive)
    tweets = select_math_tweets(records, handle=args.handle, include_replies=args.include_replies)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as out:
        for tweet in tweets:
            out.write(json.dumps(tweet, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps({
        "archive_tweets": len(records),
        "selected": len(tweets),
        "languages": dict(Counter(t["language"] for t in tweets)),
        "output": str(args.output),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
