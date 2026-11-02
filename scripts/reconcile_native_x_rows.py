"""Reconcile independently verified native-X publications into Fly durable rows.

The native queue owns delivery truth; this does not call Buffer or guess success.
Keeps historical v1 text for provenance, and merges smoke posts as new rows.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def reconcile_native_posts(rows: list[dict], queue: list[dict]) -> tuple[list[dict], int]:
    result = [dict(row) for row in rows]
    lookup = {str(row.get("idempotency_key")): row for row in result}
    applied = 0
    for item in queue:
        if item.get("status") != "published":
            continue
        tweet_url = str(item.get("tweet_url", "")).strip()
        if not tweet_url.startswith("https://x.com/") or "/status/" not in tweet_url:
            continue
        source = str(item.get("source_idempotency_key", ""))
        if source not in lookup:
            continue
        if item.get("origin") == "manual_smoke":
            target_id = "v2-smoke:" + source
            if target_id in lookup:
                continue
            cloned = dict(lookup[source])
            cloned["idempotency_key"] = target_id
            cloned["batch_id"] = "v2-smoke"
            cloned["tweet_index"] = 0
            cloned["batch_size"] = 1
            cloned["fly_original_text"] = cloned.get("text", "")
            cloned["text"] = item["text"]
            cloned["scheduled_at"] = item["scheduled_at"]
            cloned["status"] = "published"
            cloned["published_at"] = item["posted_at"]
            cloned["tweet_url"] = tweet_url
            cloned["buffer_post_id"] = ""
            cloned["last_error"] = ""
            cloned["training_consumed_at"] = ""
            cloned["decoder"] = item["generator"]
            result.append(cloned)
            lookup[target_id] = cloned
            applied += 1
        else:
            target = lookup[source]
            if target.get("buffer_post_id"):
                continue  # Buffer owns a remote post: do not overwrite it.
            if target.get("tweet_url") == tweet_url and target.get("status") == "published":
                continue
            if target.get("status") not in ("generated", "paused", "published"):
                continue
            target.setdefault("fly_original_text", target.get("text", ""))
            target["text"] = item["text"]
            target["status"] = "published"
            target["tweet_url"] = tweet_url
            target["published_at"] = item["posted_at"]
            target["last_error"] = ""
            target["decoder"] = item["generator"]
            applied += 1
    return result, applied


def save_atomic(path: Path, value: list[dict]) -> None:
    tmp = path.with_suffix(path.suffix + ".native-tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows-json", type=Path, required=True)
    parser.add_argument("--queue-json", type=Path, required=True)
    args = parser.parse_args()
    rows = json.loads(args.rows_json.read_text(encoding="utf-8-sig"))
    queue = json.loads(args.queue_json.read_text(encoding="utf-8-sig"))
    if not isinstance(rows, list) or not isinstance(queue, list):
        raise ValueError("rows and queue must be lists")
    result, changed = reconcile_native_posts(rows, queue)
    if changed:
        save_atomic(args.rows_json, result)
    print(json.dumps({"reconciled": changed, "rows": len(result), "native_posts":
                      sum(row.get("status") == "published" for row in queue)}))


if __name__ == "__main__":
    main()
