"""Build idempotent X-native delivery queue for fly-composed v2 posts.

The source is the existing durable Fly outbox. Schedules are immutable except
one explicitly requested smoke tweet; no Buffer API is needed.
"""
from __future__ import annotations

import argparse
from datetime import UTC, datetime, timedelta
import json
from pathlib import Path

from fly_window.text.style_profile import learn_style, styled_fly_trace
from fly_window.text.topic_prior import learn_topic_prior
from fly_window.text.meme_engine import lexical_similarity


def _rows(path: Path) -> list[dict]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if isinstance(value, dict):
        value = value.get("rows")
    if not isinstance(value, list) or not all(isinstance(r, dict) for r in value):
        raise ValueError("rows must be a JSON list of dicts")
    return value


def _originals(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def _due(value: str) -> datetime:
    stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if stamp.tzinfo is None:
        raise ValueError("scheduled_at must have timezone")
    return stamp.astimezone(UTC)


def plan_delivery(rows: list[dict], corpus: list[dict], previous: list[dict], *,
                  now: datetime, max_future_hours: float = 120.,
                  smoke_source: str = "") -> tuple[list[dict], int]:
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("now must have timezone")
    if max_future_hours <= 0:
        raise ValueError("invalid horizon")
    existing_ids = {str(row["source_idempotency_key"]) for row in previous}
    existing_keys = {str(row["id"]) for row in previous}
    existing_text = [str(row["text"]) for row in previous]
    profile = learn_style(corpus)
    prior = learn_topic_prior(corpus)
    queue = [dict(row) for row in previous]
    created = 0
    target = (str(smoke_source) or "").strip()
    if target and any(row.get("id") == "smoke:" + target for row in previous):
        target = ""  # An existing smoke record MUST NOT be overwritten.
    for row in rows:
        source_id = str(row.get("idempotency_key") or "")
        smoke = bool(target and source_id == target)
        if not source_id or (source_id in existing_ids and not smoke):
            continue
        if smoke and "smoke:" + source_id in existing_keys:
            continue
        if not smoke:
            if str(row.get("status")) != "generated" or row.get("buffer_post_id"):
                continue
            try:
                when = _due(str(row.get("scheduled_at") or ""))
            except ValueError:
                continue
            if not now + timedelta(seconds=15) < when <= now + timedelta(hours=max_future_hours):
                continue
        else:
            when = now - timedelta(seconds=1)
        try:
            rendered = styled_fly_trace(row, profile, topic_prior=prior)
        except (ValueError, TypeError, json.JSONDecodeError):
            if smoke:
                raise
            continue
        text = rendered["text"]
        if any(text == seen for seen in existing_text):
            continue
        if not smoke and any(lexical_similarity(text, seen) >= .78 for seen in existing_text):
            continue
        entry = {
            "id": ("smoke:" if smoke else "scheduled:") + source_id,
            "source_idempotency_key": source_id,
            "text": text,
            "scheduled_at": when.isoformat(),
            "status": "ready",
            "tweet_url": "",
            "attempt_started_at": "",
            "posted_at": "",
            "last_error": "",
            "generator": "fly_compositional_style_v2",
            "fly_topic": rendered["fly_topic"],
            "fly_scene": rendered["fly_scene"],
            "fly_move": rendered["fly_move"],
            "style_fingerprint": rendered["style_fingerprint"],
            "topic_prior_fingerprint": rendered["topic_prior_fingerprint"],
            "origin": "manual_smoke" if smoke else "fly_cadence",
        }
        queue.append(entry)
        created += 1
        existing_keys.add(entry["id"])
        if not smoke:
            existing_ids.add(source_id)
        existing_text.append(text)
    if target and not any(row.get("id") == "smoke:" + target for row in queue):
        raise ValueError(f"smoke source not found or duplicate: {target}")
    return sorted(queue, key=lambda x: (x["scheduled_at"], x["id"])), created


def save_atomic(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    intermediate = path.with_suffix(path.suffix + ".tmp")
    intermediate.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    intermediate.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows-json", type=Path, required=True)
    parser.add_argument("--owner-corpus", type=Path, required=True)
    parser.add_argument("--queue-json", type=Path, required=True)
    parser.add_argument("--smoke-source", default="")
    args = parser.parse_args()
    previous = _rows(args.queue_json) if args.queue_json.exists() else []
    now = datetime.now(UTC)
    queue, created = plan_delivery(_rows(args.rows_json), _originals(args.owner_corpus),
                                   previous, now=now, smoke_source=args.smoke_source)
    if created:
        save_atomic(args.queue_json, queue)
    print(json.dumps({"new": created, "queue": len(queue),
                      "ready": sum(row["status"] == "ready" for row in queue),
                      "due": sum(row["status"] == "ready"
                                 and _due(row["scheduled_at"]) <= now for row in queue),
                      "smoke_source": args.smoke_source}, sort_keys=True))


if __name__ == "__main__":
    main()
