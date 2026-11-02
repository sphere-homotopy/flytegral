from datetime import UTC, datetime, timedelta
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from scripts.queue_fly_v2_live import plan_delivery, save_atomic


def corpus():
    return [
        {"tweet_id": str(i), "translation_status": "original_en",
         "text_original": f"Topological group {i}", "text_en": f"Do mathematicians know group {i}?"}
        for i in range(15)
    ]


def raw(source: str, when: datetime, *, status="generated", posted_id=""):
    return {
        "idempotency_key": source, "text": "proof banana",
        "token_ids": json.dumps(list(range(1, 41))),
        "scheduled_at": when.isoformat(), "status": status,
        "buffer_post_id": posted_id,
    }


def test_existing_cadence_is_preserved_and_historical_never_reposted():
    now = datetime(2026, 11, 2, 6, 30, tzinfo=UTC)
    valid = raw("fresh", now + timedelta(hours=5))
    old = raw("past", now - timedelta(hours=3))
    sent = raw("sent", now + timedelta(hours=4), status="published")
    in_buffer = raw("in_buffer", now + timedelta(hours=5), posted_id="buffer-123")
    q, count = plan_delivery([old, sent, in_buffer, valid], corpus(), [], now=now)
    assert count == 1
    assert q[0]["id"] == "scheduled:fresh"
    assert q[0]["scheduled_at"] == valid["scheduled_at"]
    assert q[0]["generator"] == "fly_compositional_style_v2"
    assert len(q[0]["text"]) <= 240


def test_retrying_queue_does_not_change_posted_or_claimed_records():
    now = datetime(2026, 11, 2, 6, 30, tzinfo=UTC)
    item = raw("one", now + timedelta(hours=3))
    first, count = plan_delivery([item], corpus(), [], now=now)
    assert count == 1
    first[0]["status"] = "published"
    first[0]["tweet_url"] = "https://x.com/fly_topology/status/1"
    second, count2 = plan_delivery([item], corpus(), first, now=now + timedelta(minutes=2))
    assert count2 == 0
    assert second == first


def test_smoke_only_on_explicit_source_and_only_once():
    now = datetime(2026, 11, 2, 6, 30, tzinfo=UTC)
    old = raw("historical", now - timedelta(hours=20), status="published")
    q, n = plan_delivery([old], corpus(), [], now=now, smoke_source="historical")
    assert n == 1
    assert q[0]["origin"] == "manual_smoke"
    assert q[0]["status"] == "ready"
    assert q[0]["scheduled_at"] < now.isoformat()
    later, count = plan_delivery([old], corpus(), q, now=now + timedelta(hours=1), smoke_source="historical")
    assert count == 0
    assert later == q


def test_never_queues_without_actual_token_trace():
    now = datetime(2026, 11, 2, 6, 30, tzinfo=UTC)
    item = raw("bad", now + timedelta(hours=1))
    item["token_ids"] = "[]"
    result, new = plan_delivery([item], corpus(), [], now=now)
    assert result == [] and new == 0


def test_invalid_smoke_and_atomic_write(tmp_path):
    now = datetime(2026, 11, 2, 6, 30, tzinfo=UTC)
    with pytest.raises(ValueError, match="smoke source"):
        plan_delivery([], corpus(), [], now=now, smoke_source="unknown")
    path = tmp_path / "queue.json"
    save_atomic(path, [{"status": "ready"}])
    assert json.loads(path.read_text()) == [{"status": "ready"}]
