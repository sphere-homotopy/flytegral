import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from reconcile_native_x_rows import reconcile_native_posts


def raw(source, *, status="generated", buffer_id=""):
    return {"idempotency_key": source, "text": "old gibberish", "status": status,
            "buffer_post_id": buffer_id, "tweet_url": "", "published_at": "",
            "training_consumed_at": ""}


def posted(source, origin="fly_cadence", status="published", url="https://x.com/fly_topology/status/123"):
    return {"source_idempotency_key": source, "id": "scheduled:" + source,
            "origin": origin, "status": status, "tweet_url": url,
            "text": "Real v2 joke", "scheduled_at": "2026-11-02T07:00:00+00:00",
            "posted_at": "2026-11-02T07:01:00+00:00",
            "generator": "fly_compositional_style_v2"}


def test_scheduled_post_reconciles_and_preserves_neural_source():
    result, n = reconcile_native_posts([raw("fresh")], [posted("fresh")])
    assert n == 1 and len(result) == 1
    assert result[0]["status"] == "published"
    assert result[0]["text"] == "Real v2 joke"
    assert result[0]["fly_original_text"] == "old gibberish"
    assert result[0]["tweet_url"].endswith("/123")
    again, count = reconcile_native_posts(result, [posted("fresh")])
    assert count == 0 and again == result


def test_smoke_creates_separate_immutable_historical_record():
    result, n = reconcile_native_posts([raw("old", status="paused")], [posted("old", origin="manual_smoke")])
    assert n == 1 and len(result) == 2
    assert result[0]["status"] == "paused"
    assert result[1]["status"] == "published"
    assert result[1]["idempotency_key"] == "v2-smoke:old"
    assert result[1]["tweet_url"].endswith("/123")
    result2, new = reconcile_native_posts(result, [posted("old", origin="manual_smoke")])
    assert new == 0 and result2 == result


def test_only_verified_native_x_permalink_is_applied():
    for item in (posted("fresh", status="uncertain"),
                 posted("fresh", url="https://not-x.com/foo/status/123"),
                 posted("fresh", url="")):
        result, n = reconcile_native_posts([raw("fresh")], [item])
        assert n == 0 and result[0]["status"] == "generated"


def test_never_overwrite_buffer_accepted_post():
    result, n = reconcile_native_posts([raw("fresh",buffer_id="buffer-123")], [posted("fresh")])
    assert n == 0 and result[0]["text"] == "old gibberish"
