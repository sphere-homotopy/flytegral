from fly_window.publishing.metrics_merge import merge_collected_metrics


def test_collected_metrics_merge_by_tweet_url_without_dropping_queue_fields():
    rows = [
        {
            "idempotency_key": "batch-a:0",
            "text": "bzz theorem",
            "tweet_url": "https://x.com/fly/status/123",
            "views": "",
            "likes": "",
            "reposts": "",
            "replies": "",
            "bookmarks": "",
            "metrics_collected_at": "",
            "training_consumed_at": "",
        }
    ]
    metrics = [
        {
            "tweet_url": "https://x.com/fly/status/123",
            "views": 100,
            "likes": 7,
            "reposts": 3,
            "replies": 2,
            "bookmarks": 1,
            "collected_at": "2026-09-17T18:00:00.000Z",
        }
    ]

    updated = merge_collected_metrics(rows, metrics)

    assert updated == 1
    assert rows[0]["text"] == "bzz theorem"
    assert rows[0]["views"] == 100
    assert rows[0]["likes"] == 7
    assert rows[0]["reposts"] == 3
    assert rows[0]["replies"] == 2
    assert rows[0]["bookmarks"] == 1
    assert rows[0]["metrics_collected_at"] == "2026-09-17T18:00:00.000Z"


def test_metrics_for_unknown_tweet_are_ignored():
    rows = [{"idempotency_key": "batch-a:0", "tweet_url": "https://x.com/fly/status/123"}]
    metrics = [{"tweet_url": "https://x.com/fly/status/999", "views": 10}]

    assert merge_collected_metrics(rows, metrics) == 0
