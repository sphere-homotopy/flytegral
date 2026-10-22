import copy

import pytest

from fly_window.publishing.row_store import merge_outbox_rows


def _generated(index=0, text="bzzz topology"):
    return {
        "idempotency_key": f"batch-2026-09-17-a1:{index}",
        "batch_id": "batch-2026-09-17-a1",
        "tweet_index": index,
        "text": text,
        "status": "generated",
        "buffer_post_id": "",
        "tweet_url": "",
        "views": "",
        "training_consumed_at": "",
        "scheduled_at": "2026-09-18T00:38:26+00:00",
    }


def test_september_regression_scheduled_rows_survive_replayed_outbox():
    original = _generated()
    durable = {**original, "status": "scheduled", "buffer_post_id": "6aac7cd5c1a151b207b4effc"}
    rows = [durable]
    snapshot = copy.deepcopy(rows)

    assert merge_outbox_rows(rows, [original]) == 0
    assert merge_outbox_rows(rows, [original]) == 0
    assert rows == snapshot


def test_replayed_generation_never_overwrites_publication_metrics_or_training():
    generated = _generated()
    durable = {
        **generated,
        "status": "published",
        "buffer_post_id": "buffer-id",
        "tweet_url": "https://x.com/fly_topology/status/123",
        "views": 37,
        "likes": 4,
        "training_consumed_at": "2026-09-22T00:00:00Z",
    }
    rows = [durable]
    assert merge_outbox_rows(rows, [generated]) == 0
    assert rows[0] == durable


def test_outbox_replay_can_add_unseen_rows_only_once():
    rows = [_generated(index=0)]
    outbox = [_generated(index=0), _generated(index=1)]
    assert merge_outbox_rows(rows, outbox) == 1
    assert merge_outbox_rows(rows, outbox) == 0
    assert [row["idempotency_key"] for row in rows] == [
        "batch-2026-09-17-a1:0",
        "batch-2026-09-17-a1:1",
    ]


@pytest.mark.parametrize("field, new_value", [
    ("text", "different text"),
    ("scheduled_at", "2026-09-19T00:00:00Z"),
    ("batch_id", "wrong batch"),
])
def test_immutable_generation_conflict_is_diagnostic_and_atomic(field, new_value):
    rows = [_generated(index=0)]
    before = copy.deepcopy(rows)
    tampered = {**_generated(index=0), field: new_value}
    with pytest.raises(ValueError, match=f"generation field: {field}"):
        merge_outbox_rows(rows, [_generated(index=1), tampered])
    assert rows == before


def test_partial_legacy_rows_compare_only_available_generation_fields():
    rows = [{"idempotency_key": "legacy:0", "text": "original", "status": "published"}]
    assert merge_outbox_rows(
        rows, [{"idempotency_key": "legacy:0", "text": "original", "status": "generated", "rng_seed": 42}]
    ) == 0
    assert rows[0]["status"] == "published"


def test_conflicting_replayed_outbox_generation_rejected():
    rows = []
    with pytest.raises(ValueError, match="generation field: text"):
        merge_outbox_rows(rows, [_generated(), _generated(text="different")])
    assert rows == []


def test_duplicate_durable_rows_cannot_be_silently_kept():
    with pytest.raises(ValueError, match="duplicate durable row"):
        merge_outbox_rows([_generated(), _generated()], [])
