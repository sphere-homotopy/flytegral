import copy

import pytest

from fly_window.publishing.row_store import merge_outbox_rows
from scripts.reconcile_native_x_rows import reconcile_native_posts


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


def _native_publication():
    generated = {**_generated(), "token_ids": "[10,20]"}
    queue = [{
        "source_idempotency_key": generated["idempotency_key"],
        "origin": "fly_cadence",
        "status": "published",
        "text": "Is this topology?",
        "tweet_url": "https://x.com/fly_topology/status/123",
        "posted_at": "2026-09-18T00:39:00Z",
        "generator": "fly_compositional_style_v2",
    }]
    rows, reconciled = reconcile_native_posts([generated], queue)
    assert reconciled == 1
    return generated, rows


def test_native_publication_survives_replayed_original_generation():
    generated, rows = _native_publication()
    rows[0].update(views=37, likes=4, training_consumed_at="2026-09-22T00:00:00Z")
    before = copy.deepcopy(rows)

    assert merge_outbox_rows(rows, [generated]) == 0
    assert merge_outbox_rows(rows, [generated]) == 0
    assert rows == before
    assert rows[0]["text"] == "Is this topology?"
    assert rows[0]["fly_original_text"] == generated["text"]


@pytest.mark.parametrize("field, new_value", [
    ("text", "different original generation"),
    ("scheduled_at", "2026-09-19T00:00:00Z"),
    ("token_ids", "[10,21]"),
])
def test_native_publication_still_rejects_generation_conflict_atomically(field, new_value):
    generated, rows = _native_publication()
    before = copy.deepcopy(rows)
    tampered = {**generated, field: new_value}

    with pytest.raises(ValueError, match=f"generation field: {field}"):
        merge_outbox_rows(rows, [_generated(index=1), tampered])
    assert rows == before


@pytest.mark.parametrize("field, new_value", [
    ("decoder", "unverified_decoder"),
    ("status", "generated"),
    ("tweet_url", "https://x.com/other_account/status/123"),
    ("tweet_url", "https://not-x.com/fly_topology/status/123"),
    ("tweet_url", "https://x.com/fly_topology/status/not-a-number"),
    ("tweet_url", "https://x.com/fly_topology/status/123/extra"),
    ("buffer_post_id", "buffer-owned"),
    ("fly_original_text", None),
    ("fly_original_text", ["bzzz topology"]),
])
def test_unverified_rendering_metadata_cannot_hide_changed_generation_text(field, new_value):
    generated, rows = _native_publication()
    rows[0][field] = new_value
    before = copy.deepcopy(rows)

    with pytest.raises(ValueError, match="generation field: text"):
        merge_outbox_rows(rows, [_generated(index=1), generated])
    assert rows == before


def test_native_publication_without_original_text_cannot_hide_generation_conflict():
    generated, rows = _native_publication()
    del rows[0]["fly_original_text"]

    with pytest.raises(ValueError, match="generation field: text"):
        merge_outbox_rows(rows, [generated])
