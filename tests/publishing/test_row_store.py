from fly_window.publishing.row_store import merge_outbox_rows


def test_outbox_rows_are_added_once_to_durable_store():
    rows = [{"idempotency_key": "old:0", "text": "old"}]
    outbox = [
        {"idempotency_key": "old:0", "text": "old"},
        {"idempotency_key": "new:0", "text": "bzz"},
    ]

    added = merge_outbox_rows(rows, outbox)

    assert added == 1
    assert [row["idempotency_key"] for row in rows] == ["old:0", "new:0"]


def test_conflicting_duplicate_key_is_rejected():
    rows = [{"idempotency_key": "same:0", "text": "first"}]
    outbox = [{"idempotency_key": "same:0", "text": "different"}]

    try:
        merge_outbox_rows(rows, outbox)
    except ValueError as error:
        assert "conflicting durable row" in str(error)
    else:
        raise AssertionError("expected conflicting durable row to fail")
