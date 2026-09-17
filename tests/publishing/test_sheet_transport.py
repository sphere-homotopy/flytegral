from __future__ import annotations

import pytest

from fly_window.publishing.sheet_transport import (
    build_fly_tweets_request,
    complete_generated_batches,
    parse_fly_tweets_response,
)


def _row(batch: str, index: int, *, status: str = "generated") -> dict[str, object]:
    return {
        "batch_id": batch,
        "tweet_index": index,
        "idempotency_key": f"{batch}:{index}",
        "text": f"tweet {index}",
        "status": status,
        "buffer_post_id": "",
    }


def test_complete_generated_batches_requires_exactly_ten_indices() -> None:
    rows = [_row("b1", index) for index in range(10)]
    rows += [_row("partial", index) for index in range(9)]

    batches = complete_generated_batches(rows)

    assert list(batches) == ["b1"]
    assert [row["tweet_index"] for row in batches["b1"]] == list(range(10))


def test_complete_generated_batches_ignores_rows_already_owned_by_buffer() -> None:
    rows = [_row("b1", index) for index in range(10)]
    rows[3]["buffer_post_id"] = "buffer-3"

    assert complete_generated_batches(rows) == {}


def test_ingest_request_contains_one_complete_batch() -> None:
    rows = [_row("b1", index) for index in range(10)]

    payload = build_fly_tweets_request("ingest", rows)

    assert payload["action"] == "ingest"
    assert payload["rows"] == rows


def test_sync_request_contains_no_rows() -> None:
    assert build_fly_tweets_request("sync") == {"action": "sync"}


def test_parse_response_rejects_apps_script_error() -> None:
    with pytest.raises(RuntimeError, match="denied"):
        parse_fly_tweets_response({"ok": False, "error": "denied"})


def test_parse_response_requires_rows() -> None:
    with pytest.raises(RuntimeError, match="rows"):
        parse_fly_tweets_response({"ok": True, "result": {"scheduled": 10}})


def test_parse_response_returns_sheet_rows() -> None:
    rows = [_row("b1", index, status="scheduled") for index in range(10)]
    assert parse_fly_tweets_response({"ok": True, "result": {"rows": rows}}) == rows
