from __future__ import annotations

import pytest

from fly_window.publishing.sheet_transport import (
    build_fly_tweets_request,
    complete_generated_batches,
    parse_fly_tweets_response,
)


def _row(
    batch: str,
    index: int,
    *,
    batch_size: int,
    status: str = "generated",
) -> dict[str, object]:
    return {
        "batch_id": batch,
        "batch_size": batch_size,
        "tweet_index": index,
        "idempotency_key": f"{batch}:{index}",
        "text": f"tweet {index}",
        "scheduled_at": f"2026-09-18T{8 + index:02d}:00:00+02:00",
        "status": status,
        "buffer_post_id": "",
    }


def test_complete_generated_batches_accepts_fly_selected_size() -> None:
    rows = [_row("b1", index, batch_size=3) for index in range(3)]
    rows += [_row("partial", index, batch_size=3) for index in range(2)]

    batches = complete_generated_batches(rows)

    assert list(batches) == ["b1"]
    assert [row["tweet_index"] for row in batches["b1"]] == [0, 1, 2]


def test_complete_generated_batches_ignores_rows_already_owned_by_buffer() -> None:
    rows = [_row("b1", index, batch_size=3) for index in range(3)]
    rows[1]["buffer_post_id"] = "buffer-1"

    assert complete_generated_batches(rows) == {}


def test_ingest_request_preserves_fly_selected_size_and_times() -> None:
    rows = [_row("b1", index, batch_size=3) for index in range(3)]

    payload = build_fly_tweets_request("ingest", rows)

    assert payload["action"] == "ingest"
    assert payload["rows"] == rows


def test_ingest_rejects_incomplete_batch() -> None:
    rows = [_row("b1", index, batch_size=3) for index in range(2)]
    with pytest.raises(ValueError, match="batch_size"):
        build_fly_tweets_request("ingest", rows)


def test_sync_request_contains_no_rows() -> None:
    assert build_fly_tweets_request("sync") == {"action": "sync"}


def test_parse_response_rejects_apps_script_error() -> None:
    with pytest.raises(RuntimeError, match="denied"):
        parse_fly_tweets_response({"ok": False, "error": "denied"})


def test_parse_response_requires_rows() -> None:
    with pytest.raises(RuntimeError, match="rows"):
        parse_fly_tweets_response({"ok": True, "result": {"scheduled": 3}})


def test_parse_response_returns_sheet_rows() -> None:
    rows = [_row("b1", index, batch_size=3, status="scheduled") for index in range(3)]
    assert parse_fly_tweets_response({"ok": True, "result": {"rows": rows}}) == rows
