import json
from datetime import date

import pytest

from fly_window.orchestration.pending import (
    claim_oldest_request,
    complete_request,
    ensure_pending_request,
    fail_request,
    load_requests,
    save_requests_atomic,
)


def test_requests_roundtrip_with_states_and_attempt_counts(tmp_path):
    path = tmp_path / "daily-requests.json"
    requests = {}
    ensure_pending_request(requests, date(2026, 9, 17))
    first = claim_oldest_request(requests)
    complete_request(
        requests,
        first.request_date,
        checkpoint_id="daily-2026-09-17-a",
        batch_id="batch-2026-09-18-a",
    )
    ensure_pending_request(requests, date(2026, 9, 18))
    second = claim_oldest_request(requests)
    fail_request(requests, second.request_date, "collector unavailable")

    save_requests_atomic(path, requests)
    restored = load_requests(path)

    assert restored == requests
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert sorted(payload["requests"]) == ["2026-09-17", "2026-09-18"]


def test_load_rejects_mapping_key_that_disagrees_with_request_date(tmp_path):
    path = tmp_path / "daily-requests.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "requests": {
                    "2026-09-17": {
                        "request_date": "2026-09-18",
                        "state": "pending",
                        "attempts": 0,
                        "checkpoint_id": "",
                        "batch_id": "",
                        "last_error": "",
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="mapping key"):
        load_requests(path)


def test_atomic_save_preserves_previous_file_when_replace_fails(tmp_path, monkeypatch):
    path = tmp_path / "daily-requests.json"
    path.write_text('{"sentinel":true}\n', encoding="utf-8")
    requests = {}
    ensure_pending_request(requests, date(2026, 9, 17))

    def fail_replace(source, destination):
        raise OSError("simulated replace failure")

    monkeypatch.setattr("fly_window.orchestration.pending.os.replace", fail_replace)

    with pytest.raises(OSError, match="simulated replace failure"):
        save_requests_atomic(path, requests)

    assert path.read_text(encoding="utf-8") == '{"sentinel":true}\n'
    assert list(tmp_path.glob("daily-requests.json.tmp-*")) == []
