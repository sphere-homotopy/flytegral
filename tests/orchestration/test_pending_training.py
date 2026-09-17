from datetime import date

import pytest

from fly_window.orchestration.pending import (
    claim_oldest_request,
    complete_request,
    ensure_pending_request,
    fail_request,
)


def test_offline_days_accumulate_once_and_oldest_is_claimed_first():
    requests = {}
    for request_date in (date(2026, 9, 17), date(2026, 9, 18), date(2026, 9, 19)):
        ensure_pending_request(requests, request_date)
    ensure_pending_request(requests, date(2026, 9, 18))

    assert sorted(requests) == ["2026-09-17", "2026-09-18", "2026-09-19"]
    claimed = claim_oldest_request(requests)

    assert claimed.request_date == "2026-09-17"
    assert claimed.state == "running"
    assert claimed.attempts == 1


def test_failed_request_retries_same_idempotency_key_before_newer_days():
    requests = {}
    ensure_pending_request(requests, date(2026, 9, 17))
    ensure_pending_request(requests, date(2026, 9, 18))
    first = claim_oldest_request(requests)
    fail_request(requests, first.request_date, "transient X DOM failure")

    retry = claim_oldest_request(requests)

    assert retry.request_date == "2026-09-17"
    assert retry.state == "running"
    assert retry.attempts == 2


def test_completed_request_is_immutable_and_never_claimed_again():
    requests = {}
    ensure_pending_request(requests, date(2026, 9, 17))
    claimed = claim_oldest_request(requests)
    complete_request(
        requests,
        claimed.request_date,
        checkpoint_id="daily-2026-09-17-a",
        batch_id="batch-2026-09-18-a",
    )
    ensure_pending_request(requests, date(2026, 9, 17))

    completed = requests["2026-09-17"]
    assert completed.state == "complete"
    assert completed.checkpoint_id == "daily-2026-09-17-a"
    assert completed.batch_id == "batch-2026-09-18-a"
    assert claim_oldest_request(requests) is None

    with pytest.raises(ValueError, match="complete request is immutable"):
        complete_request(
            requests,
            "2026-09-17",
            checkpoint_id="different",
            batch_id="different",
        )
