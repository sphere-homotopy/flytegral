from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date
from typing import MutableMapping


_VALID_STATES = frozenset({"pending", "running", "complete", "failed"})


@dataclass(frozen=True, slots=True)
class DailyRequest:
    request_date: str
    state: str = "pending"
    attempts: int = 0
    checkpoint_id: str = ""
    batch_id: str = ""
    last_error: str = ""

    def __post_init__(self) -> None:
        try:
            parsed = date.fromisoformat(self.request_date)
        except ValueError as error:
            raise ValueError("request_date must be an ISO date") from error
        if parsed.isoformat() != self.request_date:
            raise ValueError("request_date must be a canonical ISO date")
        if self.state not in _VALID_STATES:
            raise ValueError(f"invalid request state: {self.state}")
        if self.attempts < 0:
            raise ValueError("attempts must be non-negative")
        if self.state == "complete" and (not self.checkpoint_id or not self.batch_id):
            raise ValueError("complete request requires checkpoint_id and batch_id")


def _date_key(value: date | str) -> str:
    if isinstance(value, date):
        return value.isoformat()
    text = str(value).strip()
    try:
        parsed = date.fromisoformat(text)
    except ValueError as error:
        raise ValueError("request date must be an ISO date") from error
    if parsed.isoformat() != text:
        raise ValueError("request date must be a canonical ISO date")
    return text


def ensure_pending_request(
    requests: MutableMapping[str, DailyRequest],
    request_date: date | str,
) -> DailyRequest:
    """Create an idempotent date-keyed request without mutating existing work."""
    key = _date_key(request_date)
    existing = requests.get(key)
    if existing is not None:
        if existing.request_date != key:
            raise ValueError("request mapping key does not match request_date")
        return existing

    created = DailyRequest(request_date=key)
    requests[key] = created
    return created


def claim_oldest_request(
    requests: MutableMapping[str, DailyRequest],
) -> DailyRequest | None:
    """Claim the oldest pending/failed date; completed/running work is skipped."""
    eligible = [
        request
        for request in requests.values()
        if request.state in {"pending", "failed"}
    ]
    if not eligible:
        return None

    oldest = min(eligible, key=lambda request: request.request_date)
    claimed = replace(
        oldest,
        state="running",
        attempts=oldest.attempts + 1,
        last_error="",
    )
    requests[oldest.request_date] = claimed
    return claimed


def fail_request(
    requests: MutableMapping[str, DailyRequest],
    request_date: date | str,
    error: str,
) -> DailyRequest:
    key = _date_key(request_date)
    current = requests.get(key)
    if current is None:
        raise KeyError(key)
    if current.state == "complete":
        raise ValueError("complete request is immutable")
    if current.state != "running":
        raise ValueError("only a running request can fail")

    message = str(error).strip()
    if not message:
        raise ValueError("failure error must not be empty")
    failed = replace(current, state="failed", last_error=message)
    requests[key] = failed
    return failed


def complete_request(
    requests: MutableMapping[str, DailyRequest],
    request_date: date | str,
    *,
    checkpoint_id: str,
    batch_id: str,
) -> DailyRequest:
    key = _date_key(request_date)
    current = requests.get(key)
    if current is None:
        raise KeyError(key)

    checkpoint = str(checkpoint_id).strip()
    batch = str(batch_id).strip()
    if not checkpoint or not batch:
        raise ValueError("checkpoint_id and batch_id are required")

    if current.state == "complete":
        if current.checkpoint_id == checkpoint and current.batch_id == batch:
            return current
        raise ValueError("complete request is immutable")
    if current.state != "running":
        raise ValueError("only a running request can complete")

    completed = replace(
        current,
        state="complete",
        checkpoint_id=checkpoint,
        batch_id=batch,
        last_error="",
    )
    requests[key] = completed
    return completed
