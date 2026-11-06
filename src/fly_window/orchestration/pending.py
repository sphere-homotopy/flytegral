from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, replace
from datetime import date
from pathlib import Path
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


def _request_from_payload(key: str, payload: object) -> DailyRequest:
    if not isinstance(payload, dict):
        raise ValueError(f"request {key} must be a JSON object")
    required = {
        "request_date",
        "state",
        "attempts",
        "checkpoint_id",
        "batch_id",
        "last_error",
    }
    missing = sorted(required.difference(payload))
    if missing:
        raise ValueError(f"request {key} missing fields: {', '.join(missing)}")
    attempts = payload["attempts"]
    if isinstance(attempts, bool) or not isinstance(attempts, int):
        raise ValueError(f"request {key} attempts must be an integer")
    request = DailyRequest(
        request_date=str(payload["request_date"]),
        state=str(payload["state"]),
        attempts=attempts,
        checkpoint_id=str(payload["checkpoint_id"]),
        batch_id=str(payload["batch_id"]),
        last_error=str(payload["last_error"]),
    )
    if request.request_date != key:
        raise ValueError("request mapping key does not match request_date")
    return request


def load_requests(path: Path) -> dict[str, DailyRequest]:
    """Load and fully validate the durable date-keyed request ledger."""
    path = Path(path)
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid request ledger JSON: {error.msg}") from error
    if not isinstance(payload, dict):
        raise ValueError("request ledger must be a JSON object")
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported request ledger schema_version")
    raw_requests = payload.get("requests")
    if not isinstance(raw_requests, dict):
        raise ValueError("request ledger requests must be a JSON object")

    restored: dict[str, DailyRequest] = {}
    for raw_key, raw_request in raw_requests.items():
        key = _date_key(str(raw_key))
        restored[key] = _request_from_payload(key, raw_request)
    return restored


def save_requests_atomic(
    path: Path,
    requests: MutableMapping[str, DailyRequest],
) -> None:
    """Durably replace the ledger without exposing a partially written JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    serialized: dict[str, dict[str, object]] = {}
    for raw_key, request in sorted(requests.items()):
        key = _date_key(raw_key)
        if request.request_date != key:
            raise ValueError("request mapping key does not match request_date")
        serialized[key] = asdict(request)

    body = json.dumps(
        {"schema_version": 1, "requests": serialized},
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ) + "\n"

    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f"{path.name}.tmp-",
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            handle.write(body)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
        temp_path = None
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
