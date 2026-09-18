from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections import OrderedDict
from pathlib import Path
from typing import Any, Mapping, Sequence


def _batch_size(row: Mapping[str, object]) -> int | None:
    try:
        value = int(row.get("batch_size", 0))
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def _is_complete_batch(rows: Sequence[Mapping[str, object]]) -> bool:
    if not rows:
        return False
    sizes = {_batch_size(row) for row in rows}
    if len(sizes) != 1:
        return False
    batch_size = next(iter(sizes))
    if batch_size is None or len(rows) != batch_size:
        return False
    try:
        indexes = sorted(int(row.get("tweet_index", -1)) for row in rows)
    except (TypeError, ValueError):
        return False
    return indexes == list(range(batch_size))


def complete_generated_batches(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, list[dict[str, object]]]:
    grouped: OrderedDict[str, list[dict[str, object]]] = OrderedDict()
    for source in rows:
        row = dict(source)
        if str(row.get("status", "") or "").strip().lower() != "generated":
            continue
        if str(row.get("buffer_post_id", "") or "").strip():
            continue
        batch_id = str(row.get("batch_id", "") or "").strip()
        if not batch_id:
            continue
        grouped.setdefault(batch_id, []).append(row)

    result: dict[str, list[dict[str, object]]] = {}
    for batch_id, batch_rows in grouped.items():
        if not _is_complete_batch(batch_rows):
            continue
        ordered = sorted(batch_rows, key=lambda row: int(row["tweet_index"]))
        result[batch_id] = ordered
    return result


def build_fly_tweets_request(
    action: str,
    rows: Sequence[Mapping[str, object]] | None = None,
) -> dict[str, object]:
    normalized = str(action or "").strip().lower()
    if normalized not in {"ingest", "sync"}:
        raise ValueError(f"unsupported Fly Tweets action: {action}")
    if normalized == "sync":
        return {"action": "sync"}
    if rows is None:
        raise ValueError("ingest requires rows")
    payload_rows = [dict(row) for row in rows]
    if not _is_complete_batch(payload_rows):
        raise ValueError("ingest rows must form one complete fly-selected batch_size")
    batch_ids = {str(row.get("batch_id", "") or "").strip() for row in payload_rows}
    if len(batch_ids) != 1 or "" in batch_ids:
        raise ValueError("ingest rows must share one batch_id")
    return {"action": "ingest", "rows": payload_rows}

def parse_fly_tweets_response(payload: Mapping[str, Any]) -> list[dict[str, object]]:
    if not bool(payload.get("ok")):
        raise RuntimeError(str(payload.get("error", "Apps Script Fly Tweets request failed")))
    result = payload.get("result")
    if not isinstance(result, Mapping):
        raise RuntimeError("Apps Script response is missing result")
    rows = result.get("rows")
    if not isinstance(rows, list) or any(not isinstance(row, Mapping) for row in rows):
        raise RuntimeError("Apps Script response is missing rows")
    return [dict(row) for row in rows]


def post_fly_tweets_request(
    endpoint: str,
    payload: Mapping[str, object],
    *,
    timeout_seconds: float = 90.0,
) -> list[dict[str, object]]:
    url = str(endpoint or "").strip()
    if not url.startswith("https://script.google.com/"):
        raise ValueError("Fly Tweets endpoint must be an Apps Script HTTPS URL")
    request = urllib.request.Request(
        url,
        data=json.dumps(dict(payload), ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Apps Script HTTP {error.code}: {body}") from error
    value = json.loads(body)
    if not isinstance(value, Mapping):
        raise RuntimeError("Apps Script response is not a JSON object")
    return parse_fly_tweets_response(value)


def load_transport_endpoint(path: Path) -> str:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, Mapping):
        raise ValueError("transport config must be a JSON object")
    endpoint = str(payload.get("endpoint", "") or "").strip()
    if not endpoint.startswith("https://script.google.com/"):
        raise ValueError("transport config endpoint must be an Apps Script HTTPS URL")
    return endpoint


def load_rows(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if isinstance(payload, Mapping):
        payload = payload.get("rows")
    if not isinstance(payload, list) or any(not isinstance(row, Mapping) for row in payload):
        raise ValueError("rows-json must contain a JSON array of row objects")
    return [dict(row) for row in payload]


def write_rows_atomic(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps([dict(row) for row in rows], ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
