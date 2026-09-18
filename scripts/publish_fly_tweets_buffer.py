from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.request
from pathlib import Path

from fly_window.publishing.buffer_api import (
    BUFFER_GRAPHQL_ENDPOINT,
    build_create_post_request,
    build_find_matching_posts_request,
    build_post_status_request,
    find_matching_post,
    parse_create_post_response,
    parse_post_status_response,
)


def _load_rows(path: Path) -> list[dict[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if isinstance(payload, dict):
        payload = payload.get("rows")
    if not isinstance(payload, list) or any(not isinstance(row, dict) for row in payload):
        raise ValueError("rows-json must contain a JSON array of row objects")
    return [dict(row) for row in payload]


def _write_atomic(path: Path, rows: list[dict[str, object]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _graphql(api_key: str, payload: dict[str, object]) -> dict[str, object]:
    request = urllib.request.Request(
        BUFFER_GRAPHQL_ENDPOINT,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=45) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Buffer HTTP {error.code}: {body}") from error
    value = json.loads(body)
    if not isinstance(value, dict):
        raise RuntimeError("Buffer response is not a JSON object")
    return value


def _apply_remote_status(row: dict[str, object], status: dict[str, str]) -> bool:
    remote_status = status["status"].lower()
    if remote_status == "sent":
        row["status"] = "published"
        row["published_at"] = status["sentAt"]
        row["tweet_url"] = status["externalLink"]
        row["last_error"] = ""
        return True
    if remote_status:
        row["status"] = remote_status
    return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Schedule generated Fly Tweets at fly-selected absolute times using Buffer."
    )
    parser.add_argument("--rows-json", type=Path, required=True)
    parser.add_argument("--api-key", default=os.environ.get("BUFFER_API_KEY", ""))
    parser.add_argument("--channel-id", default=os.environ.get("BUFFER_CHANNEL_ID", ""))
    parser.add_argument(
        "--organization-id", default=os.environ.get("BUFFER_ORGANIZATION_ID", "")
    )
    args = parser.parse_args()

    api_key = str(args.api_key).strip()
    channel_id = str(args.channel_id).strip()
    organization_id = str(args.organization_id).strip()
    if not api_key:
        raise ValueError("Buffer API key is required via --api-key or BUFFER_API_KEY")
    if not channel_id:
        raise ValueError("Buffer channel id is required via --channel-id or BUFFER_CHANNEL_ID")
    if not organization_id:
        raise ValueError(
            "Buffer organization id is required via --organization-id or BUFFER_ORGANIZATION_ID"
        )

    rows = _load_rows(args.rows_json)
    scheduled = 0
    reconciled = 0
    failures = 0

    for row in rows:
        buffer_post_id = str(row.get("buffer_post_id", "") or "").strip()
        if buffer_post_id:
            try:
                status = parse_post_status_response(
                    _graphql(api_key, build_post_status_request(buffer_post_id))
                )
                if _apply_remote_status(row, status):
                    reconciled += 1
                _write_atomic(args.rows_json, rows)
            except Exception as error:
                row["last_error"] = str(error)
                failures += 1
                _write_atomic(args.rows_json, rows)
            continue

        if str(row.get("status", "") or "").strip() != "generated":
            continue
        try:
            text = str(row.get("text", "") or "")
            scheduled_at = str(row.get("scheduled_at", "") or "")
            lookup = build_find_matching_posts_request(
                organization_id=organization_id,
                channel_id=channel_id,
                scheduled_at=scheduled_at,
            )
            existing = find_matching_post(
                _graphql(api_key, lookup),
                text=text,
                scheduled_at=scheduled_at,
                channel_id=channel_id,
            )
            if existing is not None:
                row["buffer_post_id"] = existing["id"]
                _apply_remote_status(row, existing)
                if str(row.get("status", "")) == "generated":
                    row["status"] = existing["status"] or "scheduled"
                row["last_error"] = ""
                reconciled += 1
                _write_atomic(args.rows_json, rows)
                continue

            payload = build_create_post_request(
                text=text,
                scheduled_at=scheduled_at,
                idempotency_key=str(row.get("idempotency_key", "") or ""),
                channel_id=channel_id,
            )
            post_id, _ = parse_create_post_response(_graphql(api_key, payload))
            row["buffer_post_id"] = post_id
            row["status"] = "scheduled"
            row["last_error"] = ""
            scheduled += 1
            _write_atomic(args.rows_json, rows)
        except Exception as error:
            row["last_error"] = str(error)
            failures += 1
            _write_atomic(args.rows_json, rows)

    print(
        json.dumps(
            {
                "scheduled": scheduled,
                "reconciled": reconciled,
                "failures": failures,
                "rows": len(rows),
            },
            sort_keys=True,
        )
    )
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
