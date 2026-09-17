from __future__ import annotations

import argparse
import json
from pathlib import Path

from fly_window.publishing.sheet_transport import (
    build_fly_tweets_request,
    complete_generated_batches,
    load_rows,
    load_transport_endpoint,
    post_fly_tweets_request,
    queue_horizon_from_rows,
    write_rows_atomic,
)


def _refresh_runtime_horizon(path: Path | None, rows: list[dict[str, object]]) -> None:
    if path is None:
        return
    payload: dict[str, object] = {}
    if path.exists():
        value = json.loads(path.read_text(encoding="utf-8-sig"))
        if not isinstance(value, dict):
            raise ValueError("runtime-state must contain a JSON object")
        payload = dict(value)
    horizon = queue_horizon_from_rows(rows)
    if horizon is None:
        return
    payload["queue_horizon_at"] = horizon.isoformat()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Synchronize Fly Tweets durable rows through Google Sheet + Apps Script."
    )
    parser.add_argument("--action", choices=("sync", "ingest"), required=True)
    parser.add_argument("--rows-json", type=Path, required=True)
    parser.add_argument("--transport-config", type=Path, required=True)
    parser.add_argument("--runtime-state", type=Path)
    args = parser.parse_args()

    endpoint = load_transport_endpoint(args.transport_config)
    rows = load_rows(args.rows_json)

    if args.action == "sync":
        remote_rows = post_fly_tweets_request(
            endpoint,
            build_fly_tweets_request("sync"),
        )
        write_rows_atomic(args.rows_json, remote_rows)
        _refresh_runtime_horizon(args.runtime_state, remote_rows)
        print(json.dumps({"action": "sync", "rows": len(remote_rows)}, sort_keys=True))
        return

    batches = complete_generated_batches(rows)
    if not batches:
        _refresh_runtime_horizon(args.runtime_state, rows)
        print(json.dumps({"action": "ingest", "batches": 0, "rows": len(rows)}, sort_keys=True))
        return

    remote_rows = rows
    submitted = 0
    for batch_rows in batches.values():
        remote_rows = post_fly_tweets_request(
            endpoint,
            build_fly_tweets_request("ingest", batch_rows),
        )
        write_rows_atomic(args.rows_json, remote_rows)
        _refresh_runtime_horizon(args.runtime_state, remote_rows)
        submitted += 1

    print(
        json.dumps(
            {"action": "ingest", "batches": submitted, "rows": len(remote_rows)},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
