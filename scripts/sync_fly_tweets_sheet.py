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
    write_rows_atomic,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Synchronize Fly Tweets durable rows through Google Sheet + Apps Script."
    )
    parser.add_argument("--action", choices=("sync", "ingest"), required=True)
    parser.add_argument("--rows-json", type=Path, required=True)
    parser.add_argument("--transport-config", type=Path, required=True)
    args = parser.parse_args()

    endpoint = load_transport_endpoint(args.transport_config)
    rows = load_rows(args.rows_json)

    if args.action == "sync":
        remote_rows = post_fly_tweets_request(
            endpoint,
            build_fly_tweets_request("sync"),
        )
        write_rows_atomic(args.rows_json, remote_rows)
        print(json.dumps({"action": "sync", "rows": len(remote_rows)}, sort_keys=True))
        return

    batches = complete_generated_batches(rows)
    if not batches:
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
        submitted += 1

    print(
        json.dumps(
            {"action": "ingest", "batches": submitted, "rows": len(remote_rows)},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
