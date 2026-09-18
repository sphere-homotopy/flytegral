from __future__ import annotations

import argparse
import json
from pathlib import Path

from fly_window.publishing.row_store import merge_outbox_rows


def _load_rows(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if isinstance(payload, dict):
        payload = payload.get("rows")
    if not isinstance(payload, list) or any(not isinstance(row, dict) for row in payload):
        raise ValueError("rows-json must contain a JSON array of row objects")
    return [dict(row) for row in payload]


def _load_outbox(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    result: list[dict[str, object]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"outbox line {line_number} is not an object")
        result.append(dict(value))
    return result


def _write_atomic(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge Fly Tweets outbox into durable rows exactly once.")
    parser.add_argument("--rows-json", type=Path, required=True)
    parser.add_argument("--outbox-jsonl", type=Path, required=True)
    args = parser.parse_args()

    rows = _load_rows(args.rows_json)
    outbox = _load_outbox(args.outbox_jsonl)
    added = merge_outbox_rows(rows, outbox)
    _write_atomic(args.rows_json, rows)
    print(json.dumps({"added_rows": added, "total_rows": len(rows)}, sort_keys=True))


if __name__ == "__main__":
    main()
