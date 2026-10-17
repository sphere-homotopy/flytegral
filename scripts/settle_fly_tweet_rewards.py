"""Settle observed X engagement without generating tweets or loading model weights."""
from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from fly_window.publishing.sheet_transport import load_rows, write_rows_atomic
from fly_window.text.reward import RewardConfig, apply_settled_rewards


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows-json", required=True, type=Path)
    parser.add_argument("--reward-config", required=True, type=Path)
    parser.add_argument("--now", default="", help="Timezone-aware ISO date for deterministic testing")
    args = parser.parse_args()
    now = datetime.fromisoformat(args.now.replace("Z", "+00:00")) if args.now else datetime.now(UTC)
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("--now must have a timezone")
    rows = load_rows(args.rows_json)
    changed = apply_settled_rewards(rows, config=RewardConfig.from_json(args.reward_config), now=now)
    if changed:
        write_rows_atomic(args.rows_json, rows)
    print(json.dumps({"rewarded_rows": changed, "unconsumed_rewards": sum(
        row.get("reward") not in (None, "") and not str(row.get("training_consumed_at", "") or "").strip()
        for row in rows
    )}, sort_keys=True))


if __name__ == "__main__":
    main()
