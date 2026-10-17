"""Settle observed X engagement without generating tweets or loading model weights."""
from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from fly_window.publishing.sheet_transport import load_rows, write_rows_atomic
from fly_window.text.reward import RewardConfig, apply_settled_rewards


def advance_observation_clock(path: Path, rows: list[dict[str, object]]) -> bool:
    """Advance only the real observation timestamp in existing runtime state."""
    observations = [
        datetime.fromisoformat(str(row["metrics_collected_at"]).replace("Z", "+00:00"))
        for row in rows if str(row.get("metrics_collected_at", "") or "").strip()
    ]
    if not observations:
        return False
    if any(value.tzinfo is None or value.utcoffset() is None for value in observations):
        raise ValueError("observation timestamp must be timezone-aware")
    latest = max(observations)
    if not path.exists():
        raise FileNotFoundError(f"runtime state is missing: {path}")
    state = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(state, dict):
        raise ValueError("runtime state must be a JSON object")
    old = state.get("last_stats_at")
    if old:
        previous = datetime.fromisoformat(str(old).replace("Z", "+00:00"))
        if previous.tzinfo is None or previous.utcoffset() is None:
            raise ValueError("last_stats_at must be timezone-aware")
        if latest <= previous:
            return False
    state["last_stats_at"] = latest.isoformat()
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
    temporary.replace(path)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows-json", required=True, type=Path)
    parser.add_argument("--reward-config", required=True, type=Path)
    parser.add_argument("--runtime-state", type=Path, help="Advance last_stats_at independently of generation")
    parser.add_argument("--now", default="", help="Timezone-aware ISO date for deterministic testing")
    args = parser.parse_args()
    now = datetime.fromisoformat(args.now.replace("Z", "+00:00")) if args.now else datetime.now(UTC)
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("--now must have a timezone")
    rows = load_rows(args.rows_json)
    changed = apply_settled_rewards(rows, config=RewardConfig.from_json(args.reward_config), now=now)
    if changed:
        write_rows_atomic(args.rows_json, rows)
    stats_advanced = advance_observation_clock(args.runtime_state, rows) if args.runtime_state else False
    print(json.dumps({"rewarded_rows": changed, "stats_advanced": stats_advanced, "unconsumed_rewards": sum(
        row.get("reward") not in (None, "") and not str(row.get("training_consumed_at", "") or "").strip()
        for row in rows
    )}, sort_keys=True))


if __name__ == "__main__":
    main()
