from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_stats_clock_advances_on_resumed_generation_without_double_reward(tmp_path: Path):
    rows_path = tmp_path / "rows.json"
    state_path = tmp_path / "runtime-state.json"
    before_state = {"current_checkpoint": "already-trained",
                    "queue_horizon_at": "2026-10-20T00:00:00+00:00",
                    "last_stats_at": "2026-10-16T17:44:57+00:00"}
    state_path.write_text(json.dumps(before_state), encoding="utf-8")
    rows = [{
        "idempotency_key": "old:0",
        "published_at": "2026-09-20T00:00:00+00:00",
        "metrics_collected_at": "2026-10-17T00:46:29.418Z",
        "views": 1516, "likes": 4, "reposts": 1, "replies": 0, "bookmarks": 0,
        "reward": 0.5, "training_consumed_at": "2026-10-14T21:55:53+00:00",
    }]
    rows_path.write_text(json.dumps(rows), encoding="utf-8")
    cmd = [sys.executable, str(ROOT / "scripts" / "settle_fly_tweet_rewards.py"),
           "--rows-json", str(rows_path),
           "--reward-config", str(ROOT / "configs" / "daily_reward_v1.json"),
           "--runtime-state", str(state_path),
           "--now", "2026-10-17T12:00:00+00:00"]
    first = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert first.returncode == 0, first.stderr
    assert json.loads(first.stdout) == {
        "rewarded_rows": 0, "stats_advanced": True, "unconsumed_rewards": 0,
    }
    after = json.loads(state_path.read_text(encoding="utf-8"))
    assert after["last_stats_at"] == "2026-10-17T00:46:29.418000+00:00"
    assert after["current_checkpoint"] == before_state["current_checkpoint"]
    assert after["queue_horizon_at"] == before_state["queue_horizon_at"]
    assert json.loads(rows_path.read_text(encoding="utf-8")) == rows
    second = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert second.returncode == 0, second.stderr
    assert json.loads(second.stdout)["stats_advanced"] is False
