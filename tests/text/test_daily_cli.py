from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_daily_training_cli_exposes_durable_transaction_inputs():
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "run_daily_text_training.py"), "--help"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    for option in (
        "--request-date",
        "--attempt",
        "--current-checkpoint",
        "--anchor-checkpoint",
        "--cadence-checkpoint",
        "--runtime-state",
        "--runtime-config",
        "--rows-json",
        "--outbox-jsonl",
        "--run-root",
        "--reward-config",
        "--training-config",
        "--cadence-training-config",
        "--base-seed",
        "--skip-training",
        "--skip-generation",
    ):
        assert option in result.stdout
    assert "--generation-count" not in result.stdout


def test_published_without_observed_engagement_does_not_train():
    from datetime import UTC, datetime

    from fly_window.text.reward import RewardConfig
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "fly_daily_training_script", REPO_ROOT / "scripts" / "run_daily_text_training.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _apply_settled_rewards = module._apply_settled_rewards

    config = RewardConfig.from_json(REPO_ROOT / "configs" / "daily_reward_v1.json")
    now = datetime(2026, 10, 14, tzinfo=UTC)
    unobserved = {
        "idempotency_key": "old:0",
        "published_at": "2026-09-20T10:00:00+00:00",
        "metrics_collected_at": "",
        "reward": "",
    }
    rows = [unobserved]
    assert _apply_settled_rewards(rows, reward_config=config, now=now) == 0
    assert rows[0]["reward"] == ""

    observed = {
        **unobserved,
        "idempotency_key": "old:1",
        "metrics_collected_at": "2026-10-14T11:00:00+00:00",
        "views": 42,
        "likes": 2,
        "reposts": 0,
        "replies": 0,
        "bookmarks": 0,
    }
    rows = [unobserved.copy(), observed]
    assert _apply_settled_rewards(rows, reward_config=config, now=now) == 1
    assert rows[0]["reward"] == ""
    assert rows[1]["reward"] == 0.0
