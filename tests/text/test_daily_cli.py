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
    ):
        assert option in result.stdout
    assert "--generation-count" not in result.stdout
