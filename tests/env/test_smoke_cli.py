import json
from pathlib import Path
import subprocess
import sys


def test_smoke_cli_seed_7_exits_successfully():
    repo_root = Path(__file__).resolve().parents[2]
    completed = subprocess.run(
        [sys.executable, "scripts/smoke_env.py", "--seed", "7"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )

    summary = json.loads(completed.stdout)
    assert summary["seed"] == 7
    assert summary["success"] is True
    assert 0 < summary["steps"] < 400
    assert summary["terminated"] is True
    assert summary["truncated"] is False
