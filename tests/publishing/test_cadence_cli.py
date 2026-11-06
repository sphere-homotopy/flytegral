from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_init_cadence_cli_exposes_durable_artifact_inputs():
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "init_cadence.py"), "--help"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    for flag in ("--config", "--graph-npz", "--nodes-parquet", "--output", "--seed"):
        assert flag in result.stdout
