"""Production transport contract and isolated Windows scheduler handoff tests."""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github/workflows/fly-tweets-orchestrator.yml"
HANDOFF = "Prepare native queue through hidden Windows task"


def step_script(name: str) -> str:
    """Read one literal run block without interpreting GitHub expressions as YAML."""
    text = WORKFLOW.read_text(encoding="utf-8")
    pattern = rf"(?ms)^      - name: {re.escape(name)}\n(?P<body>.*?)(?=^      - |\Z)"
    match = re.search(pattern, text)
    if match is None:
        raise AssertionError(f"workflow step missing: {name}")
    lines = match["body"].splitlines()
    start = next((index + 1 for index, line in enumerate(lines) if line == "        run: |"), None)
    if start is None:
        raise AssertionError(f"literal run block missing: {name}")
    code = []
    for line in lines[start:]:
        if line and not line.startswith("          "):
            break
        code.append(line[10:] if line else "")
    return "\n".join(code)


class NativeOrchestratorContract(unittest.TestCase):
    def setUp(self) -> None:
        self.workflow = WORKFLOW.read_text(encoding="utf-8")

    def test_generation_uses_compatible_native_branch(self) -> None:
        self.assertTrue(re.search(r"uses: actions/checkout@v4\s+with:\s+ref: feature/fly-tweets-meme-v2", self.workflow), "generation must check out native-compatible source")

    def test_legacy_remote_transports_cannot_receive_new_batches(self) -> None:
        for legacy in ("scripts/sync_fly_tweets_sheet.py", "scripts/publish_fly_tweets_buffer.py", "Require Google Sheet transport configuration"):
            self.assertFalse(legacy in self.workflow, f"legacy publishing path remains: {legacy}")

    def test_native_queue_is_prepared_after_rows_and_before_success(self) -> None:
        sync = self.workflow.index("- name: Sync newly generated outbox into durable rows")
        self.assertTrue(f"- name: {HANDOFF}" in self.workflow, "native handoff must precede success marker")
        handoff = self.workflow.index(f"- name: {HANDOFF}")
        marker = self.workflow.index("- name: Record successful autonomous cycle")
        self.assertLess(sync, handoff)
        self.assertLess(handoff, marker)

    def test_neural_cadence_metrics_and_separate_training_remain(self) -> None:
        generation = step_script("Run cadence-owned daily cycle")
        training = step_script("Train Fly for future cycles (nonblocking)")
        self.assertIn("--cadence-checkpoint", generation)
        self.assertIn("--skip-training", generation)
        self.assertIn("--skip-generation", training)
        self.assertIn("--cadence-checkpoint", training)
        self.assertIn("Collect X stats headlessly", self.workflow)
        self.assertIn("Settle mature observed engagement", self.workflow)

    def test_marker_namespace_remains_consistent_and_documented(self) -> None:
        self.assertEqual(self.workflow.count("feature/fly-tweets-v1"), 3)
        self.assertIn("marker namespace", self.workflow)


@unittest.skipUnless(shutil.which("powershell"), "requires Windows PowerShell")
class NativeSchedulerHandoff(unittest.TestCase):
    def run_handoff(self, result: int = 0, initially_running: bool = False) -> subprocess.CompletedProcess[str]:
        # Scheduler calls are replaced at the process boundary. The actual workflow
        # run block executes unchanged; this harness never opens the browser/task.
        prelude = r"""
$ErrorActionPreference = 'Stop'
$script:started = $false
$script:startCalls = 0
$script:waitCalls = 0
$script:priorRunning = INITIAL_RUNNING
$script:afterStartRunning = $true
function Get-ScheduledTask {
  param($TaskName)
  if ($script:priorRunning) {
    $script:priorRunning = $false
    return [pscustomobject]@{State='Running'}
  }
  if ($script:started -and $script:afterStartRunning) {
    $script:afterStartRunning = $false
    return [pscustomobject]@{State='Running'}
  }
  return [pscustomobject]@{State='Ready'}
}
function Get-ScheduledTaskInfo {
  param($TaskName)
  $at = if ($script:started) { [DateTime]::Now.AddSeconds(1) } else { [DateTime]::Now.AddHours(-1) }
  return [pscustomobject]@{LastRunTime=$at; LastTaskResult=TASK_RESULT}
}
function Start-ScheduledTask {
  param($TaskName)
  if ($TaskName -ne 'FlyTweets-V2-Native-X') { throw 'unexpected task' }
  $script:startCalls++
  $script:started = $true
}
function Start-Sleep { param($Seconds); $script:waitCalls++ }
""".replace("INITIAL_RUNNING", "$true" if initially_running else "$false").replace("TASK_RESULT", str(result))
        suffix = "\n[pscustomobject]@{starts=$script:startCalls; waits=$script:waitCalls} | ConvertTo-Json -Compress"
        return subprocess.run(
            [shutil.which("powershell"), "-NoProfile", "-NonInteractive", "-Command", prelude + step_script(HANDOFF) + suffix],
            capture_output=True, text=True, timeout=15,
        )

    def test_waits_for_existing_task_then_observes_new_successful_run(self) -> None:
        run = self.run_handoff(initially_running=True)
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
        summary = json.loads(run.stdout.splitlines()[-1])
        self.assertEqual(summary["starts"], 1)
        self.assertGreaterEqual(summary["waits"], 2)

    def test_failed_native_preparation_blocks_success_marker(self) -> None:
        run = self.run_handoff(result=1)
        self.assertNotEqual(run.returncode, 0)
        self.assertIn("native", run.stderr.lower())


@unittest.skipUnless(shutil.which("powershell"), "requires Windows PowerShell")
class IndependentTrainingEligibility(unittest.TestCase):
    def test_windows_powershell_unrolls_rewarded_rows(self) -> None:
        # This has to exercise production PowerShell 5.1: PowerShell 7 silently
        # flattens ConvertFrom-Json in the old expression and hides this bug.
        powershell = Path(os.environ["WINDIR"]) / "System32/WindowsPowerShell/v1.0/powershell.exe"
        gate = step_script("Train Fly for future cycles (nonblocking)").split("if ($pending.Count -eq 0)", 1)[0]
        rows = [
            {"training_consumed_at": "", "reward": 0},
            {"training_consumed_at": "", "cadence_reward": 0, "cadence_context": "[1,2]"},
            {"training_consumed_at": "2026-11-01T00:00:00Z", "reward": 2},
            {"training_consumed_at": "", "reward": "", "cadence_reward": ""},
        ]
        with tempfile.TemporaryDirectory(prefix="fly training gate ") as folder:
            path = Path(folder) / "rows.json"
            path.write_text(json.dumps(rows), encoding="utf-8")
            env = dict(os.environ, FLYTWEETS_ROWS=str(path))
            run = subprocess.run(
                [str(powershell), "-NoProfile", "-NonInteractive", "-Command", gate + "\nWrite-Output $pending.Count"],
                capture_output=True, text=True, timeout=15, env=env,
            )
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
        self.assertEqual(run.stdout.strip(), "2", "both zero-valued reward types remain eligible under production PowerShell")


if __name__ == "__main__":
    unittest.main()
