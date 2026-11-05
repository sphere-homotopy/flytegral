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
LOCKED_STEPS = (
    "Sync previous generated outbox into durable rows",
    "Collect X stats headlessly",
    "Merge fresh metrics when available",
    "Settle mature observed engagement into durable rewards",
    "Run cadence-owned daily cycle",
    "Sync newly generated outbox into durable rows",
    "Train Fly for future cycles (nonblocking)",
)


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


def transaction_body(name: str) -> str:
    script = step_script(name)
    if "with_fly_state_lock.ps1" not in script:
        return script
    lines = script.splitlines()
    start = next(index + 1 for index, line in enumerate(lines) if "with_fly_state_lock.ps1 -Body {" in line)
    if lines[-1] != "}":
        raise AssertionError(f"lock must enclose full workflow transaction: {name}")
    return "\n".join(line[2:] if line.startswith("  ") else line for line in lines[start:-1])


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

    def test_all_rows_transactions_and_shared_profile_are_locked(self) -> None:
        for name in LOCKED_STEPS:
            with self.subTest(step=name):
                script = step_script(name)
                expected_path = "../scripts" if name == "Collect X stats headlessly" else "scripts"
                self.assertTrue(
                    re.fullmatch(r"\$ErrorActionPreference = 'Stop'\n& " + re.escape(expected_path) + r"/with_fly_state_lock\.ps1 -Body \{\n(?:  [^\n]*\n|\n)*\}", script.strip()),
                    f"full transaction must execute under shared state lock: {name}",
                )

    def test_task_handoff_does_not_hold_cross_process_state_lock(self) -> None:
        self.assertNotIn("with_fly_state_lock.ps1", step_script(HANDOFF))

    def test_native_failures_are_checked_inside_each_transaction(self) -> None:
        for name in LOCKED_STEPS:
            with self.subTest(step=name):
                body = transaction_body(name)
                if name == "Run cadence-owned daily cycle":
                    self.assertIn("$cycleExit = $LASTEXITCODE", body)
                    self.assertIn("if ($cycleExit -ne 0)", body)
                else:
                    self.assertIn("if ($LASTEXITCODE -ne 0)", body)

    def test_post_training_health_refresh_preserves_initial_marker_isolation(self) -> None:
        refresh_name = "Refresh health heartbeat after independent training"
        upload_name = "Upload refreshed health heartbeat"
        for name in (refresh_name, upload_name):
            self.assertTrue(f"- name: {name}" in self.workflow, f"post-training health step missing: {name}")
        marker = self.workflow.index("- name: Record successful autonomous cycle")
        training = self.workflow.index("- name: Train Fly for future cycles (nonblocking)")
        refresh = self.workflow.index(f"- name: {refresh_name}")
        upload = self.workflow.index(f"- name: {upload_name}")
        alert = self.workflow.index("- name: Deliver health alert when required")
        self.assertLess(marker, training)
        self.assertLess(training, refresh)
        self.assertLess(refresh, upload)
        self.assertLess(upload, alert)
        refresh_block = self.workflow[refresh:upload]
        upload_block = self.workflow[upload:alert]
        self.assertIn("continue-on-error: true", refresh_block)
        self.assertIn("always() && env.FLYTWEETS_HEARTBEAT != ''", refresh_block)
        self.assertIn("steps.train_only.outcome == 'success'", refresh_block)
        self.assertIn("steps.train_only.outcome == 'failure'", refresh_block)
        self.assertIn("continue-on-error: true", upload_block)
        self.assertIn("steps.refresh_health.outcome == 'success'", upload_block)
        self.assertIn("overwrite: true", upload_block)
        self.assertIn("name: fly-tweets-health-heartbeat", upload_block)
        self.assertIn("$state.last_training_at", step_script(refresh_name))


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
        gate = transaction_body("Train Fly for future cycles (nonblocking)").split("if ($pending.Count -eq 0)", 1)[0]
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


@unittest.skipUnless(shutil.which("powershell"), "requires Windows PowerShell")
class PostTrainingHealthSnapshot(unittest.TestCase):
    def test_refresh_reads_current_training_and_checkpoint_without_changing_marker(self) -> None:
        script = step_script("Refresh health heartbeat after independent training")
        powershell = Path(os.environ["WINDIR"]) / "System32/WindowsPowerShell/v1.0/powershell.exe"
        with tempfile.TemporaryDirectory(prefix="fly heartbeat ") as folder:
            root = Path(folder)
            state = root / "runtime-state.json"
            heartbeat = root / "health-heartbeat.json"
            marker = root / "success-marker.json"
            marker.write_text('{"run_id": 123}', encoding="utf-8")
            previous = {"generated_at": "2026-10-14T00:00:00Z", "last_training_at": "2026-10-14T00:00:00Z", "training_reference_at": "2026-10-14T00:00:00Z", "queue_horizon_at": "old", "last_stats_at": "old", "current_checkpoint": "old", "github_run_id": "123"}
            current = {"last_training_at": "2026-11-05T22:00:00Z", "queue_horizon_at": "2026-11-07T23:00:00Z", "last_stats_at": "2026-11-05T21:00:00Z", "current_checkpoint": "new-checkpoint"}
            heartbeat.write_text(json.dumps(previous), encoding="utf-8")
            state.write_text(json.dumps(current), encoding="utf-8")
            env = dict(os.environ, FLYTWEETS_STATE_DIR=str(root), FLYTWEETS_RUNTIME_STATE=str(state), FLYTWEETS_HEARTBEAT=str(heartbeat))
            run = subprocess.run([str(powershell), "-NoProfile", "-NonInteractive", "-Command", script], capture_output=True, text=True, timeout=15, env=env, creationflags=subprocess.CREATE_NO_WINDOW)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            refreshed = json.loads(heartbeat.read_text(encoding="utf-8-sig"))
            self.assertEqual(refreshed["last_training_at"], current["last_training_at"])
            self.assertEqual(refreshed["training_reference_at"], current["last_training_at"])
            self.assertEqual(refreshed["current_checkpoint"], "new-checkpoint")
            self.assertEqual(refreshed["github_run_id"], "123")
            self.assertEqual(marker.read_text(encoding="utf-8"), '{"run_id": 123}')


if __name__ == "__main__":
    unittest.main()
