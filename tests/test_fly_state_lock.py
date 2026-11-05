from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import shutil
import subprocess
import time

import pytest


POWERSHELL = shutil.which("powershell.exe")
LOCK_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "with_fly_state_lock.ps1"
pytestmark = pytest.mark.skipif(POWERSHELL is None, reason="Windows PowerShell is required")


def _quote(value: object) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _command(body: str, *, timeout: int = 3) -> str:
    return f"& {_quote(LOCK_SCRIPT)} -TimeoutSeconds {timeout} -Body {{ {body} }}"


def _start(script: str) -> subprocess.Popen[str]:
    return subprocess.Popen(
        [POWERSHELL, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
         "-Command", "$ErrorActionPreference = 'Stop'; " + script],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )


def _run(script: str) -> subprocess.CompletedProcess[str]:
    process = _start(script)
    stdout, stderr = process.communicate(timeout=15)
    return subprocess.CompletedProcess(process.args, process.returncode, stdout, stderr)


def _wait_for(path: Path, process: subprocess.Popen[str]) -> None:
    deadline = time.monotonic() + 8
    while not path.exists():
        if process.poll() is not None:
            stdout, stderr = process.communicate()
            pytest.fail(f"PowerShell exited before {path.name}: {stdout}\n{stderr}")
        if time.monotonic() >= deadline:
            pytest.fail(f"PowerShell did not create {path.name}")
        time.sleep(0.02)


@contextmanager
def _holding_process(tmp_path: Path, *, exception: bool = False):
    entered = tmp_path / "entered"
    release = tmp_path / "release"
    wait = (
        f"while (-not (Test-Path -LiteralPath {_quote(release)})) {{ Start-Sleep -Milliseconds 20 }}"
    )
    if exception:
        script = (
            "try { " + _command("throw 'body-failed'") + " } "
            "catch { "
            "if ($_.Exception.Message -ne 'body-failed') { throw }; "
            f"Set-Content -LiteralPath {_quote(entered)} -Value 'caught'; {wait} "
            "}"
        )
    else:
        script = _command(f"Set-Content -LiteralPath {_quote(entered)} -Value 'holding'; {wait}")
    process = _start(script)
    try:
        _wait_for(entered, process)
        yield process
    finally:
        release.write_text("release", encoding="utf-8")
        try:
            process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate()


def test_lock_serializes_independent_windows_powershell_processes(tmp_path):
    second_started = tmp_path / "second-started"
    second_entered = tmp_path / "second-entered"
    with _holding_process(tmp_path):
        second = _start(
            f"Set-Content -LiteralPath {_quote(second_started)} -Value 'waiting'; "
            + _command(f"Set-Content -LiteralPath {_quote(second_entered)} -Value 'entered'")
        )
        try:
            _wait_for(second_started, second)
            time.sleep(0.2)
            assert not second_entered.exists()
            assert second.poll() is None
        finally:
            (tmp_path / "release").write_text("release", encoding="utf-8")
            stdout, stderr = second.communicate(timeout=8)
        assert second.returncode == 0, stdout + stderr
        assert second_entered.exists()


def test_lock_releases_after_body_exception_while_owner_process_stays_alive(tmp_path):
    with _holding_process(tmp_path, exception=True) as owner:
        result = _run(_command("Write-Output 'reacquired'", timeout=1))
        assert owner.poll() is None
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "reacquired"


def test_lock_wait_has_bounded_timeout(tmp_path):
    with _holding_process(tmp_path) as owner:
        started = time.monotonic()
        result = _run(_command("Write-Output 'must-not-run'", timeout=1))
        elapsed = time.monotonic() - started
        assert owner.poll() is None
        assert result.returncode != 0
        assert "Timed out waiting for Fly Tweets durable state" in result.stderr
        assert "must-not-run" not in result.stdout
        assert 0.9 <= elapsed < 5


def test_lock_preserves_body_output_and_native_exit_code():
    result = _run(_command("& $env:ComSpec /c exit 23; Write-Output 'body-output'") + "; exit $LASTEXITCODE")
    assert result.stdout.strip() == "body-output"
    assert result.returncode == 23, result.stderr


def test_lock_releases_on_early_exit():
    result = _run(_command("exit 0; throw 'early exit was ignored'"))
    assert result.returncode == 0, result.stderr
    after = _run(_command("Write-Output 'after-exit'", timeout=1))
    assert after.returncode == 0, after.stderr
    assert after.stdout.strip() == "after-exit"


def test_lock_accepts_abandoned_mutex(tmp_path):
    entered = tmp_path / "abandoned-entered"
    waiter_started = tmp_path / "abandoned-waiter-started"
    owner = _start(
        r"$mutex = [System.Threading.Mutex]::new($false, 'Local\FlyTweets-DurableState'); "
        "$null = $mutex.WaitOne(); "
        f"Set-Content -LiteralPath {_quote(entered)} -Value 'holding'; Start-Sleep -Seconds 30"
    )
    waiter = None
    try:
        _wait_for(entered, owner)
        # Retain a handle while the owner exits so Windows marks the existing
        # mutex abandoned rather than destroying it before the next opener.
        waiter = _start(
            r"$keepAlive = [System.Threading.Mutex]::new($false, 'Local\FlyTweets-DurableState'); "
            f"Set-Content -LiteralPath {_quote(waiter_started)} -Value 'waiting'; "
            "try { " + _command("Write-Output 'after-abandon'")
            + " } finally { $keepAlive.Dispose() }"
        )
        _wait_for(waiter_started, waiter)
        owner.kill()
        owner.communicate(timeout=5)
        stdout, stderr = waiter.communicate(timeout=8)
        assert waiter.returncode == 0, stderr
        assert stdout.strip() == "after-abandon"
    finally:
        if owner.poll() is None:
            owner.kill()
            owner.communicate()
        if waiter is not None and waiter.poll() is None:
            waiter.kill()
            waiter.communicate()


def test_native_pipeline_waits_for_shared_lock_before_touching_state(tmp_path):
    native_started = tmp_path / "native-started"
    state_dir = tmp_path / "flytegral" / "fly-tweets" / "state"
    native_script = LOCK_SCRIPT.with_name("run_fly_v2_native.ps1")
    with _holding_process(tmp_path):
        native = _start(
            f"$env:LOCALAPPDATA = {_quote(tmp_path)}; "
            f"Set-Content -LiteralPath {_quote(native_started)} -Value 'started'; "
            f"& {_quote(native_script)}"
        )
        try:
            _wait_for(native_started, native)
            time.sleep(0.2)
            assert native.poll() is None
            assert not state_dir.exists()
        finally:
            (tmp_path / "release").write_text("release", encoding="utf-8")
            stdout, stderr = native.communicate(timeout=8)
        assert native.returncode != 0
        assert "Fly runtime rows.json missing" in stderr, stdout + stderr
