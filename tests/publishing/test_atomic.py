from __future__ import annotations

import ctypes
import importlib
import importlib.util
from pathlib import Path
import sys
import threading

import pytest


def _atomic():
    assert importlib.util.find_spec("fly_window.publishing.atomic") is not None, (
        "bounded Windows atomic replacement helper is missing"
    )
    return importlib.import_module("fly_window.publishing.atomic")


def _permission_error(winerror: int | None) -> PermissionError:
    error = PermissionError("replacement denied")
    if winerror is not None:
        error.winerror = winerror
    return error


@pytest.mark.parametrize("winerror", [5, 32, 33])
def test_replacement_recovers_after_transient_windows_sharing_error(tmp_path, monkeypatch, winerror):
    atomic = _atomic()
    monkeypatch.setattr(atomic.sys, "platform", "win32")
    source, target = tmp_path / "state.tmp", tmp_path / "state.json"
    source.write_text("new", encoding="utf-8")
    target.write_text("old", encoding="utf-8")
    original_replace = Path.replace
    attempts = 0
    delays = []

    def replace(path, destination):
        nonlocal attempts
        attempts += 1
        if attempts <= 2:
            assert target.read_text(encoding="utf-8") == "old"
            raise _permission_error(winerror)
        return original_replace(path, destination)

    monkeypatch.setattr(Path, "replace", replace)
    monkeypatch.setattr(atomic.time, "sleep", delays.append)

    atomic.replace_with_retry(source, target)

    assert target.read_text(encoding="utf-8") == "new"
    assert not source.exists()
    assert attempts == 3
    assert delays == [0.05, 0.1]


def test_persistent_windows_access_error_is_bounded_and_keeps_old_file(tmp_path, monkeypatch):
    atomic = _atomic()
    monkeypatch.setattr(atomic.sys, "platform", "win32")
    source, target = tmp_path / "state.tmp", tmp_path / "state.json"
    source.write_text("new", encoding="utf-8")
    target.write_text("old", encoding="utf-8")
    attempts = 0
    delays = []
    failure = _permission_error(5)

    def denied(path, destination):
        nonlocal attempts
        attempts += 1
        raise failure

    monkeypatch.setattr(Path, "replace", denied)
    monkeypatch.setattr(atomic.time, "sleep", delays.append)

    with pytest.raises(PermissionError) as caught:
        atomic.replace_with_retry(source, target)

    assert caught.value is failure
    assert attempts == 7
    assert delays == [0.05, 0.1, 0.2, 0.4, 0.8, 1.6]
    assert sum(delays) < 4
    assert target.read_text(encoding="utf-8") == "old"
    assert source.read_text(encoding="utf-8") == "new"


@pytest.mark.parametrize("platform, failure", [
    ("linux", _permission_error(5)),
    ("win32", _permission_error(None)),
    ("win32", _permission_error(2)),
    ("win32", OSError("disk failure")),
])
def test_other_errors_propagate_without_retry(tmp_path, monkeypatch, platform, failure):
    atomic = _atomic()
    monkeypatch.setattr(atomic.sys, "platform", platform)
    source, target = tmp_path / "state.tmp", tmp_path / "state.json"
    source.write_text("new", encoding="utf-8")
    target.write_text("old", encoding="utf-8")
    attempts = 0
    delays = []

    def denied(path, destination):
        nonlocal attempts
        attempts += 1
        raise failure

    monkeypatch.setattr(Path, "replace", denied)
    monkeypatch.setattr(atomic.time, "sleep", delays.append)

    with pytest.raises(type(failure)) as caught:
        atomic.replace_with_retry(source, target)

    assert caught.value is failure
    assert attempts == 1
    assert delays == []
    assert target.read_text(encoding="utf-8") == "old"


@pytest.mark.skipif(sys.platform != "win32", reason="requires real Windows sharing semantics")
def test_real_windows_reader_releases_target_before_retry_limit(tmp_path):
    atomic = _atomic()
    source, target = tmp_path / "state.tmp", tmp_path / "state.json"
    source.write_text("new", encoding="utf-8")
    target.write_text("old", encoding="utf-8")
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateFileW.argtypes = [ctypes.c_wchar_p, ctypes.c_uint32, ctypes.c_uint32,
                                  ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32,
                                  ctypes.c_void_p]
    kernel.CreateFileW.restype = ctypes.c_void_p
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    kernel.CloseHandle.restype = ctypes.c_int
    # Allow reads and writes, but deliberately deny delete/replace sharing.
    handle = kernel.CreateFileW(str(target), 0x80000000, 0x1 | 0x2, None, 3, 0x80, None)
    assert handle != ctypes.c_void_p(-1).value, ctypes.get_last_error()
    closed = threading.Event()

    def release():
        kernel.CloseHandle(handle)
        closed.set()

    timer = None
    try:
        with pytest.raises(PermissionError) as caught:
            source.replace(target)
        assert caught.value.winerror in (5, 32, 33)
        assert target.read_text(encoding="utf-8") == "old"
        timer = threading.Timer(0.2, release)
        timer.start()

        atomic.replace_with_retry(source, target)

        assert closed.is_set()
        assert target.read_text(encoding="utf-8") == "new"
        assert not source.exists()
    finally:
        if timer is not None:
            timer.join(timeout=2)
        if not closed.is_set():
            release()
