from __future__ import annotations

import codecs
import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _load_script(name: str):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_flytweets_{name}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load script module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sync_outbox = _load_script("sync_fly_tweet_outbox")
daily_training = _load_script("run_daily_text_training")
buffer_publish = _load_script("publish_fly_tweets_buffer")
metrics_merge = _load_script("merge_fly_tweet_stats")


def _write_bom(path: Path, text: str) -> None:
    path.write_bytes(codecs.BOM_UTF8 + text.encode("utf-8"))


def test_persistent_row_loaders_accept_windows_powershell_utf8_bom(tmp_path: Path):
    rows_path = tmp_path / "rows.json"
    _write_bom(rows_path, '[{"idempotency_key":"batch:0","text":"bzz"}]\n')

    expected = [{"idempotency_key": "batch:0", "text": "bzz"}]
    assert sync_outbox._load_rows(rows_path) == expected
    assert daily_training._load_rows(rows_path) == expected
    assert buffer_publish._load_rows(rows_path) == expected
    assert metrics_merge._load_rows(rows_path) == expected


def test_persistent_jsonl_loaders_accept_utf8_bom(tmp_path: Path):
    outbox_path = tmp_path / "outbox.jsonl"
    _write_bom(outbox_path, '{"idempotency_key":"batch:0","text":"bzz"}\n')

    assert sync_outbox._load_outbox(outbox_path) == [
        {"idempotency_key": "batch:0", "text": "bzz"}
    ]
    assert daily_training._existing_outbox(outbox_path)["batch:0"]["text"] == "bzz"

    metrics_path = tmp_path / "metrics.jsonl"
    _write_bom(
        metrics_path,
        '{"tweet_url":"https://x.com/fly_topology/status/1","views":3}\n',
    )
    assert metrics_merge._load_jsonl(metrics_path)[0]["views"] == 3


def test_runtime_state_loader_accepts_utf8_bom(tmp_path: Path):
    state_path = tmp_path / "runtime-state.json"
    _write_bom(state_path, '{"current_checkpoint":"checkpoint-a"}\n')

    state = daily_training._load_runtime_state(state_path)

    assert state.current_checkpoint == "checkpoint-a"
