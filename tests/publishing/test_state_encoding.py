from __future__ import annotations

import codecs
from pathlib import Path

from scripts.merge_fly_tweet_stats import _load_jsonl as load_metrics_jsonl
from scripts.merge_fly_tweet_stats import _load_rows as load_metrics_rows
from scripts.publish_fly_tweets_buffer import _load_rows as load_buffer_rows
from scripts.run_daily_text_training import _existing_outbox, _load_rows as load_daily_rows
from scripts.run_daily_text_training import _load_runtime_state
from scripts.sync_fly_tweet_outbox import _load_outbox, _load_rows as load_sync_rows


def _write_bom(path: Path, text: str) -> None:
    path.write_bytes(codecs.BOM_UTF8 + text.encode("utf-8"))


def test_persistent_row_loaders_accept_windows_powershell_utf8_bom(tmp_path: Path):
    rows_path = tmp_path / "rows.json"
    _write_bom(rows_path, '[{"idempotency_key":"batch:0","text":"bzz"}]\n')

    expected = [{"idempotency_key": "batch:0", "text": "bzz"}]
    assert load_sync_rows(rows_path) == expected
    assert load_daily_rows(rows_path) == expected
    assert load_buffer_rows(rows_path) == expected
    assert load_metrics_rows(rows_path) == expected


def test_persistent_jsonl_loaders_accept_utf8_bom(tmp_path: Path):
    outbox_path = tmp_path / "outbox.jsonl"
    _write_bom(outbox_path, '{"idempotency_key":"batch:0","text":"bzz"}\n')

    assert _load_outbox(outbox_path) == [{"idempotency_key": "batch:0", "text": "bzz"}]
    assert _existing_outbox(outbox_path)["batch:0"]["text"] == "bzz"

    metrics_path = tmp_path / "metrics.jsonl"
    _write_bom(metrics_path, '{"tweet_url":"https://x.com/fly_topology/status/1","views":3}\n')
    assert load_metrics_jsonl(metrics_path)[0]["views"] == 3


def test_runtime_state_loader_accepts_utf8_bom(tmp_path: Path):
    state_path = tmp_path / "runtime-state.json"
    _write_bom(state_path, '{"current_checkpoint":"checkpoint-a"}\n')

    state = _load_runtime_state(state_path)

    assert state.current_checkpoint == "checkpoint-a"
