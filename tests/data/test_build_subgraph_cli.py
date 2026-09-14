from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys


def _load_script_module():
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "build_subgraph.py"
    spec = importlib.util.spec_from_file_location("build_subgraph_script", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_enables_streaming_for_official_aggregated_connectivity(tmp_path, monkeypatch):
    module = _load_script_module()

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "source-manifest.json").write_text(
        json.dumps({"dataset": "male-cns:v1.0", "sources": []}),
        encoding="utf-8",
    )

    config_path = tmp_path / "subgraph.json"
    config_path.write_text(
        json.dumps(
            {
                "min_edge_weight": 3,
                "max_hops": 4,
                "max_nodes": 6000,
                "output_superclass": "descending_neuron",
                "visual_type_prefixes": ["LC"],
            }
        ),
        encoding="utf-8",
    )

    calls = []

    class DummyTables:
        pass

    def fake_load(*args, **kwargs):
        calls.append((args, kwargs))
        return DummyTables()

    monkeypatch.setattr(module, "load_malecns_tables", fake_load)
    monkeypatch.setattr(module, "select_task_subgraph", lambda tables, config: tables)
    monkeypatch.setattr(
        module,
        "_summary",
        lambda graph, config: {
            "input_seed_count": 0,
            "output_seed_count": 0,
            "node_count": 0,
            "edge_count": 0,
        },
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_subgraph.py",
            "--raw-dir",
            str(raw_dir),
            "--config",
            str(config_path),
            "--output-dir",
            str(tmp_path / "out"),
            "--dry-run",
        ],
    )

    module.main()

    assert len(calls) == 1
    _, kwargs = calls[0]
    assert kwargs["min_edge_weight"] == 3
    assert kwargs["connectivity_rows_are_aggregated"] is True
