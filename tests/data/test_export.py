import json

import numpy as np
import pandas as pd

from fly_window.data.export import export_subgraph
from fly_window.data.schema import NormalizedTables
from fly_window.data.select import SubgraphConfig


def test_export_subgraph_round_trips_and_has_deterministic_semantic_hash(tmp_path):
    graph = NormalizedTables(
        nodes=pd.DataFrame(
            [
                (3, "DNa01", "descending_neuron", None, "R", "glutamate"),
                (1, "LC01", "visual_projection_neuron", None, "L", "acetylcholine"),
                (2, "MID", "interneuron", None, None, "gaba"),
            ],
            columns=["body_id", "type", "superclass", "cell_class", "side", "nt"],
        ),
        edges=pd.DataFrame([(2, 3, 7), (1, 2, 5)], columns=["src", "dst", "weight"]).astype(
            {"src": "int64", "dst": "int64", "weight": "int64"}
        ),
    )
    config = SubgraphConfig(max_hops=3, max_nodes=10)
    source_manifest = {
        "dataset": "male-cns:v1.0",
        "retrieved_at_utc": "2026-09-14T00:00:00+00:00",
        "sources": [
            {"source_name": "a.feather", "sha256": "a" * 64},
            {"source_name": "b.feather", "sha256": "b" * 64},
            {"source_name": "c.feather", "sha256": "c" * 64},
        ],
    }

    first = export_subgraph(graph, config, source_manifest, tmp_path / "first")
    second = export_subgraph(graph, config, source_manifest, tmp_path / "second")

    assert first["artifact_sha256"] == second["artifact_sha256"]
    assert first["node_count"] == 3
    assert first["edge_count"] == 2
    assert first["source_manifest"]["sources"][0]["sha256"] == "a" * 64

    nodes = pd.read_parquet(tmp_path / "first" / "subgraph_v1_nodes.parquet")
    edges = pd.read_parquet(tmp_path / "first" / "subgraph_v1_edges.parquet")
    archive = np.load(tmp_path / "first" / "subgraph_v1.npz")
    manifest = json.loads((tmp_path / "first" / "subgraph_v1_manifest.json").read_text())

    assert nodes["body_id"].tolist() == [1, 2, 3]
    assert edges.to_dict("records") == [
        {"src": 1, "dst": 2, "weight": 5},
        {"src": 2, "dst": 3, "weight": 7},
    ]
    assert archive["body_ids"].tolist() == [1, 2, 3]
    assert archive["src_idx"].tolist() == [0, 1]
    assert archive["dst_idx"].tolist() == [1, 2]
    assert archive["weights"].tolist() == [5, 7]
    assert archive["input_mask"].tolist() == [True, False, False]
    assert archive["output_mask"].tolist() == [False, False, True]
    assert manifest["artifact_sha256"] == first["artifact_sha256"]
    assert set(manifest["file_sha256"]) == {
        "subgraph_v1_nodes.parquet",
        "subgraph_v1_edges.parquet",
        "subgraph_v1.npz",
    }
