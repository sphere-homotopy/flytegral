from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from .schema import NormalizedTables
from .select import SubgraphConfig


_ARRAY_ORDER = ("body_ids", "src_idx", "dst_idx", "weights", "input_mask", "output_mask")


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _semantic_hash(metadata: dict, arrays: dict[str, np.ndarray]) -> str:
    digest = sha256()
    canonical = json.dumps(metadata, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    digest.update(canonical)
    for name in _ARRAY_ORDER:
        array = np.ascontiguousarray(arrays[name])
        digest.update(name.encode("utf-8"))
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(json.dumps(array.shape, separators=(",", ":")).encode("ascii"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def export_subgraph(
    graph: NormalizedTables,
    config: SubgraphConfig,
    source_manifest: dict,
    output_dir: Path,
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    nodes = graph.nodes.sort_values("body_id", kind="stable").reset_index(drop=True).copy()
    edges = graph.edges.sort_values(["src", "dst"], kind="stable").reset_index(drop=True).copy()

    body_ids = nodes["body_id"].astype("int64").to_numpy()
    index_by_body = {int(body_id): idx for idx, body_id in enumerate(body_ids)}
    unknown = (set(edges["src"].astype(int)) | set(edges["dst"].astype(int))).difference(index_by_body)
    if unknown:
        raise ValueError(f"Edges reference unknown body ids: {sorted(unknown)[:10]}")

    src_idx = np.asarray([index_by_body[int(value)] for value in edges["src"]], dtype=np.int64)
    dst_idx = np.asarray([index_by_body[int(value)] for value in edges["dst"]], dtype=np.int64)
    weights = edges["weight"].astype("int64").to_numpy()
    input_mask = nodes["type"].fillna("").astype(str).str.startswith(config.visual_type_prefixes).to_numpy(dtype=np.bool_)
    output_mask = nodes["superclass"].fillna("").astype(str).eq(config.output_superclass).to_numpy(dtype=np.bool_)

    arrays = {
        "body_ids": body_ids,
        "src_idx": src_idx,
        "dst_idx": dst_idx,
        "weights": weights,
        "input_mask": input_mask,
        "output_mask": output_mask,
    }

    source_identity = [
        {"source_name": item.get("source_name"), "sha256": item.get("sha256")}
        for item in source_manifest.get("sources", [])
    ]
    semantic_metadata = {
        "dataset": source_manifest.get("dataset"),
        "sources": source_identity,
        "config": asdict(config),
        "node_count": int(len(nodes)),
        "edge_count": int(len(edges)),
    }
    artifact_sha256 = _semantic_hash(semantic_metadata, arrays)

    nodes_path = output_dir / "subgraph_v1_nodes.parquet"
    edges_path = output_dir / "subgraph_v1_edges.parquet"
    archive_path = output_dir / "subgraph_v1.npz"
    manifest_path = output_dir / "subgraph_v1_manifest.json"

    nodes.to_parquet(nodes_path, index=False)
    edges.to_parquet(edges_path, index=False)
    np.savez(archive_path, **arrays)

    file_sha256 = {
        nodes_path.name: _file_sha256(nodes_path),
        edges_path.name: _file_sha256(edges_path),
        archive_path.name: _file_sha256(archive_path),
    }
    manifest = {
        **semantic_metadata,
        "source_manifest": source_manifest,
        "artifact_sha256": artifact_sha256,
        "file_sha256": file_sha256,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest
