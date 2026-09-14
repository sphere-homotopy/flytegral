from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc

from .schema import NormalizedTables


_NODE_ID_CANDIDATES = ("bodyId", "body", "body_id")
_SRC_CANDIDATES = ("body_pre", "bodyPre", "src", "source")
_DST_CANDIDATES = ("body_post", "bodyPost", "dst", "target")
_WEIGHT_CANDIDATES = ("weight", "count", "syn_count", "synapse_count")
_NT_LABEL_CANDIDATES = ("predicted_nt", "neurotransmitter", "nt")
_NT_PROBABILITY_LABELS = (
    "acetylcholine",
    "dopamine",
    "gaba",
    "glutamate",
    "octopamine",
    "serotonin",
)


def _resolve_name(names: tuple[str, ...] | list[str], candidates: tuple[str, ...]) -> str:
    for name in candidates:
        if name in names:
            return name
    available = ", ".join(map(str, names))
    raise ValueError(f"Could not resolve any of {candidates}; available columns: [{available}]")


def _resolve_column(frame: pd.DataFrame, candidates: tuple[str, ...]) -> str:
    return _resolve_name(list(map(str, frame.columns)), candidates)


def _optional_column(frame: pd.DataFrame, name: str) -> pd.Series:
    if name in frame.columns:
        return frame[name]
    return pd.Series(pd.NA, index=frame.index, dtype="object")


def _normalize_neurotransmitters(frame: pd.DataFrame) -> pd.DataFrame:
    body_col = _resolve_column(frame, _NODE_ID_CANDIDATES)

    label_col = next((name for name in _NT_LABEL_CANDIDATES if name in frame.columns), None)
    if label_col is not None:
        labels = frame[label_col].astype("object")
    else:
        available_labels = sorted(label for label in _NT_PROBABILITY_LABELS if label in frame.columns)
        if not available_labels:
            available = ", ".join(map(str, frame.columns))
            raise ValueError(
                "Could not resolve neurotransmitter label/probability columns; "
                f"available columns: [{available}]"
            )
        probabilities = frame[available_labels].apply(pd.to_numeric, errors="raise")
        labels = probabilities.idxmax(axis=1).astype("object")

    result = pd.DataFrame(
        {
            "body_id": pd.to_numeric(frame[body_col], errors="raise").astype("int64"),
            "nt": labels,
        }
    )
    return result.drop_duplicates(subset=["body_id"], keep="first")


def _empty_edges() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "src": pd.Series(dtype="int64"),
            "dst": pd.Series(dtype="int64"),
            "weight": pd.Series(dtype="int64"),
        }
    )


def _normalize_generic_connectivity(
    connectivity: pd.DataFrame,
    *,
    min_edge_weight: int | None,
) -> pd.DataFrame:
    src_col = _resolve_column(connectivity, _SRC_CANDIDATES)
    dst_col = _resolve_column(connectivity, _DST_CANDIDATES)
    weight_col = _resolve_column(connectivity, _WEIGHT_CANDIDATES)
    edges = pd.DataFrame(
        {
            "src": pd.to_numeric(connectivity[src_col], errors="raise").astype("int64"),
            "dst": pd.to_numeric(connectivity[dst_col], errors="raise").astype("int64"),
            "weight": pd.to_numeric(connectivity[weight_col], errors="raise").astype("int64"),
        }
    )

    if (edges["weight"] <= 0).any():
        bad = edges.loc[edges["weight"] <= 0, "weight"].tolist()
        raise ValueError(f"Connectivity weight must be positive; found {bad[:5]}")

    edges = edges.loc[edges["src"] != edges["dst"]]
    edges = (
        edges.groupby(["src", "dst"], as_index=False, sort=True)["weight"]
        .sum()
        .astype({"src": "int64", "dst": "int64", "weight": "int64"})
    )
    if min_edge_weight is not None:
        edges = edges.loc[edges["weight"] >= min_edge_weight]
    return edges.sort_values(["src", "dst"], kind="stable").reset_index(drop=True)


def _load_aggregated_connectivity(
    connectivity_path: Path,
    *,
    annotated_body_ids: np.ndarray,
    min_edge_weight: int | None,
) -> pd.DataFrame:
    """Stream an already aggregated weighted edge table without materializing it whole.

    MaleCNS's official ``connectome-weights`` file is a segment-to-segment connection
    strength graph, not a synapse-pair table. In this mode each row is therefore a
    complete directed edge weight and may be thresholded independently before pandas
    materialization. Edges whose endpoints lack neuron annotations are also discarded
    here because the downstream task selector cannot use them.
    """

    threshold = 1 if min_edge_weight is None else min_edge_weight
    valid_ids = np.asarray(annotated_body_ids, dtype=np.int64)
    chunks: list[pd.DataFrame] = []

    with pa.memory_map(str(connectivity_path), "r") as source:
        reader = ipc.RecordBatchFileReader(source)
        names = reader.schema.names
        src_name = _resolve_name(names, _SRC_CANDIDATES)
        dst_name = _resolve_name(names, _DST_CANDIDATES)
        weight_name = _resolve_name(names, _WEIGHT_CANDIDATES)
        src_index = reader.schema.get_field_index(src_name)
        dst_index = reader.schema.get_field_index(dst_name)
        weight_index = reader.schema.get_field_index(weight_name)

        for batch_index in range(reader.num_record_batches):
            batch = reader.get_batch(batch_index)
            src = batch.column(src_index).to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
            dst = batch.column(dst_index).to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
            weight = batch.column(weight_index).to_numpy(zero_copy_only=False).astype(np.int64, copy=False)

            if np.any(weight <= 0):
                bad = weight[weight <= 0][:5].tolist()
                raise ValueError(f"Connectivity weight must be positive; found {bad}")

            candidate_mask = (weight >= threshold) & (src != dst)
            if not np.any(candidate_mask):
                continue

            candidate_src = src[candidate_mask]
            candidate_dst = dst[candidate_mask]
            candidate_weight = weight[candidate_mask]
            annotated_mask = np.isin(candidate_src, valid_ids) & np.isin(candidate_dst, valid_ids)
            if not np.any(annotated_mask):
                continue

            chunks.append(
                pd.DataFrame(
                    {
                        "src": candidate_src[annotated_mask],
                        "dst": candidate_dst[annotated_mask],
                        "weight": candidate_weight[annotated_mask],
                    },
                    copy=False,
                )
            )

    if not chunks:
        return _empty_edges()

    edges = pd.concat(chunks, ignore_index=True, copy=False)
    return edges.astype({"src": "int64", "dst": "int64", "weight": "int64"}, copy=False)


def load_malecns_tables(
    annotation_path: Path,
    neurotransmitter_path: Path,
    connectivity_path: Path,
    *,
    min_edge_weight: int | None = None,
    connectivity_rows_are_aggregated: bool = False,
) -> NormalizedTables:
    if min_edge_weight is not None and min_edge_weight <= 0:
        raise ValueError("min_edge_weight must be positive when provided")

    annotations = pd.read_feather(annotation_path)
    neurotransmitters = pd.read_feather(neurotransmitter_path)

    body_col = _resolve_column(annotations, _NODE_ID_CANDIDATES)
    nodes = pd.DataFrame(
        {
            "body_id": pd.to_numeric(annotations[body_col], errors="raise").astype("int64"),
            "type": _optional_column(annotations, "type").astype("object"),
            "superclass": _optional_column(annotations, "superclass").astype("object"),
            "cell_class": _optional_column(annotations, "cell_class").astype("object"),
            "side": _optional_column(annotations, "side").astype("object"),
        }
    )

    nt = _normalize_neurotransmitters(neurotransmitters)
    nodes = nodes.merge(nt, how="left", on="body_id", sort=False)
    nodes = nodes[["body_id", "type", "superclass", "cell_class", "side", "nt"]]
    nodes = nodes.sort_values("body_id", kind="stable").reset_index(drop=True)

    if connectivity_rows_are_aggregated:
        edges = _load_aggregated_connectivity(
            Path(connectivity_path),
            annotated_body_ids=nodes["body_id"].to_numpy(dtype=np.int64, copy=False),
            min_edge_weight=min_edge_weight,
        )
    else:
        connectivity = pd.read_feather(connectivity_path)
        edges = _normalize_generic_connectivity(
            connectivity,
            min_edge_weight=min_edge_weight,
        )

    return NormalizedTables(nodes=nodes, edges=edges)
