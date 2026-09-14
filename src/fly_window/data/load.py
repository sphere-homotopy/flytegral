from __future__ import annotations

from pathlib import Path

import pandas as pd

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


def _resolve_column(frame: pd.DataFrame, candidates: tuple[str, ...]) -> str:
    for name in candidates:
        if name in frame.columns:
            return name
    available = ", ".join(map(str, frame.columns))
    raise ValueError(f"Could not resolve any of {candidates}; available columns: [{available}]")


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


def load_malecns_tables(
    annotation_path: Path,
    neurotransmitter_path: Path,
    connectivity_path: Path,
) -> NormalizedTables:
    annotations = pd.read_feather(annotation_path)
    neurotransmitters = pd.read_feather(neurotransmitter_path)
    connectivity = pd.read_feather(connectivity_path)

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
    edges = edges.sort_values(["src", "dst"], kind="stable").reset_index(drop=True)

    return NormalizedTables(nodes=nodes, edges=edges)
