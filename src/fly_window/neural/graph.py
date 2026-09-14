from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch


_SIGN_BY_TRANSMITTER = {
    "gaba": -1.0,
    "glutamate": -1.0,
    "acetylcholine": 1.0,
    "dopamine": 0.25,
    "serotonin": 0.25,
    "octopamine": 0.25,
}


@dataclass(frozen=True, slots=True)
class ConnectomeGraph:
    body_ids: np.ndarray
    src_idx: np.ndarray
    dst_idx: np.ndarray
    weights: np.ndarray
    input_mask: np.ndarray
    output_mask: np.ndarray
    transmitter_sign: np.ndarray

    @property
    def node_count(self) -> int:
        return int(self.body_ids.shape[0])

    @property
    def edge_count(self) -> int:
        return int(self.src_idx.shape[0])


def _transmitter_sign(value: object) -> float:
    if value is None or pd.isna(value):
        return 1.0
    return _SIGN_BY_TRANSMITTER.get(str(value).strip().lower(), 1.0)


def load_connectome_graph(npz_path: Path, nodes_parquet: Path) -> ConnectomeGraph:
    with np.load(npz_path, allow_pickle=False) as data:
        body_ids = np.asarray(data["body_ids"], dtype=np.int64)
        src_idx = np.asarray(data["src_idx"], dtype=np.int64)
        dst_idx = np.asarray(data["dst_idx"], dtype=np.int64)
        weights = np.asarray(data["weights"], dtype=np.int64)
        input_mask = np.asarray(data["input_mask"], dtype=np.bool_)
        output_mask = np.asarray(data["output_mask"], dtype=np.bool_)

    node_count = body_ids.shape[0]
    if input_mask.shape != (node_count,) or output_mask.shape != (node_count,):
        raise ValueError("input/output masks must match body_ids length")
    if not (src_idx.shape == dst_idx.shape == weights.shape):
        raise ValueError("src_idx, dst_idx, and weights must have identical shapes")
    if np.any(weights <= 0):
        raise ValueError("connectome edge weights must be positive")
    if src_idx.size and (
        int(src_idx.min()) < 0
        or int(dst_idx.min()) < 0
        or int(src_idx.max()) >= node_count
        or int(dst_idx.max()) >= node_count
    ):
        raise ValueError("edge indices are outside the node array")

    nodes = pd.read_parquet(nodes_parquet)
    if "body_id" not in nodes.columns or "nt" not in nodes.columns:
        raise ValueError("nodes parquet must contain body_id and nt columns")
    if nodes["body_id"].duplicated().any():
        raise ValueError("nodes parquet contains duplicate body_id values")

    neurotransmitters = nodes.set_index("body_id")["nt"].reindex(body_ids)
    if neurotransmitters.index.shape[0] != node_count or neurotransmitters.isna().all():
        # Missing transmitter labels are allowed, but a completely failed body-id join is not.
        known_ids = set(nodes["body_id"].astype("int64").tolist())
        if not any(int(body_id) in known_ids for body_id in body_ids):
            raise ValueError("nodes parquet body ids do not align with graph body_ids")

    transmitter_sign = np.asarray(
        [_transmitter_sign(value) for value in neurotransmitters.tolist()],
        dtype=np.float32,
    )

    return ConnectomeGraph(
        body_ids=body_ids,
        src_idx=src_idx,
        dst_idx=dst_idx,
        weights=weights,
        input_mask=input_mask,
        output_mask=output_mask,
        transmitter_sign=transmitter_sign,
    )


def to_sparse_recurrent(graph: ConnectomeGraph) -> torch.Tensor:
    if graph.edge_count == 0:
        indices = torch.empty((2, 0), dtype=torch.int64)
        values = torch.empty((0,), dtype=torch.float32)
        return torch.sparse_coo_tensor(
            indices,
            values,
            size=(graph.node_count, graph.node_count),
            dtype=torch.float32,
        ).coalesce()

    magnitude = np.log1p(graph.weights.astype(np.float64))
    incoming_abs_sum = np.zeros(graph.node_count, dtype=np.float64)
    np.add.at(incoming_abs_sum, graph.dst_idx, magnitude)
    denominators = incoming_abs_sum[graph.dst_idx]
    if np.any(denominators <= 0):
        raise ValueError("every retained edge must have a positive destination normalizer")

    values_np = (
        magnitude / denominators * graph.transmitter_sign[graph.src_idx].astype(np.float64)
    ).astype(np.float32)
    indices = torch.from_numpy(
        np.stack([graph.src_idx, graph.dst_idx], axis=0).astype(np.int64, copy=False)
    )
    values = torch.from_numpy(values_np)
    recurrent = torch.sparse_coo_tensor(
        indices,
        values,
        size=(graph.node_count, graph.node_count),
        dtype=torch.float32,
    ).coalesce()
    return recurrent.detach()
