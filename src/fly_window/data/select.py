from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from .schema import NormalizedTables


@dataclass(frozen=True)
class SubgraphConfig:
    min_edge_weight: int = 3
    max_hops: int = 4
    max_nodes: int = 6000
    output_superclass: str = "descending_neuron"
    visual_type_prefixes: tuple[str, ...] = (
        "LC",
        "LPLC",
        "LT",
        "MC",
        "Mi",
        "Tm",
        "T4",
        "T5",
    )


def _forward_depths(matrix: csr_matrix, seeds: set[int], max_hops: int) -> dict[int, int]:
    depths = {seed: 0 for seed in seeds}
    frontier = set(seeds)
    for depth in range(1, max_hops + 1):
        next_frontier: set[int] = set()
        for node in frontier:
            start, end = matrix.indptr[node], matrix.indptr[node + 1]
            next_frontier.update(int(value) for value in matrix.indices[start:end])
        next_frontier.difference_update(depths)
        if not next_frontier:
            break
        for node in next_frontier:
            depths[node] = depth
        frontier = next_frontier
    return depths


def _reverse_depths(matrix: csr_matrix, seeds: set[int], max_hops: int) -> dict[int, int]:
    csc = matrix.tocsc()
    depths = {seed: 0 for seed in seeds}
    frontier = set(seeds)
    for depth in range(1, max_hops + 1):
        next_frontier: set[int] = set()
        for node in frontier:
            start, end = csc.indptr[node], csc.indptr[node + 1]
            next_frontier.update(int(value) for value in csc.indices[start:end])
        next_frontier.difference_update(depths)
        if not next_frontier:
            break
        for node in next_frontier:
            depths[node] = depth
        frontier = next_frontier
    return depths


def select_task_subgraph(
    tables: NormalizedTables,
    config: SubgraphConfig,
) -> NormalizedTables:
    nodes = tables.nodes.sort_values("body_id", kind="stable").reset_index(drop=True).copy()
    if nodes["body_id"].duplicated().any():
        raise ValueError("body_id must be unique before subgraph selection")

    body_ids = nodes["body_id"].astype("int64").to_numpy()
    index_by_body = {int(body_id): idx for idx, body_id in enumerate(body_ids)}

    type_strings = nodes["type"].fillna("").astype(str)
    visual_mask = type_strings.map(lambda value: value.startswith(config.visual_type_prefixes))
    output_mask = nodes["superclass"].fillna("").astype(str).eq(config.output_superclass)

    visual_bodies = set(nodes.loc[visual_mask, "body_id"].astype(int))
    output_bodies = set(nodes.loc[output_mask, "body_id"].astype(int))
    if not visual_bodies or not output_bodies:
        raise ValueError(
            "Subgraph selection requires visual seeds and descending outputs; "
            f"visual={len(visual_bodies)}, outputs={len(output_bodies)}, "
            f"prefixes={config.visual_type_prefixes}"
        )

    edges = tables.edges.loc[tables.edges["weight"] >= config.min_edge_weight].copy()
    edges = edges.loc[edges["src"].isin(index_by_body) & edges["dst"].isin(index_by_body)]
    edges = edges.astype({"src": "int64", "dst": "int64", "weight": "int64"})
    edges = edges.sort_values(["src", "dst"], kind="stable").reset_index(drop=True)

    rows = np.fromiter((index_by_body[int(src)] for src in edges["src"]), dtype=np.int64, count=len(edges))
    cols = np.fromiter((index_by_body[int(dst)] for dst in edges["dst"]), dtype=np.int64, count=len(edges))
    adjacency = csr_matrix((np.ones(len(edges), dtype=np.uint8), (rows, cols)), shape=(len(nodes), len(nodes)))

    visual_indices = {index_by_body[body] for body in visual_bodies}
    output_indices = {index_by_body[body] for body in output_bodies}
    forward = _forward_depths(adjacency, visual_indices, config.max_hops)
    reverse = _reverse_depths(adjacency, output_indices, config.max_hops)

    reachable_indices = set(forward).intersection(reverse)
    required_indices = set(visual_indices).union(output_indices)
    candidate_indices = reachable_indices.union(required_indices)

    incident_weight = {idx: 0 for idx in candidate_indices}
    for row in edges.itertuples(index=False):
        src_idx = index_by_body[int(row.src)]
        dst_idx = index_by_body[int(row.dst)]
        if src_idx in incident_weight:
            incident_weight[src_idx] += int(row.weight)
        if dst_idx in incident_weight:
            incident_weight[dst_idx] += int(row.weight)

    intermediate_indices = candidate_indices.difference(required_indices)
    slots = max(0, config.max_nodes - len(required_indices))
    if len(intermediate_indices) > slots:
        ranked = sorted(
            intermediate_indices,
            key=lambda idx: (
                forward[idx] + reverse[idx],
                -incident_weight.get(idx, 0),
                int(body_ids[idx]),
            ),
        )
        intermediate_indices = set(ranked[:slots])

    kept_indices = required_indices.union(intermediate_indices)
    kept_bodies = {int(body_ids[idx]) for idx in kept_indices}

    selected_nodes = nodes.loc[nodes["body_id"].isin(kept_bodies)].copy()
    selected_nodes = selected_nodes.sort_values("body_id", kind="stable").reset_index(drop=True)

    selected_edges = edges.loc[edges["src"].isin(kept_bodies) & edges["dst"].isin(kept_bodies)].copy()
    selected_edges = selected_edges.sort_values(["src", "dst"], kind="stable").reset_index(drop=True)

    return NormalizedTables(nodes=selected_nodes, edges=selected_edges)
