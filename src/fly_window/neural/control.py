from __future__ import annotations

import numpy as np

from fly_window.neural.graph import ConnectomeGraph


def shuffle_destinations(graph: ConnectomeGraph, seed: int) -> ConnectomeGraph:
    """Return the deterministic destination-permutation control.

    Source identities, raw synapse counts, neuron metadata, and input/output masks
    remain fixed. Only the destination endpoint vector is permuted.
    """

    rng = np.random.default_rng(seed)
    shuffled_dst = rng.permutation(graph.dst_idx).astype(np.int64, copy=False)
    return ConnectomeGraph(
        body_ids=graph.body_ids.copy(),
        src_idx=graph.src_idx.copy(),
        dst_idx=shuffled_dst,
        weights=graph.weights.copy(),
        input_mask=graph.input_mask.copy(),
        output_mask=graph.output_mask.copy(),
        transmitter_sign=graph.transmitter_sign.copy(),
    )


def aggregate_duplicate_edges(graph: ConnectomeGraph) -> ConnectomeGraph:
    """Sum raw synapse counts for duplicate ``(src, dst)`` edges.

    Aggregation happens before ``log1p`` recurrent-weight normalization so that a
    shuffled control has the same raw synapse-count semantics as the biological
    graph rather than summing already transformed edge magnitudes.
    """

    if graph.edge_count == 0:
        return ConnectomeGraph(
            body_ids=graph.body_ids.copy(),
            src_idx=graph.src_idx.copy(),
            dst_idx=graph.dst_idx.copy(),
            weights=graph.weights.copy(),
            input_mask=graph.input_mask.copy(),
            output_mask=graph.output_mask.copy(),
            transmitter_sign=graph.transmitter_sign.copy(),
        )

    order = np.lexsort((graph.dst_idx, graph.src_idx))
    src = graph.src_idx[order]
    dst = graph.dst_idx[order]
    weights = graph.weights[order]

    starts = np.empty(graph.edge_count, dtype=np.bool_)
    starts[0] = True
    starts[1:] = (src[1:] != src[:-1]) | (dst[1:] != dst[:-1])
    start_idx = np.flatnonzero(starts)

    aggregated_src = src[start_idx].astype(np.int64, copy=False)
    aggregated_dst = dst[start_idx].astype(np.int64, copy=False)
    aggregated_weights = np.add.reduceat(weights, start_idx).astype(np.int64, copy=False)

    return ConnectomeGraph(
        body_ids=graph.body_ids.copy(),
        src_idx=aggregated_src,
        dst_idx=aggregated_dst,
        weights=aggregated_weights,
        input_mask=graph.input_mask.copy(),
        output_mask=graph.output_mask.copy(),
        transmitter_sign=graph.transmitter_sign.copy(),
    )


def build_shuffled_control(graph: ConnectomeGraph, seed: int) -> ConnectomeGraph:
    """Build the graph used by the shuffled-connectome control condition."""

    return aggregate_duplicate_edges(shuffle_destinations(graph, seed=seed))
