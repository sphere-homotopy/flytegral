import numpy as np

from fly_window.neural.control import aggregate_duplicate_edges, shuffle_destinations
from fly_window.neural.graph import ConnectomeGraph


def _toy_graph() -> ConnectomeGraph:
    return ConnectomeGraph(
        body_ids=np.asarray([10, 20, 30, 40], dtype=np.int64),
        src_idx=np.asarray([0, 0, 1, 2, 2], dtype=np.int64),
        dst_idx=np.asarray([1, 2, 2, 3, 1], dtype=np.int64),
        weights=np.asarray([1, 2, 3, 4, 5], dtype=np.int64),
        input_mask=np.asarray([True, False, False, False], dtype=np.bool_),
        output_mask=np.asarray([False, False, False, True], dtype=np.bool_),
        transmitter_sign=np.asarray([1.0, -1.0, 1.0, 0.25], dtype=np.float32),
    )


def test_shuffle_destinations_is_deterministic_and_preserves_edge_statistics():
    graph = _toy_graph()

    shuffled_a = shuffle_destinations(graph, seed=314159)
    shuffled_b = shuffle_destinations(graph, seed=314159)
    shuffled_c = shuffle_destinations(graph, seed=271828)

    assert np.array_equal(shuffled_a.dst_idx, shuffled_b.dst_idx)
    assert not np.array_equal(shuffled_a.dst_idx, shuffled_c.dst_idx)
    assert np.array_equal(shuffled_a.src_idx, graph.src_idx)
    assert np.array_equal(shuffled_a.weights, graph.weights)
    assert sorted(shuffled_a.dst_idx.tolist()) == sorted(graph.dst_idx.tolist())
    assert np.any(shuffled_a.dst_idx != graph.dst_idx)

    assert np.array_equal(shuffled_a.body_ids, graph.body_ids)
    assert np.array_equal(shuffled_a.input_mask, graph.input_mask)
    assert np.array_equal(shuffled_a.output_mask, graph.output_mask)
    assert np.array_equal(shuffled_a.transmitter_sign, graph.transmitter_sign)


def test_duplicate_edges_are_aggregated_by_raw_synapse_count_before_normalization():
    graph = ConnectomeGraph(
        body_ids=np.asarray([10, 20, 30], dtype=np.int64),
        src_idx=np.asarray([0, 0, 1], dtype=np.int64),
        dst_idx=np.asarray([2, 2, 2], dtype=np.int64),
        weights=np.asarray([2, 3, 5], dtype=np.int64),
        input_mask=np.asarray([True, False, False], dtype=np.bool_),
        output_mask=np.asarray([False, False, True], dtype=np.bool_),
        transmitter_sign=np.asarray([1.0, -1.0, 1.0], dtype=np.float32),
    )

    aggregated = aggregate_duplicate_edges(graph)

    assert aggregated.src_idx.tolist() == [0, 1]
    assert aggregated.dst_idx.tolist() == [2, 2]
    assert aggregated.weights.tolist() == [5, 5]
    assert int(aggregated.weights.sum()) == int(graph.weights.sum())
    assert np.array_equal(aggregated.body_ids, graph.body_ids)
    assert np.array_equal(aggregated.input_mask, graph.input_mask)
    assert np.array_equal(aggregated.output_mask, graph.output_mask)
    assert np.array_equal(aggregated.transmitter_sign, graph.transmitter_sign)
