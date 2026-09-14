import math

import numpy as np
import pandas as pd
import pytest

from fly_window.neural.graph import load_connectome_graph, to_sparse_recurrent


def test_load_and_normalize_signed_connectome_graph(tmp_path):
    npz_path = tmp_path / "toy.npz"
    nodes_path = tmp_path / "nodes.parquet"

    np.savez_compressed(
        npz_path,
        body_ids=np.asarray([10, 20, 30], dtype=np.int64),
        src_idx=np.asarray([0, 1], dtype=np.int64),
        dst_idx=np.asarray([2, 2], dtype=np.int64),
        weights=np.asarray([3, 7], dtype=np.int64),
        input_mask=np.asarray([True, False, False], dtype=np.bool_),
        output_mask=np.asarray([False, False, True], dtype=np.bool_),
    )
    pd.DataFrame(
        {
            "body_id": [10, 20, 30],
            "nt": ["acetylcholine", "GABA", None],
        }
    ).to_parquet(nodes_path, index=False)

    graph = load_connectome_graph(npz_path, nodes_path)
    recurrent = to_sparse_recurrent(graph).coalesce()
    dense = recurrent.to_dense()

    assert graph.body_ids.tolist() == [10, 20, 30]
    assert graph.transmitter_sign.tolist() == pytest.approx([1.0, -1.0, 1.0])
    assert graph.input_mask.tolist() == [True, False, False]
    assert graph.output_mask.tolist() == [False, False, True]
    assert tuple(recurrent.shape) == (3, 3)
    assert recurrent.requires_grad is False

    positive = math.log1p(3)
    negative = math.log1p(7)
    normalizer = positive + negative
    assert dense[0, 2].item() == pytest.approx(positive / normalizer)
    assert dense[1, 2].item() == pytest.approx(-negative / normalizer)
    assert dense[:, 2].abs().sum().item() == pytest.approx(1.0)
    assert dense[:, :2].abs().sum().item() == pytest.approx(0.0)


def test_modulatory_and_unknown_transmitters_use_pinned_signs(tmp_path):
    npz_path = tmp_path / "toy.npz"
    nodes_path = tmp_path / "nodes.parquet"

    np.savez_compressed(
        npz_path,
        body_ids=np.asarray([1, 2, 3, 4], dtype=np.int64),
        src_idx=np.asarray([0, 1, 2], dtype=np.int64),
        dst_idx=np.asarray([3, 3, 3], dtype=np.int64),
        weights=np.asarray([1, 1, 1], dtype=np.int64),
        input_mask=np.asarray([True, False, False, False], dtype=np.bool_),
        output_mask=np.asarray([False, False, False, True], dtype=np.bool_),
    )
    pd.DataFrame(
        {
            "body_id": [1, 2, 3, 4],
            "nt": ["dopamine", "serotonin", "mystery", "octopamine"],
        }
    ).to_parquet(nodes_path, index=False)

    graph = load_connectome_graph(npz_path, nodes_path)

    assert graph.transmitter_sign.tolist() == pytest.approx([0.25, 0.25, 1.0, 0.25])
