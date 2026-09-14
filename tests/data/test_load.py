from pathlib import Path

import pandas as pd
import pytest

from fly_window.data.load import load_malecns_tables


def _write(frame: pd.DataFrame, path):
    frame.reset_index(drop=True).to_feather(path)
    return path


def test_load_malecns_tables_normalizes_labels_edges_and_duplicates(tmp_path):
    annotations = pd.DataFrame(
        {
            "bodyId": pd.Series([1, 2, 3], dtype="int32"),
            "type": ["LC01", None, "DNa01"],
            "superclass": ["visual_projection_neuron", "interneuron", "descending_neuron"],
            "cell_class": ["optic", None, "descending"],
            "side": ["L", "R", None],
        }
    )
    neurotransmitters = pd.DataFrame(
        {
            "body": pd.Series([1, 2, 3], dtype="int64"),
            "predicted_nt": ["acetylcholine", "gaba", "glutamate"],
        }
    )
    connectivity = pd.DataFrame(
        {
            "bodyPre": pd.Series([1, 1, 2, 3], dtype="int32"),
            "bodyPost": pd.Series([2, 2, 3, 3], dtype="int64"),
            "weight": pd.Series([2, 5, 4, 9], dtype="int16"),
        }
    )

    tables = load_malecns_tables(
        _write(annotations, tmp_path / "annotations.feather"),
        _write(neurotransmitters, tmp_path / "nt.feather"),
        _write(connectivity, tmp_path / "edges.feather"),
    )

    assert list(tables.nodes.columns) == [
        "body_id",
        "type",
        "superclass",
        "cell_class",
        "side",
        "nt",
    ]
    assert tables.nodes["body_id"].dtype == "int64"
    assert tables.nodes["body_id"].tolist() == [1, 2, 3]
    assert tables.nodes["nt"].tolist() == ["acetylcholine", "gaba", "glutamate"]

    assert list(tables.edges.columns) == ["src", "dst", "weight"]
    assert tables.edges.dtypes.astype(str).tolist() == ["int64", "int64", "int64"]
    assert tables.edges.to_dict("records") == [
        {"src": 1, "dst": 2, "weight": 7},
        {"src": 2, "dst": 3, "weight": 4},
    ]


def test_generic_loader_applies_minimum_weight_after_duplicate_aggregation(tmp_path):
    annotations = pd.DataFrame({"bodyId": [1, 2], "type": ["LC01", "DNa01"]})
    neurotransmitters = pd.DataFrame({"bodyId": [1, 2], "predicted_nt": ["gaba", "gaba"]})
    connectivity = pd.DataFrame(
        {
            "body_pre": [1, 1],
            "body_post": [2, 2],
            "weight": [2, 2],
        }
    )

    tables = load_malecns_tables(
        _write(annotations, tmp_path / "annotations.feather"),
        _write(neurotransmitters, tmp_path / "nt.feather"),
        _write(connectivity, tmp_path / "edges.feather"),
        min_edge_weight=3,
    )

    assert tables.edges.to_dict("records") == [{"src": 1, "dst": 2, "weight": 4}]


def test_aggregated_connectivity_streams_and_filters_before_pandas_materialization(tmp_path, monkeypatch):
    annotations = pd.DataFrame(
        {
            "bodyId": [1, 2, 3],
            "type": ["LC01", "Mi01", "DNa01"],
            "superclass": ["visual_projection_neuron", "interneuron", "descending_neuron"],
        }
    )
    neurotransmitters = pd.DataFrame(
        {
            "bodyId": [1, 2, 3],
            "predicted_nt": ["acetylcholine", "gaba", "glutamate"],
        }
    )
    connectivity = pd.DataFrame(
        {
            "body_pre": [1, 1, 4, 2, 3],
            "body_post": [2, 3, 2, 3, 3],
            "weight": [2, 5, 100, 4, 9],
        }
    )
    annotation_path = _write(annotations, tmp_path / "annotations.feather")
    nt_path = _write(neurotransmitters, tmp_path / "nt.feather")
    connectivity_path = _write(connectivity, tmp_path / "edges.feather")

    original_read_feather = pd.read_feather

    def guarded_read_feather(path, *args, **kwargs):
        if Path(path) == Path(connectivity_path):
            raise AssertionError("aggregated connectivity must be streamed before pandas materialization")
        return original_read_feather(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_feather", guarded_read_feather)

    tables = load_malecns_tables(
        annotation_path,
        nt_path,
        connectivity_path,
        min_edge_weight=3,
        connectivity_rows_are_aggregated=True,
    )

    assert tables.edges.to_dict("records") == [
        {"src": 1, "dst": 3, "weight": 5},
        {"src": 2, "dst": 3, "weight": 4},
    ]


def test_load_malecns_tables_uses_probability_argmax_with_lexical_ties(tmp_path):
    annotations = pd.DataFrame(
        {
            "body": [10, 20],
            "type": ["Tm1", "DNa02"],
            "superclass": ["optic_lobe", "descending_neuron"],
        }
    )
    neurotransmitters = pd.DataFrame(
        {
            "bodyId": [10, 20],
            "acetylcholine": [0.4, 0.1],
            "gaba": [0.4, 0.1],
            "glutamate": [0.1, 0.6],
            "dopamine": [0.0, 0.0],
            "serotonin": [0.0, 0.0],
            "octopamine": [0.1, 0.2],
        }
    )
    connectivity = pd.DataFrame({"body_pre": [10], "body_post": [20], "weight": [3]})

    tables = load_malecns_tables(
        _write(annotations, tmp_path / "annotations.feather"),
        _write(neurotransmitters, tmp_path / "nt.feather"),
        _write(connectivity, tmp_path / "edges.feather"),
    )

    by_id = tables.nodes.set_index("body_id")
    assert by_id.loc[10, "nt"] == "acetylcholine"
    assert by_id.loc[20, "nt"] == "glutamate"
    assert pd.isna(by_id.loc[10, "cell_class"])
    assert pd.isna(by_id.loc[20, "side"])


@pytest.mark.parametrize("bad_weight", [0, -1])
def test_load_malecns_tables_rejects_nonpositive_weights(tmp_path, bad_weight):
    annotations = pd.DataFrame({"bodyId": [1, 2], "type": ["LC01", "DNa01"]})
    neurotransmitters = pd.DataFrame({"bodyId": [1, 2], "predicted_nt": ["gaba", "gaba"]})
    connectivity = pd.DataFrame({"body_pre": [1], "body_post": [2], "weight": [bad_weight]})

    with pytest.raises(ValueError, match="weight"):
        load_malecns_tables(
            _write(annotations, tmp_path / "annotations.feather"),
            _write(neurotransmitters, tmp_path / "nt.feather"),
            _write(connectivity, tmp_path / "edges.feather"),
        )
