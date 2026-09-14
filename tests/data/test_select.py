import pandas as pd

from fly_window.data.schema import NormalizedTables
from fly_window.data.select import SubgraphConfig, select_task_subgraph


def _tables(nodes, edges):
    return NormalizedTables(
        nodes=pd.DataFrame(nodes, columns=["body_id", "type", "superclass", "cell_class", "side", "nt"]),
        edges=pd.DataFrame(edges, columns=["src", "dst", "weight"]).astype(
            {"src": "int64", "dst": "int64", "weight": "int64"}
        ),
    )


def test_select_task_subgraph_keeps_only_visual_to_descending_paths_plus_seeds():
    graph = _tables(
        [
            (1, "LC01", "visual_projection_neuron", None, "L", "acetylcholine"),
            (2, "IN01", "interneuron", None, None, "gaba"),
            (3, "IN02", "interneuron", None, None, "gaba"),
            (4, "DNa01", "descending_neuron", None, "R", "glutamate"),
            (5, "LC02", "visual_projection_neuron", None, "R", "acetylcholine"),
            (6, "DEAD", "interneuron", None, None, "gaba"),
            (7, "OTHER1", "interneuron", None, None, "gaba"),
            (8, "OTHER2", "interneuron", None, None, "gaba"),
            (9, "WEAK", "interneuron", None, None, "gaba"),
        ],
        [
            (1, 2, 5),
            (2, 3, 6),
            (3, 4, 7),
            (5, 6, 9),
            (7, 8, 100),
            (1, 9, 2),
            (9, 4, 20),
        ],
    )

    selected = select_task_subgraph(graph, SubgraphConfig(max_hops=4, max_nodes=100))

    assert selected.nodes["body_id"].tolist() == [1, 2, 3, 4, 5]
    assert selected.edges.to_dict("records") == [
        {"src": 1, "dst": 2, "weight": 5},
        {"src": 2, "dst": 3, "weight": 6},
        {"src": 3, "dst": 4, "weight": 7},
    ]


def test_select_task_subgraph_is_deterministic_under_row_shuffling():
    graph = _tables(
        [
            (10, "LC10", "visual_projection_neuron", None, None, None),
            (20, "MID", "interneuron", None, None, None),
            (30, "DNa10", "descending_neuron", None, None, None),
        ],
        [(10, 20, 4), (20, 30, 5)],
    )
    shuffled = NormalizedTables(
        nodes=graph.nodes.sample(frac=1, random_state=11).reset_index(drop=True),
        edges=graph.edges.sample(frac=1, random_state=12).reset_index(drop=True),
    )
    config = SubgraphConfig(max_hops=3, max_nodes=10)

    first = select_task_subgraph(graph, config)
    second = select_task_subgraph(shuffled, config)

    assert first.nodes.to_dict("records") == second.nodes.to_dict("records")
    assert first.edges.to_dict("records") == second.edges.to_dict("records")


def test_select_task_subgraph_caps_intermediates_without_dropping_seeds():
    graph = _tables(
        [
            (1, "LC01", "visual_projection_neuron", None, None, None),
            (2, "A", "interneuron", None, None, None),
            (3, "B", "interneuron", None, None, None),
            (4, "C", "interneuron", None, None, None),
            (5, "DNa01", "descending_neuron", None, None, None),
        ],
        [
            (1, 2, 10), (2, 5, 10),
            (1, 3, 8), (3, 5, 8),
            (1, 4, 6), (4, 5, 6),
        ],
    )

    selected = select_task_subgraph(graph, SubgraphConfig(max_hops=2, max_nodes=4))

    assert 1 in selected.nodes["body_id"].tolist()
    assert 5 in selected.nodes["body_id"].tolist()
    assert len(selected.nodes) == 4
    assert selected.nodes["body_id"].tolist() == [1, 2, 3, 5]
