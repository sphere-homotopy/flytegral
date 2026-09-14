from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from fly_window.env.config import EnvironmentConfig
from fly_window.evaluation.trajectory import rollout_deterministic
from fly_window.neural.control import build_shuffled_control
from fly_window.neural.graph import load_connectome_graph, to_sparse_recurrent
from fly_window.neural.policy import ConnectomePolicy


DEMO_SEEDS = (10000, 10001, 10002)
INITIALIZATION_SEED = 20260914
CONTROL_SEED = 314159


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Record deterministic Fly Window demo trajectories on the fixed three demo seeds."
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--label", required=True)
    parser.add_argument(
        "--control", choices=("biological", "shuffled"), default="biological"
    )
    parser.add_argument(
        "--graph-npz", type=Path, default=Path("artifacts/graphs/subgraph_v1.npz")
    )
    parser.add_argument(
        "--nodes-parquet",
        type=Path,
        default=Path("artifacts/graphs/subgraph_v1_nodes.parquet"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    graph = load_connectome_graph(args.graph_npz, args.nodes_parquet)
    if args.control == "shuffled":
        graph = build_shuffled_control(graph, seed=CONTROL_SEED)
    recurrent = to_sparse_recurrent(graph)

    torch.manual_seed(INITIALIZATION_SEED)
    policy = ConnectomePolicy(graph, recurrent)
    checkpoint_steps = 0
    if args.checkpoint is not None:
        payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        policy.load_state_dict(payload["policy_state"])
        checkpoint_steps = int(payload["total_environment_steps"])

    trajectories = [
        rollout_deterministic(policy, EnvironmentConfig(), seed=seed)
        for seed in DEMO_SEEDS
    ]
    result = {
        "label": args.label,
        "control": args.control,
        "checkpoint": str(args.checkpoint) if args.checkpoint is not None else None,
        "checkpoint_environment_steps": checkpoint_steps,
        "initialization_seed": INITIALIZATION_SEED,
        "control_seed": CONTROL_SEED if args.control == "shuffled" else None,
        "demo_seeds": list(DEMO_SEEDS),
        "successes": sum(int(item["success"]) for item in trajectories),
        "trajectories": trajectories,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in result.items() if k != "trajectories"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
