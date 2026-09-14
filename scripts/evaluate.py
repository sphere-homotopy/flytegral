from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch

from fly_window.env.config import EnvironmentConfig
from fly_window.env.core import WindowExitEnv
from fly_window.evaluation.run import evaluate_policy
from fly_window.neural.control import build_shuffled_control
from fly_window.neural.graph import load_connectome_graph, to_sparse_recurrent
from fly_window.neural.policy import ConnectomePolicy


INITIALIZATION_SEED = 20260914
CONTROL_SEED = 314159


def _load_policy(graph, recurrent, checkpoint: Path | None):
    policy = ConnectomePolicy(graph, recurrent)
    if checkpoint is not None:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        policy.load_state_dict(payload["policy_state"])
    return policy


def _episode_runner(env_config: EnvironmentConfig):
    def run(policy: ConnectomePolicy, seed: int) -> tuple[bool, float, int]:
        env = WindowExitEnv(env_config)
        observation, _ = env.reset(seed)
        total_reward = 0.0
        steps = 0
        while True:
            observation_tensor = torch.as_tensor(
                observation[None, :], dtype=torch.float32
            )
            output = policy(observation_tensor)
            action = policy.action_from_latent(output.mean_action)[0].cpu().numpy()
            result = env.step(action)
            total_reward += result.reward
            steps += 1
            observation = result.observation
            if result.terminated or result.truncated:
                return result.success, total_reward, steps

    return run


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate trained, untrained, and shuffled Fly Window policies on held-out seeds."
    )
    parser.add_argument("--trained-checkpoint", type=Path, required=True)
    parser.add_argument("--shuffled-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--graph-npz", type=Path, default=Path("artifacts/graphs/subgraph_v1.npz")
    )
    parser.add_argument(
        "--nodes-parquet",
        type=Path,
        default=Path("artifacts/graphs/subgraph_v1_nodes.parquet"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/evaluation/evaluation.json"),
    )
    args = parser.parse_args()

    biological_graph = load_connectome_graph(args.graph_npz, args.nodes_parquet)
    biological_recurrent = to_sparse_recurrent(biological_graph)
    shuffled_graph = build_shuffled_control(biological_graph, seed=CONTROL_SEED)
    shuffled_recurrent = to_sparse_recurrent(shuffled_graph)

    trained = _load_policy(
        biological_graph, biological_recurrent, args.trained_checkpoint
    )
    torch.manual_seed(INITIALIZATION_SEED)
    untrained = _load_policy(biological_graph, biological_recurrent, None)
    shuffled = _load_policy(
        shuffled_graph, shuffled_recurrent, args.shuffled_checkpoint
    )

    runner = _episode_runner(EnvironmentConfig())
    summaries = [
        evaluate_policy(model_label="trained", policy=trained, episode_runner=runner),
        evaluate_policy(model_label="untrained", policy=untrained, episode_runner=runner),
        evaluate_policy(model_label="shuffled", policy=shuffled, episode_runner=runner),
    ]
    payload = {
        "held_out_seed_start": 10000,
        "held_out_seed_end": 10099,
        "initialization_seed": INITIALIZATION_SEED,
        "shuffled_control_seed": CONTROL_SEED,
        "models": [asdict(summary) for summary in summaries],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
