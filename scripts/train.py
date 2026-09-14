from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import torch

from fly_window.env.config import EnvironmentConfig
from fly_window.env.core import WindowExitEnv
from fly_window.neural.control import build_shuffled_control
from fly_window.neural.graph import load_connectome_graph, to_sparse_recurrent
from fly_window.training.run import (
    DEVELOPMENT_SEEDS,
    TrainingConfig,
    WindowVectorEnv,
    run_training,
)


CONTROL_SEED = 314159


def _git_sha() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True, encoding="utf-8"
    ).strip()


def _development_evaluator(env_config: EnvironmentConfig):
    def evaluate(policy, seeds: tuple[int, ...]) -> dict:
        if seeds != DEVELOPMENT_SEEDS:
            raise ValueError("training evaluation may use only DEVELOPMENT_SEEDS")
        was_training = policy.training
        policy.eval()
        successes = 0
        steps: list[int] = []
        with torch.no_grad():
            for seed in seeds:
                env = WindowExitEnv(env_config)
                observation, _ = env.reset(seed)
                episode_steps = 0
                while True:
                    observation_tensor = torch.as_tensor(
                        observation[None, :], dtype=torch.float32
                    )
                    output = policy(observation_tensor)
                    action = policy.action_from_latent(output.mean_action)[0].cpu().numpy()
                    result = env.step(action)
                    episode_steps += 1
                    observation = result.observation
                    if result.terminated or result.truncated:
                        successes += int(result.success)
                        steps.append(episode_steps)
                        break
        policy.train(was_training)
        return {
            "episode_count": len(seeds),
            "successes": successes,
            "success_rate": successes / len(seeds),
            "mean_steps": float(np.mean(steps)),
            "seed_start": seeds[0],
            "seed_end": seeds[-1],
        }

    return evaluate


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the Fly Window connectome-constrained PPO policy."
    )
    parser.add_argument(
        "--config", type=Path, default=Path("configs/training_v1.json")
    )
    parser.add_argument(
        "--graph-npz", type=Path, default=Path("artifacts/graphs/subgraph_v1.npz")
    )
    parser.add_argument(
        "--nodes-parquet",
        type=Path,
        default=Path("artifacts/graphs/subgraph_v1_nodes.parquet"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("artifacts/graphs/subgraph_v1_manifest.json"),
    )
    parser.add_argument(
        "--control", choices=("biological", "shuffled"), default="biological"
    )
    parser.add_argument(
        "--run-root", type=Path, default=Path("artifacts/runs")
    )
    args = parser.parse_args()

    config = TrainingConfig.from_json(args.config)
    source_graph = load_connectome_graph(args.graph_npz, args.nodes_parquet)
    control_seed = None
    graph = source_graph
    if args.control == "shuffled":
        control_seed = CONTROL_SEED
        graph = build_shuffled_control(source_graph, seed=CONTROL_SEED)
    recurrent = to_sparse_recurrent(graph)

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    graph_manifest_hash = str(manifest["artifact_sha256"])
    env_config = EnvironmentConfig()
    vector_env = WindowVectorEnv(config.parallel_envs, env_config)

    result = run_training(
        config=config,
        graph=graph,
        recurrent=recurrent,
        vector_env=vector_env,
        run_root=args.run_root,
        graph_manifest_hash=graph_manifest_hash,
        git_sha=_git_sha(),
        control_kind=args.control,
        control_seed=control_seed,
        development_evaluator=_development_evaluator(env_config),
    )
    print(
        json.dumps(
            {
                "run_dir": str(result.run_dir),
                "total_environment_steps": result.total_environment_steps,
                "control_kind": args.control,
                "control_seed": control_seed,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
