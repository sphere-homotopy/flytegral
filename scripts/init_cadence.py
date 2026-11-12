from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import torch

from fly_window.neural.graph import load_connectome_graph, to_sparse_recurrent
from fly_window.publishing.cadence import FlyCadencePolicy


def _git_sha() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True, encoding="utf-8"
    ).strip()


def _load_config(path: Path) -> tuple[tuple[int, ...], float, int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("cadence config must be a JSON object")
    raw_waits = payload.get("wait_minutes")
    if not isinstance(raw_waits, list) or not raw_waits:
        raise ValueError("cadence config wait_minutes must be a non-empty array")
    waits = tuple(int(value) for value in raw_waits)
    temperature = float(payload.get("temperature", 1.0))
    max_posts = int(payload.get("max_posts_per_24h", 30))
    return waits, temperature, max_posts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Initialize the no-LLM MaleCNS cadence-policy artifact."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/cadence_v1.json"))
    parser.add_argument("--graph-npz", type=Path, required=True)
    parser.add_argument("--nodes-parquet", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=260918)
    args = parser.parse_args()

    wait_minutes, temperature, max_posts = _load_config(args.config)
    graph = load_connectome_graph(args.graph_npz, args.nodes_parquet)
    recurrent = to_sparse_recurrent(graph)

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(args.seed)
        policy = FlyCadencePolicy(
            graph,
            recurrent,
            wait_minutes=wait_minutes,
        )

    payload = {
        "schema_version": 1,
        "cadence_state_dict": policy.state_dict(),
        "wait_minutes": wait_minutes,
        "temperature": temperature,
        "max_posts_per_24h": max_posts,
        "seed": int(args.seed),
        "git_sha": _git_sha(),
        "llm_in_runtime": False,
        "recurrent_connectome_trainable": bool(policy.recurrent.requires_grad),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "seed": args.seed,
                "wait_minutes": list(wait_minutes),
                "temperature": temperature,
                "max_posts_per_24h": max_posts,
                "git_sha": payload["git_sha"],
                "llm_in_runtime": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
