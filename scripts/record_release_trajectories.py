from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import torch

from fly_window.env.config import EnvironmentConfig
from fly_window.evaluation.trajectory import rollout_many_deterministic
from fly_window.neural.graph import load_connectome_graph, to_sparse_recurrent
from fly_window.neural.policy import ConnectomePolicy
from fly_window.rendering.records import build_release_record_requests


INITIALIZATION_SEED = 20260914


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_checkpoint(path: Path, expected_stage: str) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    stage = payload.get("stage")
    if stage != expected_stage:
        raise ValueError(
            f"checkpoint {path} has stage {stage!r}; expected {expected_stage!r}"
        )
    if "policy_state" not in payload:
        raise ValueError(f"checkpoint {path} has no policy_state")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Record the immutable Fly Window release trajectories: development seed 9000 "
            "at untrained/midpoint/final checkpoints and held-out seeds 10000..10002 at final."
        )
    )
    parser.add_argument("--untrained-checkpoint", type=Path, required=True)
    parser.add_argument("--midpoint-checkpoint", type=Path, required=True)
    parser.add_argument("--final-checkpoint", type=Path, required=True)
    parser.add_argument("--graph-npz", type=Path, required=True)
    parser.add_argument("--nodes-parquet", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--activity-top-k", type=int, default=256)
    args = parser.parse_args()

    if args.activity_top_k <= 0:
        parser.error("--activity-top-k must be positive")

    checkpoint_paths = {
        "untrained": args.untrained_checkpoint,
        "midpoint": args.midpoint_checkpoint,
        "final": args.final_checkpoint,
    }
    checkpoint_payloads = {
        label: _load_checkpoint(path, label)
        for label, path in checkpoint_paths.items()
    }

    graph = load_connectome_graph(args.graph_npz, args.nodes_parquet)
    recurrent = to_sparse_recurrent(graph)
    torch.manual_seed(INITIALIZATION_SEED)
    policy = ConnectomePolicy(graph, recurrent)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_records: list[dict] = []
    for request in build_release_record_requests():
        checkpoint_path = checkpoint_paths[request.checkpoint_label]
        checkpoint = checkpoint_payloads[request.checkpoint_label]
        policy.load_state_dict(checkpoint["policy_state"])
        trajectories = rollout_many_deterministic(
            policy,
            EnvironmentConfig(),
            seeds=(request.seed,),
            activity_top_k=args.activity_top_k,
        )
        if len(trajectories) != 1 or int(trajectories[0]["seed"]) != request.seed:
            raise RuntimeError("release recorder returned an unexpected trajectory set")

        output_path = args.output_dir / request.output_name
        payload = {
            "label": request.kind,
            "checkpoint_stage": request.checkpoint_label,
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": _sha256(checkpoint_path),
            "checkpoint_imitation_samples": int(checkpoint.get("imitation_samples", 0)),
            "checkpoint_imitation_minibatches": int(
                checkpoint.get("imitation_minibatches", 0)
            ),
            "initialization_seed": INITIALIZATION_SEED,
            "activity_top_k": args.activity_top_k,
            "demo_seeds": [request.seed],
            "successes": int(bool(trajectories[0]["success"])),
            "trajectories": trajectories,
        }
        output_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        manifest_records.append(
            {
                **asdict(request),
                "path": str(output_path),
                "sha256": _sha256(output_path),
                "success": bool(trajectories[0]["success"]),
                "steps": int(trajectories[0]["steps"]),
            }
        )

    manifest = {
        "initialization_seed": INITIALIZATION_SEED,
        "activity_top_k": args.activity_top_k,
        "graph_npz": str(args.graph_npz),
        "graph_npz_sha256": _sha256(args.graph_npz),
        "nodes_parquet": str(args.nodes_parquet),
        "nodes_parquet_sha256": _sha256(args.nodes_parquet),
        "checkpoint_sha256": {
            label: _sha256(path) for label, path in checkpoint_paths.items()
        },
        "records": manifest_records,
    }
    manifest_path = args.output_dir / "release-trajectories.manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
