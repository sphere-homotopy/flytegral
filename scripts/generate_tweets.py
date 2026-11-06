from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from fly_window.neural.graph import load_connectome_graph, to_sparse_recurrent
from fly_window.text.generation import GenerationConfig, generate_batch
from fly_window.text.policy import FlyTextPolicy
from fly_window.text.provenance import write_generation_jsonl
from fly_window.text.vocabulary import build_v1_vocabulary


def _resolve_device(name: str) -> torch.device:
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
        return torch.device("cuda")
    return torch.device("cpu")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate exactly ten autonomous tweets from a Fly Tweets checkpoint."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--graph-npz",
        type=Path,
        default=Path("artifacts/graphs/subgraph_v1.npz"),
    )
    parser.add_argument(
        "--nodes-parquet",
        type=Path,
        default=Path("artifacts/graphs/subgraph_v1_nodes.parquet"),
    )
    parser.add_argument("--batch-id", required=True)
    parser.add_argument("--base-seed", type=int, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/generated/fly-tweets.jsonl"),
    )
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--max-chars", type=int, default=240)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--allow-ungated",
        action="store_true",
        help="Diagnostic only: allow generation from a checkpoint whose launch gate failed.",
    )
    args = parser.parse_args()

    device = _resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint payload must be a mapping")

    gate_report = checkpoint.get("gate_report")
    gate_passed = bool(isinstance(gate_report, dict) and gate_report.get("passed") is True)
    if not gate_passed and not args.allow_ungated:
        raise RuntimeError(
            "checkpoint has not passed the Fly Tweets launch gate; "
            "use --allow-ungated only for private diagnostics"
        )

    vocabulary = build_v1_vocabulary()
    checkpoint_vocabulary = tuple(checkpoint.get("vocabulary", ()))
    if checkpoint_vocabulary != vocabulary.tokens:
        raise ValueError("checkpoint vocabulary does not exactly match Fly Tweets v1")

    graph = load_connectome_graph(args.graph_npz, args.nodes_parquet)
    policy = FlyTextPolicy(
        graph,
        to_sparse_recurrent(graph),
        vocab_size=len(vocabulary),
    ).to(device)
    policy.load_state_dict(checkpoint["model_state_dict"])

    config = GenerationConfig(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        max_chars=args.max_chars,
    )
    checkpoint_id = args.checkpoint.parent.name
    git_sha = str(checkpoint.get("git_sha", ""))
    tweets = generate_batch(
        policy,
        vocabulary,
        base_seed=args.base_seed,
        config=config,
        checkpoint_id=checkpoint_id,
        git_sha=git_sha,
        batch_id=args.batch_id,
    )
    write_generation_jsonl(args.output, tweets)

    print(
        json.dumps(
            {
                "batch_id": args.batch_id,
                "checkpoint_id": checkpoint_id,
                "git_sha": git_sha,
                "gate_passed": gate_passed,
                "output": str(args.output),
                "tweets": [
                    {
                        "tweet_index": tweet.tweet_index,
                        "seed": tweet.seed,
                        "termination_reason": tweet.termination_reason,
                        "text": tweet.text,
                    }
                    for tweet in tweets
                ],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
