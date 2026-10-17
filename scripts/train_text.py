from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from fly_window.neural.graph import load_connectome_graph, to_sparse_recurrent
from fly_window.text.curriculum import generate_curriculum
from fly_window.text.gate import GateConfig, evaluate_launch_gate
from fly_window.text.policy import FlyTextPolicy
from fly_window.text.pretraining import (
    TextTrainingConfig,
    limit_corpus_sequences,
    load_cook_sequences,
    scheduled_sampling_loss,
    split_train_heldout,
    teacher_forcing_loss,
)
from fly_window.text.vocabulary import build_v1_vocabulary


def _git_sha() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True, encoding="utf-8"
    ).strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _batched_indices(length: int, batch_size: int, rng: random.Random) -> list[list[int]]:
    indices = list(range(length))
    rng.shuffle(indices)
    return [indices[start : start + batch_size] for start in range(0, length, batch_size)]


def _train_stage(
    *,
    policy: FlyTextPolicy,
    optimizer: torch.optim.Optimizer,
    sequences: list[tuple[int, ...]],
    epochs: int,
    batch_size: int,
    gradient_clip_norm: float,
    seed: int,
    stage_name: str,
    teacher_probability: float | None = None,
) -> list[float]:
    if epochs == 0:
        return []
    if not sequences:
        raise ValueError(f"{stage_name} requires non-empty sequences")

    epoch_losses: list[float] = []
    for epoch in range(epochs):
        rng = random.Random(seed + epoch)
        batch_losses: list[float] = []
        for batch_index, indices in enumerate(_batched_indices(len(sequences), batch_size, rng)):
            batch = [sequences[index] for index in indices]
            optimizer.zero_grad(set_to_none=True)
            if teacher_probability is None:
                loss = teacher_forcing_loss(policy, batch)
            else:
                loss = scheduled_sampling_loss(
                    policy,
                    batch,
                    teacher_probability=teacher_probability,
                    seed=seed + epoch * 1_000_003 + batch_index,
                )
            if not torch.isfinite(loss):
                raise FloatingPointError(f"{stage_name} produced non-finite loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), gradient_clip_norm)
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu()))

        epoch_loss = float(np.mean(batch_losses))
        epoch_losses.append(epoch_loss)
        print(
            json.dumps(
                {
                    "event": "epoch_complete",
                    "stage": stage_name,
                    "epoch": epoch + 1,
                    "epochs": epochs,
                    "mean_loss": epoch_loss,
                    "examples": len(sequences),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    return epoch_losses


def _resolve_device(name: str) -> torch.device:
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
        return torch.device("cuda")
    return torch.device("cpu")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Initial no-LLM Fly Tweets pretraining: deterministic grammar -> "
            "John D. Cook math corpus -> scheduled sampling."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/text_training_v1.json"),
    )
    parser.add_argument(
        "--gate-config",
        type=Path,
        default=Path("configs/text_gate_v1.json"),
    )
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
    parser.add_argument(
        "--graph-manifest",
        type=Path,
        default=Path("artifacts/graphs/subgraph_v1_manifest.json"),
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path("artifacts/corpus/john-d-cook/cook-math-tweets.jsonl"),
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path("artifacts/text-runs"),
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cpu",
        help="CPU is the reproducible default; CUDA must be requested explicitly.",
    )
    args = parser.parse_args()

    config = TextTrainingConfig.from_json(args.config)
    gate_config = GateConfig.from_json(args.gate_config)
    vocabulary = build_v1_vocabulary()
    device = _resolve_device(args.device)

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)

    graph = load_connectome_graph(args.graph_npz, args.nodes_parquet)
    recurrent = to_sparse_recurrent(graph)
    policy = FlyTextPolicy(
        graph,
        recurrent,
        vocab_size=len(vocabulary),
    ).to(device)

    trainable_parameters = [parameter for parameter in policy.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    curriculum = generate_curriculum(
        vocabulary,
        seed=config.seed,
        count=config.curriculum_examples,
    )
    source_corpus_sequences: list[tuple[int, ...]] = []
    corpus_sequences: list[tuple[int, ...]] = []
    train_corpus: list[tuple[int, ...]] = []
    heldout_corpus: list[tuple[int, ...]] = []
    if config.stage_b_epochs > 0 or config.stage_c_epochs > 0:
        if not args.corpus.exists():
            raise FileNotFoundError(
                f"Cook corpus not found: {args.corpus}. Run the headless collector first."
            )
        source_corpus_sequences = load_cook_sequences(
            args.corpus,
            vocabulary,
            max_oov_fraction=config.max_oov_fraction,
        )
        if len(source_corpus_sequences) < 2:
            raise ValueError("Cook corpus must yield at least two usable training sequences")
        corpus_sequences = limit_corpus_sequences(
            source_corpus_sequences,
            max_sequences=config.max_corpus_sequences,
            seed=config.seed + 35_000,
        )
        train_corpus, heldout_corpus = split_train_heldout(
            corpus_sequences,
            heldout_fraction=config.heldout_fraction,
            seed=config.seed + 40_000,
        )

    losses = {
        "stage_a": _train_stage(
            policy=policy,
            optimizer=optimizer,
            sequences=curriculum,
            epochs=config.stage_a_epochs,
            batch_size=config.batch_size,
            gradient_clip_norm=config.gradient_clip_norm,
            seed=config.seed + 10_000,
            stage_name="stage_a_curriculum",
        ),
        "stage_b": _train_stage(
            policy=policy,
            optimizer=optimizer,
            sequences=train_corpus,
            epochs=config.stage_b_epochs,
            batch_size=config.batch_size,
            gradient_clip_norm=config.gradient_clip_norm,
            seed=config.seed + 20_000,
            stage_name="stage_b_cook_corpus",
        ),
        "stage_c": _train_stage(
            policy=policy,
            optimizer=optimizer,
            sequences=train_corpus,
            epochs=config.stage_c_epochs,
            batch_size=config.batch_size,
            gradient_clip_norm=config.gradient_clip_norm,
            seed=config.seed + 30_000,
            stage_name="stage_c_scheduled_sampling",
            teacher_probability=config.stage_c_teacher_probability,
        ),
    }

    if not heldout_corpus:
        raise ValueError("launch gate requires a non-empty held-out Cook corpus")
    probe_seeds = tuple(
        config.seed + 50_000 + index for index in range(gate_config.probe_count)
    )
    gate_report = evaluate_launch_gate(
        policy,
        vocabulary,
        heldout=heldout_corpus,
        probe_seeds=probe_seeds,
        config=gate_config,
    )
    print(
        json.dumps(
            {
                "event": "launch_gate",
                **asdict(gate_report),
            },
            sort_keys=True,
        ),
        flush=True,
    )

    git_sha = _git_sha()
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"text-pretrain-{timestamp}-{git_sha[:8]}"
    run_dir = args.run_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    checkpoint_path = run_dir / "checkpoint.pt"
    torch.save(
        {
            "schema_version": 1,
            "model_state_dict": policy.state_dict(),
            "vocabulary": vocabulary.tokens,
            "git_sha": git_sha,
            "training_config": asdict(config),
            "gate_config": asdict(gate_config),
            "gate_report": asdict(gate_report),
        },
        checkpoint_path,
    )

    graph_manifest = json.loads(args.graph_manifest.read_text(encoding="utf-8"))
    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "created_at": datetime.now(UTC).isoformat(),
        "git_sha": git_sha,
        "device": str(device),
        "training_config": asdict(config),
        "graph": {
            "npz": str(args.graph_npz),
            "nodes_parquet": str(args.nodes_parquet),
            "manifest": str(args.graph_manifest),
            "artifact_sha256": graph_manifest.get("artifact_sha256"),
            "node_count": graph.node_count,
        },
        "corpus": {
            "path": str(args.corpus),
            "sha256": _sha256(args.corpus) if args.corpus.exists() else None,
            "source_usable_sequences": len(source_corpus_sequences),
            "selected_sequences": len(corpus_sequences),
            "train_sequences": len(train_corpus),
            "heldout_sequences": len(heldout_corpus),
        },
        "curriculum_sequences": len(curriculum),
        "losses": losses,
        "launch_gate": {
            "config": asdict(gate_config),
            "probe_seeds": list(probe_seeds),
            "report": asdict(gate_report),
        },
        "checkpoint": {
            "path": checkpoint_path.name,
            "sha256": _sha256(checkpoint_path),
        },
        "invariants": {
            "llm_in_training_path": False,
            "recurrent_connectome_trainable": bool(policy.recurrent.requires_grad),
            "vocabulary_size": len(vocabulary),
            "public_launch_allowed": bool(gate_report.passed),
        },
    }
    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "checkpoint": str(checkpoint_path),
                "manifest": str(manifest_path),
                "source_usable_corpus_sequences": len(source_corpus_sequences),
                "selected_corpus_sequences": len(corpus_sequences),
                "train_corpus_sequences": len(train_corpus),
                "heldout_corpus_sequences": len(heldout_corpus),
                "launch_gate_passed": gate_report.passed,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
