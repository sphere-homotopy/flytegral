from __future__ import annotations

import argparse
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from fly_window.neural.graph import load_connectome_graph, to_sparse_recurrent
from fly_window.publishing.rows import generated_tweet_row
from fly_window.text.daily_training import DailyTrainingConfig, run_daily_update
from fly_window.text.generation import GenerationConfig, generate_batch
from fly_window.text.policy import FlyTextPolicy
from fly_window.text.reward import RewardConfig, compute_daily_rewards
from fly_window.text.vocabulary import build_v1_vocabulary


def _git_sha() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True, encoding="utf-8"
    ).strip()


def _resolve_device(name: str) -> torch.device:
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
        return torch.device("cuda")
    return torch.device("cpu")


def _load_rows(path: Path) -> list[dict[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("rows")
    if not isinstance(payload, list) or any(not isinstance(row, dict) for row in payload):
        raise ValueError("rows-json must contain a JSON array of row objects")
    return [dict(row) for row in payload]


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    payload = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError(f"checkpoint payload must be a mapping: {path}")
    return payload


def _build_policy(
    checkpoint: dict[str, Any],
    *,
    graph_npz: Path,
    nodes_parquet: Path,
    device: torch.device,
) -> tuple[FlyTextPolicy, object]:
    vocabulary = build_v1_vocabulary()
    if tuple(checkpoint.get("vocabulary", ())) != vocabulary.tokens:
        raise ValueError("checkpoint vocabulary does not exactly match Fly Tweets v1")
    graph = load_connectome_graph(graph_npz, nodes_parquet)
    policy = FlyTextPolicy(
        graph,
        to_sparse_recurrent(graph),
        vocab_size=len(vocabulary),
    ).to(device)
    policy.load_state_dict(checkpoint["model_state_dict"])
    return policy, vocabulary


def _existing_outbox(path: Path) -> dict[str, dict[str, object]]:
    rows: dict[str, dict[str, object]] = {}
    if not path.exists():
        return rows
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"outbox line {line_number} is not an object")
        key = str(row.get("idempotency_key", "")).strip()
        if not key:
            raise ValueError(f"outbox line {line_number} is missing idempotency_key")
        if key in rows and rows[key] != row:
            raise ValueError(f"conflicting duplicate outbox key: {key}")
        rows[key] = row
    return rows


def _append_outbox(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = _existing_outbox(path)
    additions: list[dict[str, object]] = []
    for row in rows:
        key = str(row.get("idempotency_key", "")).strip()
        if not key:
            raise ValueError("generated row missing idempotency_key")
        previous = existing.get(key)
        if previous is not None:
            if previous != row:
                raise ValueError(f"idempotency conflict for outbox key: {key}")
            continue
        existing[key] = row
        additions.append(row)
    if not additions:
        return
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        for row in additions:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _apply_settled_rewards(
    rows: list[dict[str, object]],
    *,
    reward_config: RewardConfig,
    now: datetime,
) -> int:
    published = [row for row in rows if str(row.get("published_at", "") or "").strip()]
    if not published:
        return 0
    rewarded = compute_daily_rewards(published, config=reward_config, now=now)
    rewards = {item.idempotency_key: item.reward for item in rewarded}
    updated = 0
    for row in rows:
        key = str(row.get("idempotency_key", "") or "").strip()
        if key in rewards and not str(row.get("training_consumed_at", "") or "").strip():
            row["reward"] = rewards[key]
            updated += 1
    return updated


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run one durable no-LLM Fly Tweets daily cycle. Fresh settled engagement "
            "is used for a conservative update when available; otherwise training is "
            "skipped and generation continues from the latest valid checkpoint."
        )
    )
    parser.add_argument("--request-date", required=True, help="Scheduled occurrence date, YYYY-MM-DD")
    parser.add_argument("--attempt", type=int, required=True, help="Stable retry attempt number")
    parser.add_argument("--current-checkpoint", type=Path, required=True)
    parser.add_argument("--anchor-checkpoint", type=Path, required=True)
    parser.add_argument("--rows-json", type=Path, required=True)
    parser.add_argument("--outbox-jsonl", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--reward-config", type=Path, required=True)
    parser.add_argument("--training-config", type=Path, required=True)
    parser.add_argument("--generation-count", type=int, required=True, help="Count selected by the cadence policy")
    parser.add_argument("--base-seed", type=int, required=True)
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
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--max-chars", type=int, default=240)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--now",
        default="",
        help="Optional timezone-aware ISO timestamp for deterministic replay/testing.",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    try:
        request_date = datetime.strptime(args.request_date, "%Y-%m-%d").date()
    except ValueError as error:
        raise ValueError("request-date must be YYYY-MM-DD") from error
    if args.attempt <= 0:
        raise ValueError("attempt must be positive")
    if args.generation_count <= 0:
        raise ValueError("generation-count must be positive")

    if args.now:
        now = datetime.fromisoformat(args.now.replace("Z", "+00:00"))
        if now.tzinfo is None or now.utcoffset() is None:
            raise ValueError("now must be timezone-aware")
        now = now.astimezone(UTC)
    else:
        now = datetime.now(UTC)

    reward_config = RewardConfig.from_json(args.reward_config)
    training_config = DailyTrainingConfig.from_json(args.training_config)
    generation_config = GenerationConfig(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        max_chars=args.max_chars,
    )
    rows = _load_rows(args.rows_json)
    rewarded_rows = _apply_settled_rewards(rows, reward_config=reward_config, now=now)

    device = _resolve_device(args.device)
    current_checkpoint = _load_checkpoint(args.current_checkpoint, device)
    anchor_checkpoint = _load_checkpoint(args.anchor_checkpoint, device)
    policy, vocabulary = _build_policy(
        current_checkpoint,
        graph_npz=args.graph_npz,
        nodes_parquet=args.nodes_parquet,
        device=device,
    )
    anchor_policy, anchor_vocabulary = _build_policy(
        anchor_checkpoint,
        graph_npz=args.graph_npz,
        nodes_parquet=args.nodes_parquet,
        device=device,
    )
    if anchor_vocabulary.tokens != vocabulary.tokens:
        raise ValueError("current and anchor checkpoint vocabularies differ")

    git_sha = _git_sha()
    occurrence = request_date.isoformat()
    checkpoint_id = f"daily-{occurrence}-a{args.attempt}"
    batch_id = f"batch-{occurrence}-a{args.attempt}"
    run_dir = args.run_root / checkpoint_id

    eligible_for_training = any(
        row.get("reward", "") not in (None, "")
        and not str(row.get("training_consumed_at", "") or "").strip()
        for row in rows
    )

    if eligible_for_training:
        def write_checkpoint(updated_policy, written_checkpoint_id, manifest):
            if written_checkpoint_id != checkpoint_id:
                raise ValueError("checkpoint writer received unexpected checkpoint id")
            run_dir.mkdir(parents=True, exist_ok=False)
            payload = dict(current_checkpoint)
            payload["schema_version"] = max(int(payload.get("schema_version", 1)), 1)
            payload["model_state_dict"] = updated_policy.state_dict()
            payload["git_sha"] = git_sha
            payload["parent_checkpoint"] = str(args.current_checkpoint)
            payload["daily_manifest"] = dict(manifest)
            torch.save(payload, run_dir / "checkpoint.pt")
            (run_dir / "manifest.json").write_text(
                json.dumps(dict(manifest), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

        result = run_daily_update(
            policy,
            anchor_policy,
            vocabulary=vocabulary,
            rows=rows,
            config=training_config,
            generation_config=generation_config,
            checkpoint_id=checkpoint_id,
            batch_id=batch_id,
            git_sha=git_sha,
            base_seed=args.base_seed,
            now=now,
            write_checkpoint=write_checkpoint,
            append_rows=lambda new_rows: _append_outbox(args.outbox_jsonl, new_rows),
            generation_count=args.generation_count,
        )
        deployed_checkpoint = run_dir / "checkpoint.pt"
        training_applied = True
        consumed = len(result.consumed_keys)
        generated_count = len(result.generated_rows)
    else:
        source_checkpoint_id = args.current_checkpoint.parent.name or args.current_checkpoint.stem
        generated = generate_batch(
            policy,
            vocabulary,
            base_seed=args.base_seed,
            config=generation_config,
            checkpoint_id=source_checkpoint_id,
            git_sha=str(current_checkpoint.get("git_sha", git_sha)),
            batch_id=batch_id,
            count=args.generation_count,
        )
        generated_rows = [generated_tweet_row(tweet, generated_at=now) for tweet in generated]
        _append_outbox(args.outbox_jsonl, generated_rows)
        deployed_checkpoint = args.current_checkpoint
        training_applied = False
        consumed = 0
        generated_count = len(generated_rows)

    _write_rows(args.rows_json, rows)
    print(
        json.dumps(
            {
                "request_date": occurrence,
                "attempt": args.attempt,
                "training_applied": training_applied,
                "rewarded_rows": rewarded_rows,
                "consumed_rows": consumed,
                "generated_rows": generated_count,
                "checkpoint": str(deployed_checkpoint),
                "outbox": str(args.outbox_jsonl),
                "llm_in_runtime": False,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
