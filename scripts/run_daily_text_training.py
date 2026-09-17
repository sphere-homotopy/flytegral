from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from fly_window.neural.graph import load_connectome_graph, to_sparse_recurrent
from fly_window.publishing.cadence import FlyCadencePolicy
from fly_window.publishing.daily_cycle import run_scheduled_daily_cycle
from fly_window.publishing.runtime import (
    RuntimeConfig,
    RuntimeState,
    generation_window,
    health_alerts,
)
from fly_window.text.daily_training import DailyTrainingConfig
from fly_window.text.generation import GenerationConfig
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


def _parse_datetime(value: object, *, field: str) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field} must be timezone-aware")
    return parsed.astimezone(UTC)


def _load_rows(path: Path) -> list[dict[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("rows")
    if not isinstance(payload, list) or any(not isinstance(row, dict) for row in payload):
        raise ValueError("rows-json must contain a JSON array of row objects")
    return [dict(row) for row in payload]


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    _write_json(path, rows)


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


def _build_cadence_policy(
    checkpoint: dict[str, Any],
    *,
    graph_npz: Path,
    nodes_parquet: Path,
    device: torch.device,
) -> tuple[FlyCadencePolicy, float, int]:
    raw_waits = checkpoint.get("wait_minutes")
    if not isinstance(raw_waits, (list, tuple)) or not raw_waits:
        raise ValueError("cadence checkpoint is missing wait_minutes")
    wait_minutes = tuple(int(value) for value in raw_waits)
    graph = load_connectome_graph(graph_npz, nodes_parquet)
    policy = FlyCadencePolicy(
        graph,
        to_sparse_recurrent(graph),
        wait_minutes=wait_minutes,
    ).to(device)
    state_dict = checkpoint.get("cadence_state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError("cadence checkpoint is missing cadence_state_dict")
    policy.load_state_dict(state_dict)
    temperature = float(checkpoint.get("temperature", 1.0))
    max_posts = int(checkpoint.get("max_posts_per_24h", 30))
    return policy, temperature, max_posts


def _load_runtime_state(path: Path) -> RuntimeState:
    if not path.exists():
        return RuntimeState()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("runtime-state must contain a JSON object")
    return RuntimeState(
        last_stats_at=_parse_datetime(payload.get("last_stats_at"), field="last_stats_at"),
        last_training_at=_parse_datetime(
            payload.get("last_training_at"), field="last_training_at"
        ),
        last_generation_at=_parse_datetime(
            payload.get("last_generation_at"), field="last_generation_at"
        ),
        queue_horizon_at=_parse_datetime(
            payload.get("queue_horizon_at"), field="queue_horizon_at"
        ),
        current_checkpoint=str(payload.get("current_checkpoint", "") or ""),
        last_alert_at=_parse_datetime(payload.get("last_alert_at"), field="last_alert_at"),
    )


def _write_runtime_state(path: Path, state: RuntimeState) -> None:
    payload = {
        "last_stats_at": None if state.last_stats_at is None else state.last_stats_at.isoformat(),
        "last_training_at": (
            None if state.last_training_at is None else state.last_training_at.isoformat()
        ),
        "last_generation_at": (
            None if state.last_generation_at is None else state.last_generation_at.isoformat()
        ),
        "queue_horizon_at": (
            None if state.queue_horizon_at is None else state.queue_horizon_at.isoformat()
        ),
        "current_checkpoint": state.current_checkpoint,
        "last_alert_at": None if state.last_alert_at is None else state.last_alert_at.isoformat(),
    }
    _write_json(path, payload)


def _latest_metrics_collected_at(rows: list[dict[str, object]]) -> datetime | None:
    values = [
        _parse_datetime(row.get("metrics_collected_at"), field="metrics_collected_at")
        for row in rows
        if str(row.get("metrics_collected_at", "") or "").strip()
    ]
    return max((value for value in values if value is not None), default=None)


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


def _checkpoint_label(path: Path) -> str:
    return path.parent.name or path.stem


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run one durable no-LLM Fly Tweets cycle. MaleCNS cadence owns both "
            "publication count and absolute publish times. Fresh settled engagement "
            "updates the text policy when available; missing stats never block "
            "generation."
        )
    )
    parser.add_argument("--request-date", required=True, help="Scheduled occurrence date, YYYY-MM-DD")
    parser.add_argument("--attempt", type=int, required=True, help="Stable retry attempt number")
    parser.add_argument("--current-checkpoint", type=Path, required=True)
    parser.add_argument("--anchor-checkpoint", type=Path, required=True)
    parser.add_argument("--cadence-checkpoint", type=Path, required=True)
    parser.add_argument("--runtime-state", type=Path, required=True)
    parser.add_argument("--runtime-config", type=Path, required=True)
    parser.add_argument("--rows-json", type=Path, required=True)
    parser.add_argument("--outbox-jsonl", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--reward-config", type=Path, required=True)
    parser.add_argument("--training-config", type=Path, required=True)
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

    if args.now:
        parsed_now = _parse_datetime(args.now, field="now")
        if parsed_now is None:
            raise ValueError("now must not be empty")
        now = parsed_now
    else:
        now = datetime.now(UTC)

    runtime_config = RuntimeConfig.from_json(args.runtime_config)
    runtime_state = _load_runtime_state(args.runtime_state)
    reward_config = RewardConfig.from_json(args.reward_config)
    training_config = DailyTrainingConfig.from_json(args.training_config)
    generation_config = GenerationConfig(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        max_chars=args.max_chars,
    )
    rows = _load_rows(args.rows_json)
    rewarded_rows = _apply_settled_rewards(rows, reward_config=reward_config, now=now)
    latest_stats = _latest_metrics_collected_at(rows)

    device = _resolve_device(args.device)
    current_checkpoint = _load_checkpoint(args.current_checkpoint, device)
    anchor_checkpoint = _load_checkpoint(args.anchor_checkpoint, device)
    cadence_checkpoint = _load_checkpoint(args.cadence_checkpoint, device)

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

    cadence_policy, cadence_temperature, cadence_max_posts = _build_cadence_policy(
        cadence_checkpoint,
        graph_npz=args.graph_npz,
        nodes_parquet=args.nodes_parquet,
        device=device,
    )
    if min(cadence_policy.wait_minutes) < runtime_config.min_gap_minutes:
        raise ValueError("cadence checkpoint violates runtime min_gap_minutes safety bound")
    max_posts_per_24h = min(runtime_config.max_posts_per_24h, cadence_max_posts)

    bootstrap = runtime_state.queue_horizon_at is None
    window_start, window_end = generation_window(
        runtime_state,
        now=now,
        bootstrap=bootstrap,
        config=runtime_config,
    )

    git_sha = _git_sha()
    occurrence = request_date.isoformat()
    next_checkpoint_id = f"daily-{occurrence}-a{args.attempt}"
    batch_id = f"batch-{occurrence}-a{args.attempt}"
    current_checkpoint_id = runtime_state.current_checkpoint.strip() or _checkpoint_label(
        args.current_checkpoint
    )
    run_dir = args.run_root / next_checkpoint_id

    def write_checkpoint(updated_policy, updated_cadence, written_checkpoint_id, manifest):
        if written_checkpoint_id != next_checkpoint_id:
            raise ValueError("checkpoint writer received unexpected checkpoint id")
        run_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = run_dir / "manifest.json"
        checkpoint_path = run_dir / "checkpoint.pt"
        if manifest_path.exists() or checkpoint_path.exists():
            if not manifest_path.exists() or not checkpoint_path.exists():
                raise RuntimeError("partial immutable daily checkpoint already exists")
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            stable_fields = ("checkpoint_id", "parent_checkpoint_id", "batch_id", "git_sha")
            if any(existing.get(field) != manifest.get(field) for field in stable_fields):
                raise RuntimeError("conflicting immutable daily checkpoint already exists")
            return

        payload = dict(current_checkpoint)
        payload["schema_version"] = max(int(payload.get("schema_version", 1)), 1)
        payload["model_state_dict"] = updated_policy.state_dict()
        payload["cadence_state_dict"] = updated_cadence.state_dict()
        payload["cadence_wait_minutes"] = tuple(updated_cadence.wait_minutes)
        payload["git_sha"] = git_sha
        payload["parent_checkpoint"] = str(args.current_checkpoint)
        payload["cadence_checkpoint"] = str(args.cadence_checkpoint)
        payload["daily_manifest"] = dict(manifest)

        checkpoint_tmp = checkpoint_path.with_suffix(".pt.tmp")
        torch.save(payload, checkpoint_tmp)
        checkpoint_tmp.replace(checkpoint_path)
        _write_json(manifest_path, dict(manifest))

    result = run_scheduled_daily_cycle(
        policy,
        anchor_policy,
        cadence_policy,
        vocabulary=vocabulary,
        rows=rows,
        training_config=training_config,
        generation_config=generation_config,
        current_checkpoint_id=current_checkpoint_id,
        next_checkpoint_id=next_checkpoint_id,
        batch_id=batch_id,
        git_sha=git_sha,
        text_seed=args.base_seed,
        cadence_seed=args.base_seed + 1_000_003,
        window_start=window_start,
        window_end=window_end,
        now=now,
        write_checkpoint=write_checkpoint,
        append_rows=lambda new_rows: _append_outbox(args.outbox_jsonl, new_rows),
        cadence_temperature=cadence_temperature,
        max_posts_per_24h=max_posts_per_24h,
    )

    _write_rows(args.rows_json, rows)
    deployed_checkpoint = (
        run_dir / "checkpoint.pt" if result.training_applied else args.current_checkpoint
    )
    last_stats_at = runtime_state.last_stats_at
    if latest_stats is not None and (last_stats_at is None or latest_stats > last_stats_at):
        last_stats_at = latest_stats
    next_state = replace(
        runtime_state,
        last_stats_at=last_stats_at,
        last_training_at=now if result.training_applied else runtime_state.last_training_at,
        last_generation_at=now,
        queue_horizon_at=window_end,
        current_checkpoint=result.checkpoint_id,
    )
    _write_runtime_state(args.runtime_state, next_state)
    alerts = health_alerts(next_state, now=now, config=runtime_config)

    print(
        json.dumps(
            {
                "request_date": occurrence,
                "attempt": args.attempt,
                "bootstrap": bootstrap,
                "training_applied": result.training_applied,
                "rewarded_rows": rewarded_rows,
                "consumed_rows": len(result.consumed_keys),
                "generated_rows": len(result.generated_rows),
                "publish_times": [value.isoformat() for value in result.publish_times],
                "window_start": window_start.isoformat(),
                "window_end": window_end.isoformat(),
                "checkpoint": str(deployed_checkpoint),
                "current_checkpoint_id": result.checkpoint_id,
                "runtime_state": str(args.runtime_state),
                "outbox": str(args.outbox_jsonl),
                "alerts": [
                    {
                        "code": alert.code,
                        "message": alert.message,
                        "urgent": alert.urgent,
                    }
                    for alert in alerts
                ],
                "llm_in_runtime": False,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
