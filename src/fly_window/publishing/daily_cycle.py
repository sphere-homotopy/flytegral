from __future__ import annotations

from collections.abc import Callable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from datetime import datetime

import torch

from fly_window.publishing.cadence import FlyCadencePolicy, sample_publish_times
from fly_window.publishing.rows import generated_tweet_row
from fly_window.text.daily_training import (
    DailyTrainingConfig,
    DailyUpdateStats,
    apply_daily_policy_update,
    training_examples_from_rows,
)
from fly_window.text.generation import GenerationConfig, generate_batch
from fly_window.text.policy import FlyTextPolicy
from fly_window.text.vocabulary import FlyVocabulary


@dataclass(frozen=True, slots=True)
class ScheduledDailyCycleResult:
    training_applied: bool
    checkpoint_id: str
    batch_id: str
    consumed_keys: tuple[str, ...]
    update_stats: DailyUpdateStats | None
    publish_times: tuple[datetime, ...]
    generated_rows: tuple[dict[str, object], ...]


CheckpointWriter = Callable[
    [FlyTextPolicy, FlyCadencePolicy, str, Mapping[str, object]], None
]
RowsAppender = Callable[[list[dict[str, object]]], None]


def _snapshot_state(policy: FlyTextPolicy) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().clone() for name, tensor in policy.state_dict().items()}


def _require_aware(value: datetime, name: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")


def run_scheduled_daily_cycle(
    policy: FlyTextPolicy,
    anchor_policy: FlyTextPolicy,
    cadence_policy: FlyCadencePolicy,
    *,
    vocabulary: FlyVocabulary,
    rows: Sequence[MutableMapping[str, object]],
    training_config: DailyTrainingConfig,
    generation_config: GenerationConfig,
    current_checkpoint_id: str,
    next_checkpoint_id: str,
    batch_id: str,
    git_sha: str,
    text_seed: int,
    cadence_seed: int,
    window_start: datetime,
    window_end: datetime,
    now: datetime,
    write_checkpoint: CheckpointWriter,
    append_rows: RowsAppender,
    cadence_temperature: float = 1.0,
    max_posts_per_24h: int = 30,
) -> ScheduledDailyCycleResult:
    """Run one production Fly Tweets cycle with cadence owned by MaleCNS.

    Settled rewards are optional. When they exist, the text policy is updated and a
    new immutable checkpoint is written before the newly scheduled queue rows become
    visible. When they do not exist, training is skipped but cadence inference and
    text generation still run from the current checkpoint. Queue append failure rolls
    back any text update and leaves source rows unconsumed.
    """
    current_checkpoint = str(current_checkpoint_id).strip()
    next_checkpoint = str(next_checkpoint_id).strip()
    batch = str(batch_id).strip()
    sha = str(git_sha).strip()
    if not current_checkpoint:
        raise ValueError("current_checkpoint_id is required")
    if not next_checkpoint:
        raise ValueError("next_checkpoint_id is required")
    if not batch:
        raise ValueError("batch_id is required")
    if not sha:
        raise ValueError("git_sha is required")
    _require_aware(window_start, "window_start")
    _require_aware(window_end, "window_end")
    _require_aware(now, "now")
    if window_end <= window_start:
        raise ValueError("window_end must be later than window_start")

    examples = training_examples_from_rows(rows, vocabulary=vocabulary)
    consumed_keys = tuple(example.idempotency_key for example in examples)
    consumed_set = set(consumed_keys)
    target_rows = [
        row
        for row in rows
        if not str(row.get("training_consumed_at", "") or "").strip()
        and str(row.get("idempotency_key", "") or "").strip() in consumed_set
    ]
    if len(target_rows) != len(examples):
        raise ValueError("training rows do not map one-to-one to unconsumed examples")

    training_applied = bool(examples)
    active_checkpoint = next_checkpoint if training_applied else current_checkpoint
    snapshot = _snapshot_state(policy) if training_applied else None
    update_stats: DailyUpdateStats | None = None

    try:
        if training_applied:
            update_stats = apply_daily_policy_update(
                policy,
                anchor_policy,
                vocabulary=vocabulary,
                examples=examples,
                config=training_config,
            )
            checkpoint_manifest: dict[str, object] = {
                "schema_version": 1,
                "checkpoint_id": next_checkpoint,
                "parent_checkpoint_id": current_checkpoint,
                "batch_id": batch,
                "git_sha": sha,
                "example_count": update_stats.example_count,
                "update_steps": update_stats.update_steps,
                "losses": list(update_stats.losses),
                "gradient_norms": list(update_stats.gradient_norms),
                "final_policy_loss": update_stats.final_policy_loss,
                "final_entropy": update_stats.final_entropy,
                "final_anchor_kl": update_stats.final_anchor_kl,
                "consumed_keys": list(consumed_keys),
                "completed_at": now.isoformat(),
                "llm_in_training_path": False,
                "recurrent_connectome_trainable": bool(policy.recurrent.requires_grad),
                "cadence_recurrent_connectome_trainable": bool(
                    cadence_policy.recurrent.requires_grad
                ),
            }
            write_checkpoint(
                policy,
                cadence_policy,
                next_checkpoint,
                checkpoint_manifest,
            )

        publish_times = sample_publish_times(
            cadence_policy,
            start=window_start,
            end=window_end,
            seed=int(cadence_seed),
            temperature=cadence_temperature,
            max_posts_per_24h=max_posts_per_24h,
        )
        generated = generate_batch(
            policy,
            vocabulary,
            base_seed=int(text_seed),
            config=generation_config,
            checkpoint_id=active_checkpoint,
            git_sha=sha,
            batch_id=batch,
            count=len(publish_times),
        )
        generated_rows = [
            generated_tweet_row(
                tweet,
                generated_at=now,
                scheduled_at=publish_at,
            )
            for tweet, publish_at in zip(generated, publish_times, strict=True)
        ]
        append_rows(generated_rows)

        if training_applied:
            consumed_at = now.isoformat()
            for row in target_rows:
                row["training_consumed_at"] = consumed_at
    except BaseException:
        if snapshot is not None:
            policy.load_state_dict(snapshot)
        raise

    return ScheduledDailyCycleResult(
        training_applied=training_applied,
        checkpoint_id=active_checkpoint,
        batch_id=batch,
        consumed_keys=consumed_keys,
        update_stats=update_stats,
        publish_times=publish_times,
        generated_rows=tuple(generated_rows),
    )
