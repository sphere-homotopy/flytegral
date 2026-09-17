from __future__ import annotations

from collections.abc import Callable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from datetime import datetime

import torch

from fly_window.publishing.cadence import (
    FlyCadencePolicy,
    sample_publish_decisions,
)
from fly_window.publishing.cadence_training import (
    CadenceTrainingConfig,
    CadenceUpdateStats,
    apply_cadence_policy_update,
    cadence_training_examples_from_rows,
)
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
    cadence_training_applied: bool
    checkpoint_id: str
    batch_id: str
    consumed_keys: tuple[str, ...]
    update_stats: DailyUpdateStats | None
    cadence_update_stats: CadenceUpdateStats | None
    publish_times: tuple[datetime, ...]
    generated_rows: tuple[dict[str, object], ...]


CheckpointWriter = Callable[
    [FlyTextPolicy, FlyCadencePolicy, str, Mapping[str, object]], None
]
RowsAppender = Callable[[list[dict[str, object]]], None]


def _snapshot_state(policy: torch.nn.Module) -> dict[str, torch.Tensor]:
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
    cadence_training_config: CadenceTrainingConfig | None = None,
    cadence_temperature: float = 1.0,
    max_posts_per_24h: int = 30,
) -> ScheduledDailyCycleResult:
    """Run one production cycle where engagement updates text and cadence policies."""
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
    cadence_examples = cadence_training_examples_from_rows(
        rows,
        action_count=cadence_policy.action_count,
    )
    text_consumed_keys = {example.idempotency_key for example in examples}
    cadence_consumed_keys = {example.idempotency_key for example in cadence_examples}
    consumed_keys = tuple(sorted(text_consumed_keys | cadence_consumed_keys))
    consumed_set = set(consumed_keys)
    target_rows = [
        row
        for row in rows
        if not str(row.get("training_consumed_at", "") or "").strip()
        and str(row.get("idempotency_key", "") or "").strip() in consumed_set
    ]
    if len(target_rows) != len(consumed_set):
        raise ValueError("training rows do not map one-to-one to unconsumed examples")

    training_applied = bool(examples)
    cadence_training_applied = bool(cadence_examples) and cadence_training_config is not None
    any_training_applied = training_applied or cadence_training_applied
    active_checkpoint = next_checkpoint if any_training_applied else current_checkpoint
    text_snapshot = _snapshot_state(policy) if training_applied else None
    cadence_snapshot = _snapshot_state(cadence_policy) if cadence_training_applied else None
    update_stats: DailyUpdateStats | None = None
    cadence_update_stats: CadenceUpdateStats | None = None

    try:
        if training_applied:
            update_stats = apply_daily_policy_update(
                policy,
                anchor_policy,
                vocabulary=vocabulary,
                examples=examples,
                config=training_config,
            )
        if cadence_training_applied:
            assert cadence_training_config is not None
            cadence_update_stats = apply_cadence_policy_update(
                cadence_policy,
                examples=cadence_examples,
                config=cadence_training_config,
            )

        if any_training_applied:
            checkpoint_manifest: dict[str, object] = {
                "schema_version": 1,
                "checkpoint_id": next_checkpoint,
                "parent_checkpoint_id": current_checkpoint,
                "batch_id": batch,
                "git_sha": sha,
                "example_count": 0 if update_stats is None else update_stats.example_count,
                "update_steps": 0 if update_stats is None else update_stats.update_steps,
                "losses": [] if update_stats is None else list(update_stats.losses),
                "gradient_norms": [] if update_stats is None else list(update_stats.gradient_norms),
                "final_policy_loss": None if update_stats is None else update_stats.final_policy_loss,
                "final_entropy": None if update_stats is None else update_stats.final_entropy,
                "final_anchor_kl": None if update_stats is None else update_stats.final_anchor_kl,
                "cadence_example_count": (
                    0 if cadence_update_stats is None else cadence_update_stats.example_count
                ),
                "cadence_update_steps": (
                    0 if cadence_update_stats is None else cadence_update_stats.update_steps
                ),
                "cadence_losses": (
                    [] if cadence_update_stats is None else list(cadence_update_stats.losses)
                ),
                "cadence_gradient_norms": (
                    []
                    if cadence_update_stats is None
                    else list(cadence_update_stats.gradient_norms)
                ),
                "cadence_final_policy_loss": (
                    None
                    if cadence_update_stats is None
                    else cadence_update_stats.final_policy_loss
                ),
                "cadence_final_entropy": (
                    None if cadence_update_stats is None else cadence_update_stats.final_entropy
                ),
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

        decisions = sample_publish_decisions(
            cadence_policy,
            start=window_start,
            end=window_end,
            seed=int(cadence_seed),
            temperature=cadence_temperature,
            max_posts_per_24h=max_posts_per_24h,
        )
        publish_times = tuple(decision.publish_at for decision in decisions)
        generated = generate_batch(
            policy,
            vocabulary,
            base_seed=int(text_seed),
            config=generation_config,
            checkpoint_id=active_checkpoint,
            git_sha=sha,
            batch_id=batch,
            count=len(decisions),
        )
        batch_size = len(decisions)
        generated_rows = [
            generated_tweet_row(
                tweet,
                generated_at=now,
                scheduled_at=decision.publish_at,
                batch_size=batch_size,
                cadence_context=decision.context,
                cadence_action=decision.action,
            )
            for tweet, decision in zip(generated, decisions, strict=True)
        ]
        append_rows(generated_rows)

        if any_training_applied:
            consumed_at = now.isoformat()
            for row in target_rows:
                row["training_consumed_at"] = consumed_at
    except BaseException:
        if text_snapshot is not None:
            policy.load_state_dict(text_snapshot)
        if cadence_snapshot is not None:
            cadence_policy.load_state_dict(cadence_snapshot)
        raise

    return ScheduledDailyCycleResult(
        training_applied=training_applied,
        cadence_training_applied=cadence_training_applied,
        checkpoint_id=active_checkpoint,
        batch_id=batch,
        consumed_keys=consumed_keys,
        update_stats=update_stats,
        cadence_update_stats=cadence_update_stats,
        publish_times=publish_times,
        generated_rows=tuple(generated_rows),
    )
