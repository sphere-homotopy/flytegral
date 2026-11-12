from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import torch
from torch.nn import functional as F

from fly_window.publishing.rows import generated_tweet_row
from fly_window.text.generation import GenerationConfig, generate_batch
from fly_window.text.policy import FlyTextPolicy
from fly_window.text.vocabulary import FlyVocabulary


@dataclass(frozen=True, slots=True)
class DailyTrainingConfig:
    learning_rate: float
    weight_decay: float
    entropy_coefficient: float
    anchor_kl_coefficient: float
    gradient_clip_norm: float
    max_update_steps: int
    training_mode: str = "full"

    def __post_init__(self) -> None:
        finite_nonnegative = {
            "weight_decay": self.weight_decay,
            "entropy_coefficient": self.entropy_coefficient,
            "anchor_kl_coefficient": self.anchor_kl_coefficient,
        }
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be finite and positive")
        for name, value in finite_nonnegative.items():
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if not math.isfinite(self.gradient_clip_norm) or self.gradient_clip_norm <= 0.0:
            raise ValueError("gradient_clip_norm must be finite and positive")
        if self.max_update_steps <= 0:
            raise ValueError("max_update_steps must be positive")
        if self.training_mode not in ("full", "readout_only"):
            raise ValueError("training_mode must be full or readout_only")

    @classmethod
    def from_json(cls, path: Path) -> "DailyTrainingConfig":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("daily training config must be a JSON object")
        try:
            return cls(
                learning_rate=float(payload["learning_rate"]),
                weight_decay=float(payload.get("weight_decay", 0.0)),
                entropy_coefficient=float(payload["entropy_coefficient"]),
                anchor_kl_coefficient=float(payload["anchor_kl_coefficient"]),
                gradient_clip_norm=float(payload["gradient_clip_norm"]),
                max_update_steps=int(payload["max_update_steps"]),
                training_mode=str(payload.get("training_mode", "full")),
            )
        except KeyError as error:
            raise ValueError(f"daily training config missing {error.args[0]}") from error


@dataclass(frozen=True, slots=True)
class TrainingExample:
    idempotency_key: str
    token_ids: tuple[int, ...]
    reward: float


@dataclass(frozen=True, slots=True)
class DailyPolicyObjective:
    loss: torch.Tensor
    policy_loss: torch.Tensor
    entropy: torch.Tensor
    anchor_kl: torch.Tensor


@dataclass(frozen=True, slots=True)
class DailyUpdateStats:
    example_count: int
    update_steps: int
    losses: tuple[float, ...]
    gradient_norms: tuple[float, ...]
    final_policy_loss: float
    final_entropy: float
    final_anchor_kl: float


@dataclass(frozen=True, slots=True)
class DailyRunResult:
    checkpoint_id: str
    batch_id: str
    consumed_keys: tuple[str, ...]
    update_stats: DailyUpdateStats
    generated_rows: tuple[dict[str, object], ...]


CheckpointWriter = Callable[[FlyTextPolicy, str, Mapping[str, object]], None]
RowsAppender = Callable[[list[dict[str, object]]], None]


def _parse_token_ids(value: object) -> tuple[int, ...]:
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return ()
        try:
            value = json.loads(text)
        except json.JSONDecodeError as error:
            raise ValueError("token_ids must be valid JSON") from error
    if not isinstance(value, (list, tuple)):
        raise ValueError("token_ids must be a JSON array or sequence")
    try:
        return tuple(int(token_id) for token_id in value)
    except (TypeError, ValueError) as error:
        raise ValueError("token_ids must contain integers") from error


def training_examples_from_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    vocabulary: FlyVocabulary,
) -> list[TrainingExample]:
    examples: list[TrainingExample] = []
    seen: set[str] = set()
    for row in rows:
        consumed = str(row.get("training_consumed_at", "") or "").strip()
        if consumed:
            continue

        reward_value = row.get("reward", "")
        if reward_value in (None, ""):
            continue

        key = str(row.get("idempotency_key", "") or "").strip()
        if not key:
            raise ValueError("idempotency_key is required for unconsumed rewarded rows")
        if key in seen:
            raise ValueError(f"duplicate idempotency_key: {key}")
        seen.add(key)

        reward = float(reward_value)
        if not math.isfinite(reward):
            raise ValueError(f"non-finite reward for {key}")

        token_ids = _parse_token_ids(row.get("token_ids", ""))
        if not token_ids:
            raise ValueError(f"token_ids must not be empty for {key}")
        if any(token_id < 0 or token_id >= len(vocabulary) for token_id in token_ids):
            raise ValueError(f"token id outside vocabulary for {key}")

        examples.append(
            TrainingExample(
                idempotency_key=key,
                token_ids=token_ids,
                reward=reward,
            )
        )
    return examples


def _assert_compatible_anchor(policy: FlyTextPolicy, anchor_policy: FlyTextPolicy) -> None:
    if policy.vocab_size != anchor_policy.vocab_size:
        raise ValueError("policy and anchor vocabulary sizes differ")
    if policy.node_count != anchor_policy.node_count:
        raise ValueError("policy and anchor graph sizes differ")

    current = policy.recurrent.coalesce()
    anchor = anchor_policy.recurrent.coalesce()
    if not torch.equal(current.indices().cpu(), anchor.indices().cpu()) or not torch.equal(
        current.values().cpu(), anchor.values().cpu()
    ):
        raise ValueError("policy and anchor recurrent connectomes differ")
    if current.requires_grad or anchor.requires_grad:
        raise ValueError("recurrent connectome must remain frozen")


def daily_policy_objective(
    policy: FlyTextPolicy,
    anchor_policy: FlyTextPolicy,
    *,
    vocabulary: FlyVocabulary,
    examples: Sequence[TrainingExample],
    config: DailyTrainingConfig,
    verify_anchor: bool = True,
) -> DailyPolicyObjective:
    if not examples:
        raise ValueError("no unconsumed training examples")
    if policy.vocab_size != len(vocabulary):
        raise ValueError("policy vocabulary size does not match vocabulary")
    if verify_anchor:
        _assert_compatible_anchor(policy, anchor_policy)

    device = policy.input_gain.device
    if anchor_policy.input_gain.device != device:
        raise ValueError("policy and anchor must be on the same device")
    bos_id = vocabulary.id_for("<BOS>")

    policy_terms: list[torch.Tensor] = []
    entropies: list[torch.Tensor] = []
    anchor_kls: list[torch.Tensor] = []

    for example in examples:
        state = policy.initial_state(batch_size=1)
        anchor_state = anchor_policy.initial_state(batch_size=1)
        current_token = bos_id
        chosen_logprobs: list[torch.Tensor] = []

        for target_id in example.token_ids:
            input_ids = torch.tensor([current_token], dtype=torch.long, device=device)
            output = policy.step(input_ids, state)
            if not bool(torch.isfinite(output.logits).all().item()) or not bool(
                torch.isfinite(output.state).all().item()
            ):
                raise FloatingPointError("non-finite policy state or logits")

            log_probs = F.log_softmax(output.logits, dim=-1)
            probabilities = log_probs.exp()
            chosen_logprobs.append(log_probs[0, int(target_id)])
            entropies.append(-(probabilities * log_probs).sum(dim=-1).mean())

            with torch.no_grad():
                anchor_output = anchor_policy.step(input_ids, anchor_state)
                if not bool(torch.isfinite(anchor_output.logits).all().item()) or not bool(
                    torch.isfinite(anchor_output.state).all().item()
                ):
                    raise FloatingPointError("non-finite anchor state or logits")
                anchor_log_probs = F.log_softmax(anchor_output.logits, dim=-1)

            anchor_kls.append(
                (probabilities * (log_probs - anchor_log_probs)).sum(dim=-1).mean()
            )
            state = output.state
            anchor_state = anchor_output.state
            current_token = int(target_id)

        mean_logprob = torch.stack(chosen_logprobs).mean()
        policy_terms.append(-float(example.reward) * mean_logprob)

    policy_loss = torch.stack(policy_terms).mean()
    entropy = torch.stack(entropies).mean()
    anchor_kl = torch.stack(anchor_kls).mean()
    loss = (
        policy_loss
        - config.entropy_coefficient * entropy
        + config.anchor_kl_coefficient * anchor_kl
    )
    return DailyPolicyObjective(
        loss=loss,
        policy_loss=policy_loss,
        entropy=entropy,
        anchor_kl=anchor_kl,
    )


def _snapshot_state(policy: FlyTextPolicy) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().clone() for name, tensor in policy.state_dict().items()}


def _all_trainable_parameters_finite(policy: FlyTextPolicy) -> bool:
    return all(
        bool(torch.isfinite(parameter).all().item())
        for parameter in policy.parameters()
        if parameter.requires_grad
    )



@dataclass(frozen=True, slots=True)
class CachedReadoutTrajectory:
    """Frozen MaleCNS motor activations for one rewarded token sequence.

    This in-memory cache is valid while the input encoder and recurrent wiring
    remain fixed. Updating only the readout does not invalidate the features.
    """

    features: torch.Tensor
    targets: torch.Tensor
    anchor_log_probs: torch.Tensor
    reward: float


def _cache_readout_trajectories(
    policy: FlyTextPolicy,
    anchor_policy: FlyTextPolicy,
    *,
    examples: Sequence[TrainingExample],
    vocabulary: FlyVocabulary,
) -> list[CachedReadoutTrajectory]:
    device = policy.input_gain.device
    if anchor_policy.input_gain.device != device:
        raise ValueError("policy and anchor must be on the same device")
    bos_id = vocabulary.id_for("<BOS>")
    cached: list[CachedReadoutTrajectory] = []

    # No autograd through the 1.54M-edge graph: only the trainable readout
    # will be evaluated with gradients below.
    with torch.no_grad():
        for example in examples:
            state = policy.initial_state(batch_size=1)
            anchor_state = anchor_policy.initial_state(batch_size=1)
            current_token = bos_id
            features: list[torch.Tensor] = []
            anchor_log_probs: list[torch.Tensor] = []
            for target in example.token_ids:
                input_ids = torch.tensor([current_token], dtype=torch.long, device=device)
                output = policy.step(input_ids, state)
                anchor_output = anchor_policy.step(input_ids, anchor_state)
                if (
                    not bool(torch.isfinite(output.state).all().item())
                    or not bool(torch.isfinite(anchor_output.state).all().item())
                    or not bool(torch.isfinite(anchor_output.logits).all().item())
                ):
                    raise FloatingPointError("non-finite frozen connectome trajectory")
                features.append(output.state[0, policy.output_indices])
                anchor_log_probs.append(F.log_softmax(anchor_output.logits[0], dim=-1))
                state = output.state
                anchor_state = anchor_output.state
                current_token = int(target)
            cached.append(
                CachedReadoutTrajectory(
                    features=torch.stack(features),
                    targets=torch.tensor(example.token_ids, dtype=torch.long, device=device),
                    anchor_log_probs=torch.stack(anchor_log_probs),
                    reward=float(example.reward),
                )
            )
    return cached


def _apply_readout_only_policy_update(
    policy: FlyTextPolicy,
    anchor_policy: FlyTextPolicy,
    *,
    vocabulary: FlyVocabulary,
    examples: Sequence[TrainingExample],
    config: DailyTrainingConfig,
) -> DailyUpdateStats:
    """Train the motor readout while keeping sensory input and MaleCNS fixed.

    Cached motor activations are reused across optimizer steps. This retains
    the same policy-gradient, entropy, and anchor-KL objective as full mode,
    restricted to the readout parameters.
    """
    trainable = [p for p in policy.readout.parameters() if p.requires_grad]
    if not trainable:
        raise ValueError("readout-only training requires a trainable readout")
    if policy.vocab_size != len(vocabulary):
        raise ValueError("policy vocabulary size does not match vocabulary")
    cached = _cache_readout_trajectories(
        policy, anchor_policy, examples=examples, vocabulary=vocabulary
    )
    total_tokens = sum(len(example.token_ids) for example in examples)
    snapshot = {
        key: value.detach().clone() for key, value in policy.readout.state_dict().items()
    }
    optimizer = torch.optim.AdamW(
        trainable, lr=config.learning_rate, weight_decay=config.weight_decay
    )
    losses: list[float] = []
    gradient_norms: list[float] = []
    final_policy_loss = 0.0
    final_entropy = 0.0
    final_anchor_kl = 0.0
    try:
        for _ in range(config.max_update_steps):
            optimizer.zero_grad(set_to_none=True)
            policy_value = 0.0
            entropy_value = 0.0
            anchor_kl_value = 0.0
            for trajectory in cached:
                logits = policy.readout(trajectory.features)
                if not bool(torch.isfinite(logits).all().item()):
                    raise FloatingPointError("non-finite readout logits")
                log_probs = F.log_softmax(logits, dim=-1)
                probabilities = log_probs.exp()
                policy_loss = -trajectory.reward * (
                    log_probs.gather(1, trajectory.targets[:, None]).mean()
                )
                entropy = -(probabilities * log_probs).sum(dim=-1).mean()
                anchor_kl = (
                    probabilities * (log_probs - trajectory.anchor_log_probs)
                ).sum(dim=-1).mean()
                token_weight = len(trajectory.targets) / total_tokens
                scaled_loss = (
                    policy_loss / len(cached)
                    - config.entropy_coefficient * token_weight * entropy
                    + config.anchor_kl_coefficient * token_weight * anchor_kl
                )
                if not bool(torch.isfinite(scaled_loss).item()):
                    raise FloatingPointError("non-finite readout-only objective")
                scaled_loss.backward()
                policy_value += float(policy_loss.detach().cpu()) / len(cached)
                entropy_value += float(entropy.detach().cpu()) * token_weight
                anchor_kl_value += float(anchor_kl.detach().cpu()) * token_weight
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                trainable, config.gradient_clip_norm, error_if_nonfinite=True
            )
            optimizer.step()
            if not all(bool(torch.isfinite(p).all().item()) for p in trainable):
                raise FloatingPointError("readout-only update produced non-finite parameters")
            losses.append(
                policy_value
                - config.entropy_coefficient * entropy_value
                + config.anchor_kl_coefficient * anchor_kl_value
            )
            gradient_norms.append(float(gradient_norm.detach().cpu()))
            final_policy_loss = policy_value
            final_entropy = entropy_value
            final_anchor_kl = anchor_kl_value
    except BaseException:
        policy.readout.load_state_dict(snapshot)
        raise

    return DailyUpdateStats(
        example_count=len(examples),
        update_steps=config.max_update_steps,
        losses=tuple(losses),
        gradient_norms=tuple(gradient_norms),
        final_policy_loss=final_policy_loss,
        final_entropy=final_entropy,
        final_anchor_kl=final_anchor_kl,
    )


def apply_daily_policy_update(
    policy: FlyTextPolicy,
    anchor_policy: FlyTextPolicy,
    *,
    vocabulary: FlyVocabulary,
    examples: Sequence[TrainingExample],
    config: DailyTrainingConfig,
) -> DailyUpdateStats:
    if not examples:
        raise ValueError("no unconsumed training examples")
    _assert_compatible_anchor(policy, anchor_policy)
    if config.training_mode == "readout_only":
        if any(not example.token_ids for example in examples):
            raise ValueError("training examples must contain at least one token")
        return _apply_readout_only_policy_update(
            policy, anchor_policy, vocabulary=vocabulary,
            examples=examples, config=config,
        )

    trainable = [parameter for parameter in policy.parameters() if parameter.requires_grad]
    if not trainable:
        raise ValueError("policy has no trainable text-interface parameters")
    if any(not example.token_ids for example in examples):
        raise ValueError("training examples must contain at least one token")

    # An individual text trajectory already creates an expensive autograd graph
    # through the frozen connectome. Keeping all trajectories live until one
    # backward() retained N entire graphs, exhausting Windows memory and causing
    # a native crash in PyTorch's c10.dll. Accumulate gradients per trajectory
    # and release each graph immediately after its own backward().
    example_count = len(examples)
    total_tokens = sum(len(example.token_ids) for example in examples)
    snapshot = _snapshot_state(policy)
    optimizer = torch.optim.AdamW(
        trainable,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    losses: list[float] = []
    gradient_norms: list[float] = []
    final_policy_loss: float | None = None
    final_entropy: float | None = None
    final_anchor_kl: float | None = None

    try:
        for _ in range(config.max_update_steps):
            optimizer.zero_grad(set_to_none=True)
            policy_loss_value = 0.0
            entropy_value = 0.0
            anchor_kl_value = 0.0

            for example in examples:
                objective = daily_policy_objective(
                    policy,
                    anchor_policy,
                    vocabulary=vocabulary,
                    examples=[example],
                    config=config,
                    verify_anchor=False,
                )
                # The original policy loss averages over examples, whereas
                # entropy and anchor KL average over *all tokens*. These
                # weights preserve the exact original objective and gradient
                # even when individual tweets have different lengths.
                token_weight = len(example.token_ids) / total_tokens
                scaled_loss = (
                    objective.policy_loss / example_count
                    - config.entropy_coefficient * token_weight * objective.entropy
                    + config.anchor_kl_coefficient * token_weight * objective.anchor_kl
                )
                if not bool(torch.isfinite(scaled_loss).item()):
                    raise FloatingPointError("daily policy objective is non-finite")
                scaled_loss.backward()
                policy_loss_value += float(objective.policy_loss.detach().cpu()) / example_count
                entropy_value += float(objective.entropy.detach().cpu()) * token_weight
                anchor_kl_value += float(objective.anchor_kl.detach().cpu()) * token_weight
                del scaled_loss, objective

            gradient_norm = torch.nn.utils.clip_grad_norm_(
                trainable,
                config.gradient_clip_norm,
                error_if_nonfinite=True,
            )
            optimizer.step()
            if not _all_trainable_parameters_finite(policy):
                raise FloatingPointError("daily update produced non-finite parameters")

            losses.append(
                policy_loss_value
                - config.entropy_coefficient * entropy_value
                + config.anchor_kl_coefficient * anchor_kl_value
            )
            gradient_norms.append(float(gradient_norm.detach().cpu()))
            final_policy_loss = policy_loss_value
            final_entropy = entropy_value
            final_anchor_kl = anchor_kl_value
    except BaseException:
        policy.load_state_dict(snapshot)
        raise

    if final_policy_loss is None or final_entropy is None or final_anchor_kl is None:
        raise RuntimeError("daily update completed without optimizer steps")
    return DailyUpdateStats(
        example_count=example_count,
        update_steps=config.max_update_steps,
        losses=tuple(losses),
        gradient_norms=tuple(gradient_norms),
        final_policy_loss=final_policy_loss,
        final_entropy=final_entropy,
        final_anchor_kl=final_anchor_kl,
    )


def run_daily_update(
    policy: FlyTextPolicy,
    anchor_policy: FlyTextPolicy,
    *,
    vocabulary: FlyVocabulary,
    rows: Sequence[MutableMapping[str, object]],
    config: DailyTrainingConfig,
    generation_config: GenerationConfig,
    checkpoint_id: str,
    batch_id: str,
    git_sha: str,
    base_seed: int,
    now: datetime,
    write_checkpoint: CheckpointWriter,
    append_rows: RowsAppender,
    generation_count: int = 10,
) -> DailyRunResult:
    """Run one logically atomic Fly Tweets learning/generation transaction.

    The model is rolled back on any downstream failure. Training source rows are not
    marked consumed until both the immutable checkpoint write and idempotent queue
    append have succeeded. A checkpoint created before a later failure may remain as
    an unreferenced immutable artifact; it is never treated as deployed by this call.
    The caller supplies the fly-selected generation count; this function does not
    impose a normal tweets-per-day cadence.
    """
    checkpoint = str(checkpoint_id).strip()
    batch = str(batch_id).strip()
    sha = str(git_sha).strip()
    if not checkpoint:
        raise ValueError("checkpoint_id is required")
    if not batch:
        raise ValueError("batch_id is required")
    if not sha:
        raise ValueError("git_sha is required")
    if isinstance(generation_count, bool) or not isinstance(generation_count, int) or generation_count <= 0:
        raise ValueError("generation_count must be a positive integer")
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("now must be timezone-aware")

    examples = training_examples_from_rows(rows, vocabulary=vocabulary)
    if not examples:
        raise ValueError("no unconsumed training examples")
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

    snapshot = _snapshot_state(policy)
    try:
        stats = apply_daily_policy_update(
            policy,
            anchor_policy,
            vocabulary=vocabulary,
            examples=examples,
            config=config,
        )
        manifest: dict[str, object] = {
            "schema_version": 1,
            "checkpoint_id": checkpoint,
            "batch_id": batch,
            "git_sha": sha,
            "example_count": stats.example_count,
            "update_steps": stats.update_steps,
            "losses": list(stats.losses),
            "gradient_norms": list(stats.gradient_norms),
            "final_policy_loss": stats.final_policy_loss,
            "final_entropy": stats.final_entropy,
            "final_anchor_kl": stats.final_anchor_kl,
            "consumed_keys": list(consumed_keys),
            "completed_at": now.isoformat(),
            "generated_count": generation_count,
            "llm_in_training_path": False,
            "recurrent_connectome_trainable": bool(policy.recurrent.requires_grad),
        }
        write_checkpoint(policy, checkpoint, manifest)

        generated = generate_batch(
            policy,
            vocabulary,
            base_seed=int(base_seed),
            config=generation_config,
            checkpoint_id=checkpoint,
            git_sha=sha,
            batch_id=batch,
            count=generation_count,
        )
        generated_rows = [
            generated_tweet_row(tweet, generated_at=now) for tweet in generated
        ]
        if len(generated_rows) != generation_count:
            raise RuntimeError("daily update generated an unexpected queue row count")
        append_rows(generated_rows)

        consumed_at = now.isoformat()
        for row in target_rows:
            row["training_consumed_at"] = consumed_at
    except BaseException:
        policy.load_state_dict(snapshot)
        raise

    return DailyRunResult(
        checkpoint_id=checkpoint,
        batch_id=batch,
        consumed_keys=consumed_keys,
        update_stats=stats,
        generated_rows=tuple(generated_rows),
    )
