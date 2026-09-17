from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.nn import functional as F

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
) -> DailyPolicyObjective:
    if not examples:
        raise ValueError("no unconsumed training examples")
    if policy.vocab_size != len(vocabulary):
        raise ValueError("policy vocabulary size does not match vocabulary")
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

    trainable = [parameter for parameter in policy.parameters() if parameter.requires_grad]
    if not trainable:
        raise ValueError("policy has no trainable text-interface parameters")

    snapshot = _snapshot_state(policy)
    optimizer = torch.optim.AdamW(
        trainable,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    losses: list[float] = []
    gradient_norms: list[float] = []
    final_objective: DailyPolicyObjective | None = None

    try:
        for _ in range(config.max_update_steps):
            optimizer.zero_grad(set_to_none=True)
            objective = daily_policy_objective(
                policy,
                anchor_policy,
                vocabulary=vocabulary,
                examples=examples,
                config=config,
            )
            if not bool(torch.isfinite(objective.loss).item()):
                raise FloatingPointError("daily policy objective is non-finite")
            objective.loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                trainable,
                config.gradient_clip_norm,
                error_if_nonfinite=True,
            )
            optimizer.step()
            if not _all_trainable_parameters_finite(policy):
                raise FloatingPointError("daily update produced non-finite parameters")

            losses.append(float(objective.loss.detach().cpu()))
            gradient_norms.append(float(gradient_norm.detach().cpu()))
            final_objective = objective
    except BaseException:
        policy.load_state_dict(snapshot)
        raise

    if final_objective is None:
        raise RuntimeError("daily update completed without optimizer steps")
    return DailyUpdateStats(
        example_count=len(examples),
        update_steps=config.max_update_steps,
        losses=tuple(losses),
        gradient_norms=tuple(gradient_norms),
        final_policy_loss=float(final_objective.policy_loss.detach().cpu()),
        final_entropy=float(final_objective.entropy.detach().cpu()),
        final_anchor_kl=float(final_objective.anchor_kl.detach().cpu()),
    )
