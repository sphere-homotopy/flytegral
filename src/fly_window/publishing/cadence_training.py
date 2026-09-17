from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.nn import functional as F

from fly_window.publishing.cadence import FlyCadencePolicy


@dataclass(frozen=True, slots=True)
class CadenceTrainingConfig:
    learning_rate: float
    weight_decay: float
    entropy_coefficient: float
    gradient_clip_norm: float
    max_update_steps: int

    def __post_init__(self) -> None:
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be finite and positive")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0.0:
            raise ValueError("weight_decay must be finite and non-negative")
        if not math.isfinite(self.entropy_coefficient) or self.entropy_coefficient < 0.0:
            raise ValueError("entropy_coefficient must be finite and non-negative")
        if not math.isfinite(self.gradient_clip_norm) or self.gradient_clip_norm <= 0.0:
            raise ValueError("gradient_clip_norm must be finite and positive")
        if self.max_update_steps <= 0:
            raise ValueError("max_update_steps must be positive")

    @classmethod
    def from_json(cls, path: Path) -> "CadenceTrainingConfig":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("cadence training config must be a JSON object")
        try:
            return cls(
                learning_rate=float(payload["learning_rate"]),
                weight_decay=float(payload.get("weight_decay", 0.0)),
                entropy_coefficient=float(payload["entropy_coefficient"]),
                gradient_clip_norm=float(payload["gradient_clip_norm"]),
                max_update_steps=int(payload["max_update_steps"]),
            )
        except KeyError as error:
            raise ValueError(f"cadence training config missing {error.args[0]}") from error


@dataclass(frozen=True, slots=True)
class CadenceTrainingExample:
    idempotency_key: str
    context: tuple[float, float, float, float]
    action: int
    reward: float


@dataclass(frozen=True, slots=True)
class CadenceUpdateStats:
    example_count: int
    update_steps: int
    losses: tuple[float, ...]
    gradient_norms: tuple[float, ...]
    final_policy_loss: float
    final_entropy: float


def _parse_context(value: object) -> tuple[float, float, float, float]:
    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise ValueError("cadence_context must not be empty")
        try:
            value = json.loads(text)
        except json.JSONDecodeError as error:
            raise ValueError("cadence_context must be valid JSON") from error
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError("cadence_context must contain exactly four values")
    context = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in context):
        raise ValueError("cadence_context must contain finite values")
    return context  # type: ignore[return-value]


def cadence_training_examples_from_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    action_count: int,
) -> list[CadenceTrainingExample]:
    if action_count <= 1:
        raise ValueError("action_count must include STOP and at least one wait action")
    examples: list[CadenceTrainingExample] = []
    seen: set[str] = set()
    for row in rows:
        if str(row.get("training_consumed_at", "") or "").strip():
            continue
        cadence_reward = row.get("cadence_reward", "")
        reward_value = cadence_reward if cadence_reward not in (None, "") else row.get("reward", "")
        if reward_value in (None, ""):
            continue
        cadence_context = row.get("cadence_context", "")
        cadence_action = row.get("cadence_action", "")
        if cadence_context in (None, "") or cadence_action in (None, ""):
            continue

        key = str(row.get("idempotency_key", "") or "").strip()
        if not key:
            raise ValueError("idempotency_key is required for cadence training")
        if key in seen:
            raise ValueError(f"duplicate idempotency_key: {key}")
        seen.add(key)

        reward = float(reward_value)
        if not math.isfinite(reward):
            raise ValueError(f"non-finite reward for {key}")
        action = int(cadence_action)
        if action <= 0 or action >= action_count:
            raise ValueError(f"cadence_action outside wait-action range for {key}")
        examples.append(
            CadenceTrainingExample(
                idempotency_key=key,
                context=_parse_context(cadence_context),
                action=action,
                reward=reward,
            )
        )
    return examples


def _all_trainable_parameters_finite(policy: FlyCadencePolicy) -> bool:
    return all(
        bool(torch.isfinite(parameter).all().item())
        for parameter in policy.parameters()
        if parameter.requires_grad
    )


def _objective(
    policy: FlyCadencePolicy,
    examples: Sequence[CadenceTrainingExample],
    config: CadenceTrainingConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = policy.input_gain.device
    policy_terms: list[torch.Tensor] = []
    entropies: list[torch.Tensor] = []
    for example in examples:
        context = torch.tensor([example.context], dtype=policy.input_gain.dtype, device=device)
        state = policy.initial_state(batch_size=1)
        output = policy.step(context, state)
        if not bool(torch.isfinite(output.logits).all().item()):
            raise FloatingPointError("non-finite cadence logits")
        log_probs = F.log_softmax(output.logits, dim=-1)
        probabilities = log_probs.exp()
        policy_terms.append(-float(example.reward) * log_probs[0, example.action])
        entropies.append(-(probabilities * log_probs).sum(dim=-1).mean())
    policy_loss = torch.stack(policy_terms).mean()
    entropy = torch.stack(entropies).mean()
    loss = policy_loss - config.entropy_coefficient * entropy
    return loss, policy_loss, entropy


def apply_cadence_policy_update(
    policy: FlyCadencePolicy,
    *,
    examples: Sequence[CadenceTrainingExample],
    config: CadenceTrainingConfig,
) -> CadenceUpdateStats:
    if not examples:
        raise ValueError("no cadence training examples")
    if policy.recurrent.requires_grad:
        raise ValueError("recurrent connectome must remain frozen")
    trainable = [parameter for parameter in policy.parameters() if parameter.requires_grad]
    if not trainable:
        raise ValueError("cadence policy has no trainable parameters")

    snapshot = {name: tensor.detach().clone() for name, tensor in policy.state_dict().items()}
    optimizer = torch.optim.AdamW(
        trainable,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    losses: list[float] = []
    gradient_norms: list[float] = []
    final_policy_loss: torch.Tensor | None = None
    final_entropy: torch.Tensor | None = None
    try:
        for _ in range(config.max_update_steps):
            optimizer.zero_grad(set_to_none=True)
            loss, policy_loss, entropy = _objective(policy, examples, config)
            if not bool(torch.isfinite(loss).item()):
                raise FloatingPointError("cadence objective is non-finite")
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                trainable,
                config.gradient_clip_norm,
                error_if_nonfinite=True,
            )
            optimizer.step()
            if not _all_trainable_parameters_finite(policy):
                raise FloatingPointError("cadence update produced non-finite parameters")
            losses.append(float(loss.detach().cpu()))
            gradient_norms.append(float(gradient_norm.detach().cpu()))
            final_policy_loss = policy_loss
            final_entropy = entropy
    except BaseException:
        policy.load_state_dict(snapshot)
        raise

    if final_policy_loss is None or final_entropy is None:
        raise RuntimeError("cadence update completed without optimizer steps")
    return CadenceUpdateStats(
        example_count=len(examples),
        update_steps=config.max_update_steps,
        losses=tuple(losses),
        gradient_norms=tuple(gradient_norms),
        final_policy_loss=float(final_policy_loss.detach().cpu()),
        final_entropy=float(final_entropy.detach().cpu()),
    )
