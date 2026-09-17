from __future__ import annotations

import json
import random
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import torch
from torch.nn import functional as F

from fly_window.text.corpus import CorpusTweet, prepare_corpus_tweet
from fly_window.text.policy import FlyTextPolicy
from fly_window.text.vocabulary import FlyVocabulary


@dataclass(frozen=True, slots=True)
class TextTrainingConfig:
    seed: int
    curriculum_examples: int
    stage_a_epochs: int
    stage_b_epochs: int
    stage_c_epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    gradient_clip_norm: float
    max_oov_fraction: float
    stage_c_teacher_probability: float

    def __post_init__(self) -> None:
        if self.curriculum_examples <= 0:
            raise ValueError("curriculum_examples must be positive")
        if min(self.stage_a_epochs, self.stage_b_epochs, self.stage_c_epochs) < 0:
            raise ValueError("training epoch counts must be non-negative")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if self.weight_decay < 0.0:
            raise ValueError("weight_decay must be non-negative")
        if self.gradient_clip_norm <= 0.0:
            raise ValueError("gradient_clip_norm must be positive")
        if not 0.0 <= self.max_oov_fraction <= 1.0:
            raise ValueError("max_oov_fraction must lie in [0, 1]")
        if not 0.0 <= self.stage_c_teacher_probability <= 1.0:
            raise ValueError("stage_c_teacher_probability must lie in [0, 1]")

    @classmethod
    def from_json(cls, path: Path) -> "TextTrainingConfig":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("text training config must be a JSON object")
        try:
            return cls(
                seed=int(payload["seed"]),
                curriculum_examples=int(payload["curriculum_examples"]),
                stage_a_epochs=int(payload["stage_a_epochs"]),
                stage_b_epochs=int(payload["stage_b_epochs"]),
                stage_c_epochs=int(payload["stage_c_epochs"]),
                batch_size=int(payload["batch_size"]),
                learning_rate=float(payload["learning_rate"]),
                weight_decay=float(payload["weight_decay"]),
                gradient_clip_norm=float(payload["gradient_clip_norm"]),
                max_oov_fraction=float(payload["max_oov_fraction"]),
                stage_c_teacher_probability=float(payload["stage_c_teacher_probability"]),
            )
        except KeyError as error:
            raise ValueError(f"text training config missing {error.args[0]}") from error


def choose_next_input(
    *,
    teacher_token: int,
    sampled_token: int,
    teacher_probability: float,
    draw: float,
) -> int:
    """Pure scheduled-sampling decision used by Stage C and its deterministic tests."""
    if not 0.0 <= teacher_probability <= 1.0:
        raise ValueError("teacher_probability must lie in [0, 1]")
    if not 0.0 <= draw < 1.0:
        raise ValueError("draw must lie in [0, 1)")
    return int(teacher_token if draw < teacher_probability else sampled_token)


def _canonical_status_url(tweet_url: str) -> str | None:
    try:
        parsed = urlparse(tweet_url)
    except ValueError:
        return None

    parts = [part for part in parsed.path.split("/") if part]
    if len(parts) < 3 or parts[1].lower() != "status" or not parts[2].isdigit():
        return None
    return f"https://x.com/{parts[0]}/status/{parts[2]}"


def load_cook_sequences(
    path: Path,
    vocabulary: FlyVocabulary,
    *,
    max_oov_fraction: float = 0.35,
) -> list[tuple[int, ...]]:
    """Load a scraped Cook JSONL corpus as deduplicated teacher-forcing sequences."""
    sequences: list[tuple[int, ...]] = []
    seen_statuses: set[str] = set()

    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSONL at line {line_number}: {error.msg}") from error
            if not isinstance(payload, dict):
                raise ValueError(f"invalid JSONL at line {line_number}: expected object")

            try:
                row = CorpusTweet(
                    account=str(payload["account"]),
                    tweet_url=str(payload["tweet_url"]),
                    tweet_datetime=str(payload.get("tweet_datetime", "")),
                    text=str(payload["text"]),
                    is_reply=bool(payload.get("is_reply", False)),
                    is_repost=bool(payload.get("is_repost", False)),
                )
            except KeyError as error:
                raise ValueError(
                    f"invalid JSONL at line {line_number}: missing {error.args[0]}"
                ) from error

            canonical_url = _canonical_status_url(row.tweet_url)
            if canonical_url is None or canonical_url in seen_statuses:
                continue

            prepared = prepare_corpus_tweet(
                row,
                vocabulary,
                max_oov_fraction=max_oov_fraction,
            )
            if prepared is None:
                continue

            seen_statuses.add(canonical_url)
            sequences.append(prepared.token_ids)

    return sequences


def _validated_sequence_batch(
    sequences: Sequence[Sequence[int]],
) -> tuple[list[tuple[int, ...]], int]:
    if not sequences:
        raise ValueError("sequences must not be empty")
    normalized = [tuple(int(token) for token in sequence) for sequence in sequences]
    if any(len(sequence) < 2 for sequence in normalized):
        raise ValueError("each training sequence must contain at least two tokens")
    return normalized, max(len(sequence) - 1 for sequence in normalized)


def teacher_forcing_loss(
    policy: FlyTextPolicy,
    sequences: Sequence[Sequence[int]],
) -> torch.Tensor:
    """Token-weighted next-token CE using one recurrent pass per batch timestep."""
    normalized, max_steps = _validated_sequence_batch(sequences)
    device = policy.input_gain.device
    batch_size = len(normalized)
    state = policy.initial_state(batch_size=batch_size)
    loss_sum = torch.zeros((), dtype=policy.input_gain.dtype, device=device)
    target_count = 0

    for step in range(max_steps):
        active = torch.tensor(
            [step < len(sequence) - 1 for sequence in normalized],
            dtype=torch.bool,
            device=device,
        )
        input_ids = torch.tensor(
            [sequence[step] if step < len(sequence) - 1 else 0 for sequence in normalized],
            dtype=torch.long,
            device=device,
        )
        targets = torch.tensor(
            [sequence[step + 1] if step < len(sequence) - 1 else 0 for sequence in normalized],
            dtype=torch.long,
            device=device,
        )
        output = policy.step(input_ids, state)
        loss_sum = loss_sum + F.cross_entropy(
            output.logits[active], targets[active], reduction="sum"
        )
        target_count += int(active.sum().item())
        state = output.state

    if target_count == 0:
        raise ValueError("sequences did not contain any prediction targets")
    return loss_sum / target_count


def scheduled_sampling_loss(
    policy: FlyTextPolicy,
    sequences: Sequence[Sequence[int]],
    *,
    teacher_probability: float,
    seed: int,
) -> torch.Tensor:
    """Batched next-token CE with deterministic sampling of the fly's own prefixes."""
    if not 0.0 <= teacher_probability <= 1.0:
        raise ValueError("teacher_probability must lie in [0, 1]")
    normalized, max_steps = _validated_sequence_batch(sequences)

    chooser = random.Random(seed)
    sampler = torch.Generator(device="cpu")
    sampler.manual_seed(int(seed))
    device = policy.input_gain.device
    batch_size = len(normalized)
    state = policy.initial_state(batch_size=batch_size)
    input_ids = torch.tensor(
        [sequence[0] for sequence in normalized], dtype=torch.long, device=device
    )
    loss_sum = torch.zeros((), dtype=policy.input_gain.dtype, device=device)
    target_count = 0

    for step in range(max_steps):
        active_flags = [step < len(sequence) - 1 for sequence in normalized]
        active = torch.tensor(active_flags, dtype=torch.bool, device=device)
        targets = torch.tensor(
            [sequence[step + 1] if is_active else 0 for sequence, is_active in zip(normalized, active_flags, strict=True)],
            dtype=torch.long,
            device=device,
        )
        output = policy.step(input_ids, state)
        loss_sum = loss_sum + F.cross_entropy(
            output.logits[active], targets[active], reduction="sum"
        )
        target_count += int(active.sum().item())
        state = output.state

        if step + 1 >= max_steps:
            continue

        probabilities = torch.softmax(output.logits.detach(), dim=-1).cpu()
        sampled_ids = torch.multinomial(probabilities, 1, generator=sampler).squeeze(1).tolist()
        next_inputs: list[int] = []
        for index, sequence in enumerate(normalized):
            has_next_prediction = step + 1 < len(sequence) - 1
            if not has_next_prediction:
                next_inputs.append(0)
                continue
            next_inputs.append(
                choose_next_input(
                    teacher_token=sequence[step + 1],
                    sampled_token=int(sampled_ids[index]),
                    teacher_probability=teacher_probability,
                    draw=chooser.random(),
                )
            )
        input_ids = torch.tensor(next_inputs, dtype=torch.long, device=device)

    if target_count == 0:
        raise ValueError("sequences did not contain any prediction targets")
    return loss_sum / target_count
