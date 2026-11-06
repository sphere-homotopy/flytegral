from __future__ import annotations

import json
import math
import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import torch

from fly_window.text.generation import render_token_ids
from fly_window.text.policy import FlyTextPolicy
from fly_window.text.pretraining import teacher_forcing_loss
from fly_window.text.vocabulary import FlyVocabulary


@dataclass(frozen=True, slots=True)
class GateConfig:
    max_heldout_loss: float
    min_eos_rate: float
    max_repeat_run: int
    min_non_degenerate_rate: float
    min_median_chars: int
    max_median_chars: int
    probe_max_tokens: int
    temperature: float
    probe_count: int = 8
    replay_probe_count: int = 2

    def __post_init__(self) -> None:
        if self.max_heldout_loss < 0.0:
            raise ValueError("max_heldout_loss must be non-negative")
        if not 0.0 <= self.min_eos_rate <= 1.0:
            raise ValueError("min_eos_rate must lie in [0, 1]")
        if self.max_repeat_run < 1:
            raise ValueError("max_repeat_run must be at least one")
        if not 0.0 <= self.min_non_degenerate_rate <= 1.0:
            raise ValueError("min_non_degenerate_rate must lie in [0, 1]")
        if self.min_median_chars < 0:
            raise ValueError("min_median_chars must be non-negative")
        if self.max_median_chars < self.min_median_chars:
            raise ValueError("max_median_chars must be >= min_median_chars")
        if self.probe_max_tokens < 1:
            raise ValueError("probe_max_tokens must be positive")
        if not math.isfinite(self.temperature) or self.temperature <= 0.0:
            raise ValueError("temperature must be finite and positive")
        if self.probe_count < 1:
            raise ValueError("probe_count must be positive")
        if self.replay_probe_count < 1:
            raise ValueError("replay_probe_count must be positive")
        if self.replay_probe_count > self.probe_count:
            raise ValueError("replay_probe_count must not exceed probe_count")

    @classmethod
    def from_json(cls, path: Path) -> "GateConfig":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("gate config must be a JSON object")
        try:
            return cls(
                max_heldout_loss=float(payload["max_heldout_loss"]),
                min_eos_rate=float(payload["min_eos_rate"]),
                max_repeat_run=int(payload["max_repeat_run"]),
                min_non_degenerate_rate=float(payload["min_non_degenerate_rate"]),
                min_median_chars=int(payload["min_median_chars"]),
                max_median_chars=int(payload["max_median_chars"]),
                probe_max_tokens=int(payload["probe_max_tokens"]),
                temperature=float(payload["temperature"]),
                probe_count=int(payload.get("probe_count", 8)),
                replay_probe_count=int(payload.get("replay_probe_count", 2)),
            )
        except KeyError as error:
            raise ValueError(f"gate config missing {error.args[0]}") from error


@dataclass(frozen=True, slots=True)
class GateReport:
    passed: bool
    heldout_loss: float
    eos_rate: float
    non_degenerate_rate: float
    median_chars: float
    finite: bool
    replay_exact: bool
    probe_count: int
    replay_probe_count: int


@dataclass(frozen=True, slots=True)
class _Probe:
    token_ids: tuple[int, ...]
    terminated_with_eos: bool
    max_repeat_run: int
    rendered_chars: int
    finite: bool


def _max_repeat_run(token_ids: Sequence[int]) -> int:
    longest = 0
    current = 0
    previous: int | None = None
    for raw in token_ids:
        token_id = int(raw)
        if token_id == previous:
            current += 1
        else:
            previous = token_id
            current = 1
        longest = max(longest, current)
    return longest


def _probe_once(
    policy: FlyTextPolicy,
    vocabulary: FlyVocabulary,
    *,
    seed: int,
    config: GateConfig,
) -> _Probe:
    device = policy.input_gain.device
    bos_id = vocabulary.id_for("<BOS>")
    eos_id = vocabulary.id_for("<EOS>")
    state = policy.initial_state(batch_size=1)
    current = bos_id
    body: list[int] = []
    finite = bool(torch.isfinite(state).all().item())
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    terminated = False

    for _ in range(config.probe_max_tokens):
        if not finite:
            break
        token = torch.tensor([current], dtype=torch.long, device=device)
        output = policy.step(token, state)
        if not bool(torch.isfinite(output.logits).all().item()) or not bool(
            torch.isfinite(output.state).all().item()
        ):
            finite = False
            break

        probabilities = torch.softmax(output.logits / config.temperature, dim=-1).detach().cpu()[0]
        if not bool(torch.isfinite(probabilities).all().item()):
            finite = False
            break
        sampled = int(torch.multinomial(probabilities, 1, generator=generator).item())
        body.append(sampled)
        state = output.state
        if sampled == eos_id:
            terminated = True
            break
        current = sampled

    repeat_body = [token_id for token_id in body if token_id != eos_id]
    rendered = render_token_ids(vocabulary, body)
    return _Probe(
        token_ids=tuple(body),
        terminated_with_eos=terminated,
        max_repeat_run=_max_repeat_run(repeat_body),
        rendered_chars=len(rendered),
        finite=finite,
    )


def evaluate_launch_gate(
    policy: FlyTextPolicy,
    vocabulary: FlyVocabulary,
    *,
    heldout: Sequence[Sequence[int]],
    probe_seeds: Sequence[int],
    config: GateConfig,
) -> GateReport:
    if not heldout:
        raise ValueError("heldout sequences must not be empty")
    if not probe_seeds:
        raise ValueError("probe_seeds must not be empty")

    was_training = policy.training
    policy.eval()
    try:
        with torch.no_grad():
            heldout_tensor = teacher_forcing_loss(policy, heldout)
            heldout_loss = float(heldout_tensor.detach().cpu())
            heldout_finite = math.isfinite(heldout_loss)

            probes = [
                _probe_once(policy, vocabulary, seed=int(seed), config=config)
                for seed in probe_seeds
            ]
            replay_count = min(config.replay_probe_count, len(probes))
            replays = [
                _probe_once(policy, vocabulary, seed=int(seed), config=config)
                for seed in probe_seeds[:replay_count]
            ]
    finally:
        policy.train(was_training)

    probe_count = len(probes)
    eos_rate = sum(probe.terminated_with_eos for probe in probes) / probe_count
    non_degenerate_rate = (
        sum(probe.max_repeat_run <= config.max_repeat_run for probe in probes) / probe_count
    )
    median_chars = float(statistics.median(probe.rendered_chars for probe in probes))
    finite = heldout_finite and all(probe.finite for probe in probes)
    replay_exact = all(
        first.token_ids == second.token_ids
        and first.terminated_with_eos == second.terminated_with_eos
        and first.finite == second.finite
        for first, second in zip(probes[:replay_count], replays, strict=True)
    )

    passed = (
        finite
        and replay_exact
        and heldout_loss <= config.max_heldout_loss
        and eos_rate >= config.min_eos_rate
        and non_degenerate_rate >= config.min_non_degenerate_rate
        and config.min_median_chars <= median_chars <= config.max_median_chars
    )
    return GateReport(
        passed=passed,
        heldout_loss=heldout_loss,
        eos_rate=eos_rate,
        non_degenerate_rate=non_degenerate_rate,
        median_chars=median_chars,
        finite=finite,
        replay_exact=replay_exact,
        probe_count=probe_count,
        replay_probe_count=replay_count,
    )
