from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import numpy as np
import torch
from torch import nn

from fly_window.neural.graph import ConnectomeGraph


@dataclass(frozen=True, slots=True)
class CadencePolicyOutput:
    logits: torch.Tensor
    state: torch.Tensor
    activity: torch.Tensor


@dataclass(frozen=True, slots=True)
class CadenceDecision:
    publish_at: datetime
    context: tuple[float, float, float, float]
    action: int


class FlyCadencePolicy(nn.Module):
    """MaleCNS policy head for POST/WAIT cadence decisions.

    Action 0 is STOP for the current UTC-day-sized generation segment. Actions
    1..N mean "wait this many minutes, then publish". The recurrent MaleCNS
    connectivity is frozen; only the sensory interface and cadence readout train.
    """

    def __init__(
        self,
        graph: ConnectomeGraph,
        recurrent: torch.Tensor,
        *,
        wait_minutes: tuple[int, ...],
        context_dim: int = 4,
        microsteps: int = 4,
        leak: float = 0.35,
    ) -> None:
        super().__init__()
        if not wait_minutes:
            raise ValueError("wait_minutes must not be empty")
        if any(isinstance(value, bool) or int(value) != value or value <= 0 for value in wait_minutes):
            raise ValueError("wait_minutes must contain positive integers")
        if tuple(sorted(set(wait_minutes))) != tuple(wait_minutes):
            raise ValueError("wait_minutes must be strictly increasing and unique")
        if context_dim != 4:
            raise ValueError("Fly cadence v1 requires four time-context features")
        if microsteps <= 0:
            raise ValueError("microsteps must be positive")
        if not 0.0 < leak <= 1.0:
            raise ValueError("leak must lie in (0, 1]")
        if tuple(recurrent.shape) != (graph.node_count, graph.node_count):
            raise ValueError("recurrent matrix shape must match graph node count")
        if recurrent.layout != torch.sparse_coo:
            raise ValueError("recurrent matrix must be sparse COO")

        input_indices = np.flatnonzero(graph.input_mask).astype(np.int64, copy=False)
        output_indices = np.flatnonzero(graph.output_mask).astype(np.int64, copy=False)
        if input_indices.size == 0:
            raise ValueError("graph must contain at least one input node")
        if output_indices.size == 0:
            raise ValueError("graph must contain at least one output node")

        self.wait_minutes = tuple(int(value) for value in wait_minutes)
        self.context_dim = int(context_dim)
        self.microsteps = int(microsteps)
        self.leak = float(leak)
        self.node_count = int(graph.node_count)
        self.input_gain = nn.Parameter(torch.empty(input_indices.size, self.context_dim))
        self.input_bias = nn.Parameter(torch.zeros(input_indices.size))
        self.action_readout = nn.Linear(output_indices.size, 1 + len(self.wait_minutes))

        nn.init.xavier_uniform_(self.input_gain)
        nn.init.xavier_uniform_(self.action_readout.weight)
        nn.init.zeros_(self.action_readout.bias)

        self.register_buffer("recurrent", recurrent.coalesce().detach())
        self.register_buffer("input_indices", torch.from_numpy(input_indices))
        self.register_buffer("output_indices", torch.from_numpy(output_indices))

    @property
    def action_count(self) -> int:
        return 1 + len(self.wait_minutes)

    def initial_state(self, batch_size: int = 1) -> torch.Tensor:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        return torch.zeros(
            (batch_size, self.node_count),
            dtype=self.input_gain.dtype,
            device=self.input_gain.device,
        )

    def step(self, context: torch.Tensor, state: torch.Tensor) -> CadencePolicyOutput:
        if context.ndim != 2 or context.shape[1] != self.context_dim:
            raise ValueError(
                f"context must have shape (batch, {self.context_dim}), got {tuple(context.shape)}"
            )
        if state.ndim != 2 or state.shape[1] != self.node_count:
            raise ValueError(
                f"state must have shape (batch, {self.node_count}), got {tuple(state.shape)}"
            )
        if context.shape[0] != state.shape[0]:
            raise ValueError("context and state batch sizes must match")

        context = context.to(device=self.input_gain.device, dtype=self.input_gain.dtype)
        activity = state.to(device=self.input_gain.device, dtype=self.input_gain.dtype)
        sensory_drive = context @ self.input_gain.transpose(0, 1) + self.input_bias
        for _ in range(self.microsteps):
            recurrent_drive = torch.sparse.mm(
                self.recurrent.transpose(0, 1), activity.transpose(0, 1)
            ).transpose(0, 1)
            injected = torch.zeros_like(activity)
            injected[:, self.input_indices] = sensory_drive
            proposal = torch.tanh(recurrent_drive + injected)
            activity = (1.0 - self.leak) * activity + self.leak * proposal

        logits = self.action_readout(activity[:, self.output_indices])
        return CadencePolicyOutput(logits=logits, state=activity, activity=activity)


def _require_aware(value: datetime, name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")
    return value.astimezone(UTC)


def _time_context(
    cursor: datetime, segment_start: datetime, segment_end: datetime
) -> tuple[float, float, float, float]:
    segment_seconds = (segment_end - segment_start).total_seconds()
    if segment_seconds <= 0.0:
        raise ValueError("generation segment must be positive")
    elapsed = (cursor - segment_start).total_seconds()
    progress = min(1.0, max(0.0, elapsed / segment_seconds))
    remaining = 1.0 - progress
    seconds_of_day = (
        cursor.hour * 3600
        + cursor.minute * 60
        + cursor.second
        + cursor.microsecond / 1_000_000
    )
    phase = 2.0 * math.pi * seconds_of_day / 86_400.0
    return (math.sin(phase), math.cos(phase), progress, remaining)


def sample_publish_decisions(
    policy: FlyCadencePolicy,
    *,
    start: datetime,
    end: datetime,
    seed: int,
    temperature: float = 1.0,
    max_posts_per_24h: int = 30,
) -> tuple[CadenceDecision, ...]:
    """Sample fly-owned publication decisions with training provenance."""
    start_utc = _require_aware(start, "start")
    end_utc = _require_aware(end, "end")
    if end_utc <= start_utc:
        raise ValueError("end must be later than start")
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be finite and positive")
    if max_posts_per_24h <= 0:
        raise ValueError("max_posts_per_24h must be positive")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    decisions: list[CadenceDecision] = []
    was_training = policy.training
    policy.eval()
    try:
        segment_start = start_utc
        with torch.no_grad():
            while segment_start < end_utc:
                segment_end = min(segment_start + timedelta(hours=24), end_utc)
                cursor = segment_start
                state = policy.initial_state(batch_size=1)
                segment_posts = 0

                while segment_posts < max_posts_per_24h:
                    context_values = _time_context(cursor, segment_start, segment_end)
                    context = torch.tensor(
                        [context_values],
                        dtype=policy.input_gain.dtype,
                        device=policy.input_gain.device,
                    )
                    output = policy.step(context, state)
                    if not bool(torch.isfinite(output.logits).all().item()) or not bool(
                        torch.isfinite(output.state).all().item()
                    ):
                        raise FloatingPointError("non-finite fly cadence state or logits")
                    probabilities = torch.softmax(
                        output.logits / temperature, dim=-1
                    ).detach().cpu()[0]
                    action = int(torch.multinomial(probabilities, 1, generator=generator).item())
                    state = output.state

                    if action == 0:
                        break
                    wait_minutes = policy.wait_minutes[action - 1]
                    candidate = cursor + timedelta(minutes=wait_minutes)
                    if candidate >= segment_end:
                        break
                    decisions.append(
                        CadenceDecision(
                            publish_at=candidate,
                            context=context_values,
                            action=action,
                        )
                    )
                    cursor = candidate
                    segment_posts += 1

                segment_start = segment_end
    finally:
        policy.train(was_training)

    return tuple(decisions)


def sample_publish_times(
    policy: FlyCadencePolicy,
    *,
    start: datetime,
    end: datetime,
    seed: int,
    temperature: float = 1.0,
    max_posts_per_24h: int = 30,
) -> tuple[datetime, ...]:
    """Backward-compatible projection of cadence decisions to absolute times."""
    return tuple(
        decision.publish_at
        for decision in sample_publish_decisions(
            policy,
            start=start,
            end=end,
            seed=seed,
            temperature=temperature,
            max_posts_per_24h=max_posts_per_24h,
        )
    )
