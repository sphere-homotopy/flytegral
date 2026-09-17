from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from fly_window.neural.graph import ConnectomeGraph


@dataclass(frozen=True, slots=True)
class TextPolicyOutput:
    logits: torch.Tensor
    state: torch.Tensor
    activity: torch.Tensor


class FlyTextPolicy(nn.Module):
    def __init__(
        self,
        graph: ConnectomeGraph,
        recurrent: torch.Tensor,
        *,
        vocab_size: int,
        sensory_dim: int = 32,
        microsteps: int = 4,
        leak: float = 0.35,
    ) -> None:
        super().__init__()
        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if sensory_dim <= 0:
            raise ValueError("sensory_dim must be positive")
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

        self.vocab_size = int(vocab_size)
        self.sensory_dim = int(sensory_dim)
        self.microsteps = int(microsteps)
        self.leak = float(leak)
        self.node_count = graph.node_count

        self.token_embedding = nn.Embedding(self.vocab_size, self.sensory_dim)
        self.input_gain = nn.Parameter(torch.empty(input_indices.size, self.sensory_dim))
        self.input_bias = nn.Parameter(torch.zeros(input_indices.size))
        self.readout = nn.Linear(output_indices.size, self.vocab_size)

        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.05)
        nn.init.xavier_uniform_(self.input_gain)
        nn.init.xavier_uniform_(self.readout.weight)
        nn.init.zeros_(self.readout.bias)

        self.register_buffer("recurrent", recurrent.coalesce().detach())
        self.register_buffer("input_indices", torch.from_numpy(input_indices))
        self.register_buffer("output_indices", torch.from_numpy(output_indices))

    def initial_state(self, batch_size: int) -> torch.Tensor:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        return torch.zeros(
            (batch_size, self.node_count),
            dtype=self.input_gain.dtype,
            device=self.input_gain.device,
        )

    def step(self, token_ids: torch.Tensor, state: torch.Tensor) -> TextPolicyOutput:
        if token_ids.ndim != 1:
            raise ValueError("token_ids must have shape (batch,)")
        if token_ids.shape[0] != state.shape[0]:
            raise ValueError("token_ids and state batch sizes must match")
        if state.ndim != 2 or state.shape[1] != self.node_count:
            raise ValueError(
                f"state must have shape (batch, {self.node_count}), got {tuple(state.shape)}"
            )

        token_ids = token_ids.to(device=self.token_embedding.weight.device, dtype=torch.long)
        activity = state.to(device=self.input_gain.device, dtype=self.input_gain.dtype)
        sensory = self.token_embedding(token_ids)
        sensory_drive = sensory @ self.input_gain.transpose(0, 1) + self.input_bias

        for _ in range(self.microsteps):
            recurrent_drive = torch.sparse.mm(
                self.recurrent.transpose(0, 1), activity.transpose(0, 1)
            ).transpose(0, 1)
            injected = torch.zeros_like(activity)
            injected[:, self.input_indices] = sensory_drive
            proposal = torch.tanh(recurrent_drive + injected)
            activity = (1.0 - self.leak) * activity + self.leak * proposal

        logits = self.readout(activity[:, self.output_indices])
        return TextPolicyOutput(logits=logits, state=activity, activity=activity)

    @staticmethod
    def detach_state(state: torch.Tensor) -> torch.Tensor:
        return state.detach()
