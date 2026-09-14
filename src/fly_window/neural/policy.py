from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.distributions import Independent, Normal

from fly_window.neural.graph import ConnectomeGraph


@dataclass(frozen=True, slots=True)
class PolicyOutput:
    mean_action: torch.Tensor
    activity: torch.Tensor


class ConnectomePolicy(nn.Module):
    def __init__(
        self,
        graph: ConnectomeGraph,
        recurrent: torch.Tensor,
        visual_dim: int = 16,
        microsteps: int = 4,
        leak: float = 0.35,
    ) -> None:
        super().__init__()
        if visual_dim <= 0:
            raise ValueError("visual_dim must be positive")
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

        self.visual_dim = int(visual_dim)
        self.microsteps = int(microsteps)
        self.leak = float(leak)
        self.node_count = graph.node_count

        self.input_gain = nn.Parameter(torch.empty(input_indices.size, self.visual_dim))
        self.input_bias = nn.Parameter(torch.zeros(input_indices.size))
        self.readout = nn.Linear(output_indices.size, 2)
        self.log_std = nn.Parameter(torch.zeros(2))

        nn.init.xavier_uniform_(self.input_gain)
        nn.init.xavier_uniform_(self.readout.weight)
        nn.init.zeros_(self.readout.bias)

        self.register_buffer("recurrent", recurrent.coalesce().detach())
        self.register_buffer("input_indices", torch.from_numpy(input_indices))
        self.register_buffer("output_indices", torch.from_numpy(output_indices))

    def forward(self, observation: torch.Tensor) -> PolicyOutput:
        if observation.ndim != 2 or observation.shape[1] != self.visual_dim:
            raise ValueError(
                f"observation must have shape (batch, {self.visual_dim}), got {tuple(observation.shape)}"
            )

        observation = observation.to(dtype=self.input_gain.dtype)
        activity = torch.zeros(
            (observation.shape[0], self.node_count),
            dtype=observation.dtype,
            device=observation.device,
        )
        sensory_drive = observation @ self.input_gain.transpose(0, 1) + self.input_bias

        for _ in range(self.microsteps):
            recurrent_drive = torch.sparse.mm(
                self.recurrent.transpose(0, 1), activity.transpose(0, 1)
            ).transpose(0, 1)
            injected = torch.zeros_like(activity)
            injected[:, self.input_indices] = sensory_drive
            proposal = torch.tanh(recurrent_drive + injected)
            activity = (1.0 - self.leak) * activity + self.leak * proposal

        mean_action = self.readout(activity[:, self.output_indices])
        return PolicyOutput(mean_action=mean_action, activity=activity)

    def distribution(self, mean_action: torch.Tensor) -> Independent:
        if mean_action.ndim != 2 or mean_action.shape[1] != 2:
            raise ValueError("mean_action must have shape (batch, 2)")
        std = self.log_std.exp().expand_as(mean_action)
        return Independent(Normal(mean_action, std), 1)

    @staticmethod
    def action_from_latent(latent_action: torch.Tensor) -> torch.Tensor:
        if latent_action.shape[-1] != 2:
            raise ValueError("latent_action must have final dimension 2")
        yaw = torch.tanh(latent_action[..., 0])
        thrust = torch.sigmoid(latent_action[..., 1])
        return torch.stack((yaw, thrust), dim=-1)
