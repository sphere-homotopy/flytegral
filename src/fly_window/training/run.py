from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Protocol

import numpy as np
import torch
from torch import nn

from fly_window.env.config import EnvironmentConfig
from fly_window.env.core import WindowExitEnv
from fly_window.neural.graph import ConnectomeGraph
from fly_window.neural.policy import ConnectomePolicy
from fly_window.training.buffer import RolloutBuffer
from fly_window.training.ppo import ppo_loss


DEVELOPMENT_SEEDS = tuple(range(9000, 9032))
HELD_OUT_SEEDS = tuple(range(10000, 10100))


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    seed: int = 20260914
    parallel_envs: int = 16
    rollout_steps: int = 256
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    learning_rate: float = 3e-4
    ppo_epochs: int = 4
    minibatch_size: int = 512
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_environment_steps: int = 2_000_000
    checkpoint_every_steps: int = 50_000
    eval_every_steps: int = 50_000

    @classmethod
    def from_json(cls, path: Path) -> "TrainingConfig":
        return cls(**json.loads(path.read_text()))

    def __post_init__(self) -> None:
        for name in (
            "parallel_envs",
            "rollout_steps",
            "ppo_epochs",
            "minibatch_size",
            "max_environment_steps",
            "checkpoint_every_steps",
            "eval_every_steps",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if not 0.0 < self.gamma <= 1.0:
            raise ValueError("gamma must lie in (0, 1]")
        if not 0.0 <= self.gae_lambda <= 1.0:
            raise ValueError("gae_lambda must lie in [0, 1]")
        if not 0.0 < self.clip_ratio < 1.0:
            raise ValueError("clip_ratio must lie in (0, 1)")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")


@dataclass(frozen=True, slots=True)
class TrainingResult:
    run_dir: Path
    total_environment_steps: int


class VectorEnvironment(Protocol):
    env_count: int

    def reset(self, seeds: list[int] | tuple[int, ...]) -> np.ndarray: ...

    def step(
        self, actions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: ...


class WindowVectorEnv:
    """Deterministic vector adapter around :class:`WindowExitEnv`.

    Finished episodes are immediately reset with monotonically increasing seeds.
    The terminal transition still returns ``done=True`` while its next observation
    is the reset observation, which is the standard rollout convention used here.
    """

    def __init__(self, env_count: int, config: EnvironmentConfig | None = None):
        if env_count <= 0:
            raise ValueError("env_count must be positive")
        self.env_count = int(env_count)
        self.config = config or EnvironmentConfig()
        self.envs = [WindowExitEnv(self.config) for _ in range(self.env_count)]
        self._next_seed = 0

    def reset(self, seeds: list[int] | tuple[int, ...]) -> np.ndarray:
        if len(seeds) != self.env_count:
            raise ValueError("seed count must equal env_count")
        observations = []
        max_seed = -1
        for env, seed in zip(self.envs, seeds, strict=True):
            observation, _ = env.reset(int(seed))
            observations.append(observation)
            max_seed = max(max_seed, int(seed))
        self._next_seed = max_seed + 1
        return np.asarray(observations, dtype=np.float32)

    def step(
        self, actions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        action_array = np.asarray(actions, dtype=np.float32)
        if action_array.shape != (self.env_count, 2):
            raise ValueError(f"actions must have shape ({self.env_count}, 2)")

        observations: list[np.ndarray] = []
        rewards = np.empty(self.env_count, dtype=np.float32)
        dones = np.empty(self.env_count, dtype=np.bool_)
        successes = np.empty(self.env_count, dtype=np.bool_)

        for index, (env, action) in enumerate(zip(self.envs, action_array, strict=True)):
            result = env.step(action)
            done = result.terminated or result.truncated
            observation = result.observation
            if done:
                observation, _ = env.reset(self._next_seed)
                self._next_seed += 1
            observations.append(observation)
            rewards[index] = result.reward
            dones[index] = done
            successes[index] = result.success

        return np.asarray(observations, dtype=np.float32), rewards, dones, successes


class ValueHead(nn.Module):
    def __init__(self, visual_dim: int = 16):
        super().__init__()
        self.linear = nn.Linear(visual_dim + 1, 1)

    def forward(self, observation: torch.Tensor, output_activity: torch.Tensor) -> torch.Tensor:
        mean_output_activity = output_activity.mean(dim=1, keepdim=True)
        features = torch.cat((observation, mean_output_activity), dim=1)
        return self.linear(features)[:, 0]


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _run_id(graph_manifest_hash: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_hash = graph_manifest_hash[:8] if graph_manifest_hash else "unknown"
    return f"{timestamp}-{safe_hash}"


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _save_checkpoint(
    *,
    path: Path,
    policy: ConnectomePolicy,
    value_head: ValueHead,
    optimizer: torch.optim.Optimizer,
    total_environment_steps: int,
    config: TrainingConfig,
    graph_manifest_hash: str,
    git_sha: str,
) -> None:
    torch.save(
        {
            "policy_state": policy.state_dict(),
            "value_head_state": value_head.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "total_environment_steps": total_environment_steps,
            "python_rng_state": random.getstate(),
            "numpy_rng_state": np.random.get_state(),
            "torch_rng_state": torch.get_rng_state(),
            "config": asdict(config),
            "graph_manifest_hash": graph_manifest_hash,
            "git_sha": git_sha,
        },
        path,
    )


def _value_from_output(
    value_head: ValueHead,
    observation: torch.Tensor,
    policy_output,
    output_indices: torch.Tensor,
) -> torch.Tensor:
    return value_head(observation, policy_output.activity[:, output_indices])


def run_training(
    *,
    config: TrainingConfig,
    graph: ConnectomeGraph,
    recurrent: torch.Tensor,
    vector_env: VectorEnvironment,
    run_root: Path,
    graph_manifest_hash: str,
    git_sha: str,
    control_kind: str,
    control_seed: int | None = None,
    development_evaluator: Callable[[ConnectomePolicy, tuple[int, ...]], dict] | None = None,
) -> TrainingResult:
    if vector_env.env_count != config.parallel_envs:
        raise ValueError("vector_env.env_count must equal config.parallel_envs")
    if control_kind not in {"biological", "shuffled"}:
        raise ValueError("control_kind must be 'biological' or 'shuffled'")
    if control_kind == "shuffled" and control_seed is None:
        raise ValueError("shuffled training requires control_seed")
    if set(DEVELOPMENT_SEEDS) & set(HELD_OUT_SEEDS):
        raise RuntimeError("development and held-out seed sets must be disjoint")

    _seed_everything(config.seed)
    run_root = Path(run_root)
    run_root.mkdir(parents=True, exist_ok=True)
    run_dir = run_root / _run_id(graph_manifest_hash)
    suffix = 1
    while run_dir.exists():
        run_dir = run_root / f"{_run_id(graph_manifest_hash)}-{suffix}"
        suffix += 1
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True)

    config_payload = asdict(config)
    _write_json(run_dir / "config.json", config_payload)
    manifest = {
        "graph_manifest_hash": graph_manifest_hash,
        "git_sha": git_sha,
        "seed": config.seed,
        "control_kind": control_kind,
        "control_seed": control_seed,
        "development_seeds": list(DEVELOPMENT_SEEDS),
        "held_out_seeds_used_for_training": False,
    }
    _write_json(run_dir / "manifest.json", manifest)
    metrics_path = run_dir / "metrics.jsonl"
    metrics_path.write_text("")

    policy = ConnectomePolicy(graph, recurrent)
    value_head = ValueHead(visual_dim=policy.visual_dim)
    trainable_parameters = [parameter for parameter in policy.parameters() if parameter.requires_grad]
    trainable_parameters.extend(value_head.parameters())
    optimizer = torch.optim.Adam(trainable_parameters, lr=config.learning_rate)

    observation_np = vector_env.reset(
        [config.seed + index for index in range(config.parallel_envs)]
    )
    total_environment_steps = 0
    next_checkpoint = config.checkpoint_every_steps
    next_evaluation = config.eval_every_steps
    saved_steps: set[int] = set()

    while total_environment_steps < config.max_environment_steps:
        buffer = RolloutBuffer()
        rollout_successes = 0

        for _ in range(config.rollout_steps):
            if total_environment_steps >= config.max_environment_steps:
                break

            observation = torch.as_tensor(observation_np, dtype=torch.float32)
            with torch.no_grad():
                policy_output = policy(observation)
                distribution = policy.distribution(policy_output.mean_action)
                latent_action = distribution.sample()
                log_prob = distribution.log_prob(latent_action)
                value = _value_from_output(
                    value_head,
                    observation,
                    policy_output,
                    policy.output_indices,
                )
                environment_action = policy.action_from_latent(latent_action)

            next_observation_np, rewards_np, dones_np, successes_np = vector_env.step(
                environment_action.cpu().numpy()
            )
            buffer.add(
                observation=observation,
                latent_action=latent_action,
                log_prob=log_prob,
                reward=torch.as_tensor(rewards_np, dtype=torch.float32),
                done=torch.as_tensor(dones_np, dtype=torch.bool),
                value=value,
            )
            observation_np = next_observation_np
            rollout_successes += int(np.asarray(successes_np, dtype=np.int64).sum())
            total_environment_steps += config.parallel_envs

        with torch.no_grad():
            final_observation = torch.as_tensor(observation_np, dtype=torch.float32)
            final_output = policy(final_observation)
            last_value = _value_from_output(
                value_head,
                final_observation,
                final_output,
                policy.output_indices,
            )

        advantages, returns = buffer.compute_gae(
            last_value=last_value,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
        )
        observations = torch.stack(buffer.observations).reshape(-1, policy.visual_dim)
        latent_actions = torch.stack(buffer.latent_actions).reshape(-1, 2)
        old_log_probs = torch.stack(buffer.log_probs).reshape(-1)
        advantages_flat = advantages.reshape(-1)
        returns_flat = returns.reshape(-1)
        sample_count = observations.shape[0]

        last_losses = None
        for _ in range(config.ppo_epochs):
            permutation = torch.randperm(sample_count)
            for start in range(0, sample_count, config.minibatch_size):
                indices = permutation[start : start + config.minibatch_size]
                batch_observation = observations[indices]
                output = policy(batch_observation)
                distribution = policy.distribution(output.mean_action)
                new_log_prob = distribution.log_prob(latent_actions[indices])
                new_value = _value_from_output(
                    value_head,
                    batch_observation,
                    output,
                    policy.output_indices,
                )
                losses = ppo_loss(
                    new_log_prob=new_log_prob,
                    old_log_prob=old_log_probs[indices],
                    advantage=advantages_flat[indices],
                    new_value=new_value,
                    target_return=returns_flat[indices],
                    entropy=distribution.entropy(),
                    clip_ratio=config.clip_ratio,
                    value_coef=config.value_coef,
                    entropy_coef=config.entropy_coef,
                )
                optimizer.zero_grad(set_to_none=True)
                losses[0].backward()
                torch.nn.utils.clip_grad_norm_(trainable_parameters, max_norm=0.5)
                optimizer.step()
                last_losses = losses

        assert last_losses is not None
        metric = {
            "total_environment_steps": total_environment_steps,
            "rollout_successes": rollout_successes,
            "policy_loss": float(last_losses[1].detach()),
            "value_loss": float(last_losses[2].detach()),
            "entropy": float(last_losses[3].detach()),
        }

        if total_environment_steps >= next_evaluation:
            if development_evaluator is not None:
                evaluation = development_evaluator(policy, DEVELOPMENT_SEEDS)
                metric["development_evaluation"] = evaluation
            while next_evaluation <= total_environment_steps:
                next_evaluation += config.eval_every_steps

        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(metric, sort_keys=True) + "\n")

        if total_environment_steps >= next_checkpoint:
            checkpoint_path = checkpoint_dir / f"step-{total_environment_steps:09d}.pt"
            _save_checkpoint(
                path=checkpoint_path,
                policy=policy,
                value_head=value_head,
                optimizer=optimizer,
                total_environment_steps=total_environment_steps,
                config=config,
                graph_manifest_hash=graph_manifest_hash,
                git_sha=git_sha,
            )
            saved_steps.add(total_environment_steps)
            while next_checkpoint <= total_environment_steps:
                next_checkpoint += config.checkpoint_every_steps

    if total_environment_steps not in saved_steps:
        _save_checkpoint(
            path=checkpoint_dir / f"step-{total_environment_steps:09d}.pt",
            policy=policy,
            value_head=value_head,
            optimizer=optimizer,
            total_environment_steps=total_environment_steps,
            config=config,
            graph_manifest_hash=graph_manifest_hash,
            git_sha=git_sha,
        )

    return TrainingResult(run_dir=run_dir, total_environment_steps=total_environment_steps)
