from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import torch

from fly_window.neural.policy import ConnectomePolicy


@dataclass(frozen=True, slots=True)
class PolicyBenchmarkResult:
    batch_size: int
    iterations: int
    samples: int
    elapsed_seconds: float
    samples_per_second: float


@dataclass(frozen=True, slots=True)
class TrainingTimeEstimate:
    rollout_seconds: float
    ppo_seconds: float
    total_seconds: float


def _validate_benchmark_args(batch_size: int, iterations: int, warmup: int) -> None:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    if warmup < 0:
        raise ValueError("warmup must be non-negative")


def benchmark_policy_forward(
    policy: ConnectomePolicy,
    *,
    batch_size: int,
    iterations: int,
    warmup: int = 1,
) -> PolicyBenchmarkResult:
    """Measure inference throughput through the full recurrent connectome policy."""
    _validate_benchmark_args(batch_size, iterations, warmup)
    observation = torch.rand(batch_size, policy.visual_dim, dtype=torch.float32)
    previous_training = policy.training
    policy.eval()
    try:
        # torch.inference_mode() currently conflicts with sparse COO transpose;
        # no_grad keeps the exact inference computation while supporting sparse.mm.
        with torch.no_grad():
            for _ in range(warmup):
                policy(observation)
            started = perf_counter()
            for _ in range(iterations):
                policy(observation)
            elapsed = perf_counter() - started
    finally:
        policy.train(previous_training)

    samples = batch_size * iterations
    return PolicyBenchmarkResult(
        batch_size=batch_size,
        iterations=iterations,
        samples=samples,
        elapsed_seconds=elapsed,
        samples_per_second=samples / elapsed,
    )


def benchmark_policy_backward(
    policy: ConnectomePolicy,
    *,
    batch_size: int,
    iterations: int,
    warmup: int = 1,
) -> PolicyBenchmarkResult:
    """Measure forward+backward throughput without applying an optimizer step."""
    _validate_benchmark_args(batch_size, iterations, warmup)
    observation = torch.rand(batch_size, policy.visual_dim, dtype=torch.float32)
    previous_training = policy.training
    policy.train()

    def one_pass() -> None:
        policy.zero_grad(set_to_none=True)
        output = policy(observation)
        # The readout loss forces gradients through the recurrent microsteps back
        # into the visual encoder while leaving the frozen connectome untouched.
        loss = output.mean_action.square().mean()
        loss.backward()
        policy.zero_grad(set_to_none=True)

    try:
        for _ in range(warmup):
            one_pass()
        started = perf_counter()
        for _ in range(iterations):
            one_pass()
        elapsed = perf_counter() - started
    finally:
        policy.zero_grad(set_to_none=True)
        policy.train(previous_training)

    samples = batch_size * iterations
    return PolicyBenchmarkResult(
        batch_size=batch_size,
        iterations=iterations,
        samples=samples,
        elapsed_seconds=elapsed,
        samples_per_second=samples / elapsed,
    )


def estimate_training_seconds(
    *,
    max_environment_steps: int,
    ppo_epochs: int,
    rollout_samples_per_second: float,
    training_samples_per_second: float,
) -> TrainingTimeEstimate:
    if max_environment_steps <= 0:
        raise ValueError("max_environment_steps must be positive")
    if ppo_epochs <= 0:
        raise ValueError("ppo_epochs must be positive")
    if rollout_samples_per_second <= 0 or training_samples_per_second <= 0:
        raise ValueError("throughputs must be positive")

    rollout_seconds = max_environment_steps / rollout_samples_per_second
    ppo_seconds = max_environment_steps * ppo_epochs / training_samples_per_second
    return TrainingTimeEstimate(
        rollout_seconds=rollout_seconds,
        ppo_seconds=ppo_seconds,
        total_seconds=rollout_seconds + ppo_seconds,
    )
