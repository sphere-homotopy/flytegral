from __future__ import annotations

import argparse
import ctypes
import json
import math
import os
import sys
from dataclasses import asdict
from pathlib import Path

import torch

from fly_window.neural.benchmark import (
    benchmark_policy_backward,
    benchmark_policy_forward,
    estimate_training_seconds,
)
from fly_window.neural.graph import load_connectome_graph, to_sparse_recurrent
from fly_window.neural.policy import ConnectomePolicy


def _peak_rss_mb() -> float:
    if os.name == "nt":
        from ctypes import wintypes

        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        handle = ctypes.windll.kernel32.GetCurrentProcess()
        ok = ctypes.windll.psapi.GetProcessMemoryInfo(
            handle,
            ctypes.byref(counters),
            counters.cb,
        )
        if not ok:
            raise OSError("GetProcessMemoryInfo failed")
        return counters.PeakWorkingSetSize / (1024 * 1024)

    import resource

    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return float(usage) / (1024 * 1024)
    return float(usage) / 1024


def _parse_int_list(value: str) -> list[int]:
    values = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark the real sparse connectome policy without updating model weights."
    )
    parser.add_argument("--graph-dir", type=Path, default=Path("artifacts/graphs"))
    parser.add_argument("--rollout-batch-size", type=int, default=16)
    parser.add_argument("--forward-samples", type=_parse_int_list, default=[100, 1000])
    parser.add_argument("--training-batch-sizes", type=_parse_int_list, default=[8, 16, 32])
    parser.add_argument("--training-iterations", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--max-environment-steps", type=int, default=2_000_000)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--threads", type=int, default=0, help="0 keeps PyTorch's default thread count")
    args = parser.parse_args()

    if args.rollout_batch_size <= 0 or args.training_iterations <= 0 or args.warmup < 0:
        parser.error("batch sizes/iterations must be positive and warmup non-negative")
    if args.threads < 0:
        parser.error("threads must be non-negative")
    if args.threads:
        torch.set_num_threads(args.threads)

    graph_dir = args.graph_dir
    graph = load_connectome_graph(
        graph_dir / "subgraph_v1.npz",
        graph_dir / "subgraph_v1_nodes.parquet",
    )
    recurrent = to_sparse_recurrent(graph)
    policy = ConnectomePolicy(graph, recurrent, microsteps=4)

    forward_results = []
    for requested_samples in args.forward_samples:
        iterations = max(1, math.ceil(requested_samples / args.rollout_batch_size))
        result = benchmark_policy_forward(
            policy,
            batch_size=args.rollout_batch_size,
            iterations=iterations,
            warmup=args.warmup,
        )
        forward_results.append(
            {
                "requested_samples": requested_samples,
                **asdict(result),
            }
        )

    backward_results = []
    for batch_size in args.training_batch_sizes:
        result = benchmark_policy_backward(
            policy,
            batch_size=batch_size,
            iterations=args.training_iterations,
            warmup=args.warmup,
        )
        backward_results.append(asdict(result))

    rollout_rate = forward_results[-1]["samples_per_second"]
    fastest_backward = max(backward_results, key=lambda item: item["samples_per_second"])
    estimate = estimate_training_seconds(
        max_environment_steps=args.max_environment_steps,
        ppo_epochs=args.ppo_epochs,
        rollout_samples_per_second=float(rollout_rate),
        training_samples_per_second=float(fastest_backward["samples_per_second"]),
    )

    payload = {
        "node_count": graph.node_count,
        "edge_count": graph.edge_count,
        "input_count": int(graph.input_mask.sum()),
        "output_count": int(graph.output_mask.sum()),
        "microsteps": policy.microsteps,
        "torch_threads": torch.get_num_threads(),
        "forward": forward_results,
        "backward": backward_results,
        "fastest_training_batch_size": fastest_backward["batch_size"],
        "estimated_training": {
            **asdict(estimate),
            "total_hours": estimate.total_seconds / 3600.0,
            "total_days": estimate.total_seconds / 86400.0,
        },
        "peak_process_rss_mb": _peak_rss_mb(),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
