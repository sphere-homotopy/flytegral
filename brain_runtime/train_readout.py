"""Train a compact Flytegral scalar readout on fixed MaleCNS spike activity.

This script does not alter connectome topology or synaptic weights. It repeatedly
resets the pinned DOOMFLY MaleCNS model, presents deterministic graph rasters,
projects whole-brain spike rates into stable hashed population features, and fits
one ridge-regression readout to the slider target.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from brain_runtime.runtime import activity_features, load_doomfly_brain
from brain_runtime.training import answer_to_slider, fit_ridge, generate_problem, render_graph_raster

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DOOMFLY = ROOT / ".vendor" / "doomfly"
DEFAULT_GRAPH = DEFAULT_DOOMFLY / "outputs" / "doom" / "malecns_v1" / "graph.npz"
DEFAULT_OUTPUT = ROOT / "brain_runtime" / "readout.npz"
UPSTREAM_COMMIT = "71ecf53d78eaffaf1a57ed7b0ccf5d458abc9f33"


def make_stimulus(problem: dict, width: int = 64, height: int = 40) -> dict:
    raster = render_graph_raster(problem, width=width, height=height)
    return {
        "width": width,
        "height": height,
        "luminance": raster.reshape(-1).astype(float).tolist(),
    }


def collect(brain, seeds, *, bins: int, frame_ms: float, repeats: int):
    rows = []
    targets = []
    spike_counts = []
    started = time.perf_counter()
    total = len(seeds)
    for offset, seed in enumerate(seeds, start=1):
        problem = generate_problem(seed)
        stimulus = make_stimulus(problem)
        features, telemetry = activity_features(
            brain,
            stimulus,
            bins=bins,
            frame_ms=frame_ms,
            repeats=repeats,
        )
        rows.append(features)
        targets.append(answer_to_slider(problem["target"], problem["answerRange"]))
        spike_counts.append(telemetry["totalSpikes"])
        if offset == 1 or offset % 10 == 0 or offset == total:
            elapsed = time.perf_counter() - started
            print(
                f"[{offset:4d}/{total}] seed={seed} spikes={telemetry['totalSpikes']} "
                f"sim={telemetry['simMs']:.1f}ms elapsed={elapsed:.1f}s",
                flush=True,
            )
    return (
        np.asarray(rows, dtype=np.float32),
        np.asarray(targets, dtype=np.float32),
        np.asarray(spike_counts, dtype=np.int64),
    )


def score(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.clip(np.asarray(y_pred, dtype=np.float64), 0.0, 1.0)
    return {
        "mae": float(np.mean(np.abs(y_true - y_pred))),
        "rmse": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
        "maxAbsoluteError": float(np.max(np.abs(y_true - y_pred))),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--doomfly-root", type=Path, default=DEFAULT_DOOMFLY)
    parser.add_argument("--graph", type=Path, default=DEFAULT_GRAPH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--train", type=int, default=128)
    parser.add_argument("--test", type=int, default=64)
    parser.add_argument("--seed-base", type=int, default=910_000)
    parser.add_argument("--bins", type=int, default=512)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--frame-ms", type=float, default=20.0)
    parser.add_argument("--repeats", type=int, default=4)
    args = parser.parse_args()

    if args.train < 8 or args.test < 4:
        raise SystemExit("--train must be >= 8 and --test must be >= 4")
    if args.bins < 8:
        raise SystemExit("--bins must be >= 8")

    brain = load_doomfly_brain(args.doomfly_root, args.graph)
    train_seeds = list(range(args.seed_base, args.seed_base + args.train))
    test_start = args.seed_base + 1_000_000
    test_seeds = list(range(test_start, test_start + args.test))

    print(
        f"MaleCNS readout training: neurons={brain.n} train={args.train} test={args.test} "
        f"bins={args.bins} stimulus={args.frame_ms * args.repeats:.1f}ms",
        flush=True,
    )
    x_train, y_train, train_spikes = collect(
        brain, train_seeds, bins=args.bins, frame_ms=args.frame_ms, repeats=args.repeats
    )
    model = fit_ridge(x_train, y_train, alpha=args.alpha)
    train_pred = np.clip(model.predict(x_train), 0.0, 1.0)

    x_test, y_test, test_spikes = collect(
        brain, test_seeds, bins=args.bins, frame_ms=args.frame_ms, repeats=args.repeats
    )
    test_pred = np.clip(model.predict(x_test), 0.0, 1.0)
    train_metrics = score(y_train, train_pred)
    test_metrics = score(y_test, test_pred)
    midpoint_metrics = score(y_test, np.full_like(y_test, 0.5))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        weights=model.weights,
        mean=model.mean,
        scale=model.scale,
        bias=np.asarray(model.bias, dtype=np.float32),
        bins=np.asarray(args.bins, dtype=np.int32),
        validation_mae=np.asarray(test_metrics["mae"], dtype=np.float32),
    )

    metrics = {
        "schema": 1,
        "upstream": {
            "repository": "nftechie/doomfly",
            "commit": UPSTREAM_COMMIT,
            "dataset": "MaleCNS v1.0",
        },
        "training": {
            "trainSamples": args.train,
            "testSamples": args.test,
            "seedBase": args.seed_base,
            "testSeedBase": test_start,
            "bins": args.bins,
            "alpha": args.alpha,
            "frameMs": args.frame_ms,
            "repeats": args.repeats,
            "topologyChanged": False,
            "synapticWeightsChanged": False,
            "trainedParameters": "scalar ridge readout only",
        },
        "train": train_metrics,
        "test": test_metrics,
        "midpointBaseline": midpoint_metrics,
        "activity": {
            "trainMedianSpikes": float(np.median(train_spikes)),
            "testMedianSpikes": float(np.median(test_spikes)),
        },
    }
    metrics_path = args.output.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))
    print(f"saved readout: {args.output}")
    print(f"saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
