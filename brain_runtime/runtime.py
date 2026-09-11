"""MaleCNS runtime primitives for Flytegral.

The neural engine itself comes from a pinned DOOMFLY checkout. This module owns
only Flytegral's visual stimulus sampling, reset policy and trained scalar
readout. It never changes the MaleCNS topology.
"""
from __future__ import annotations

from dataclasses import dataclass
import importlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


def sample_retina(raster: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """Nearest-neighbour sample of a 2-D luminance raster at normalized UV points."""
    raster = np.asarray(raster, dtype=np.float32)
    uv = np.asarray(uv, dtype=np.float32)
    if raster.ndim != 2 or raster.size == 0:
        raise ValueError("raster must be a non-empty 2-D array")
    if uv.ndim != 2 or uv.shape[1] != 2 or not np.isfinite(uv).all():
        raise ValueError("uv must have shape (n, 2) with finite values")
    if np.any(uv < 0) or np.any(uv > 1):
        raise ValueError("uv coordinates must lie in [0, 1]")
    y = np.rint(uv[:, 1] * (raster.shape[0] - 1)).astype(np.int64)
    x = np.rint(uv[:, 0] * (raster.shape[1] - 1)).astype(np.int64)
    return np.ascontiguousarray(raster[y, x], dtype=np.float32)


def project_counts(counts: np.ndarray, ids: np.ndarray, bins: int = 512) -> np.ndarray:
    """Deterministic signed hash projection of whole-brain spike counts."""
    if not isinstance(bins, int) or bins < 8:
        raise ValueError("bins must be an integer >= 8")
    counts = np.asarray(counts)
    ids = np.asarray(ids)
    if counts.ndim != 1 or ids.ndim != 1 or len(counts) != len(ids):
        raise ValueError("counts and ids must be aligned 1-D arrays")
    if not np.isfinite(counts).all():
        raise ValueError("counts must be finite")

    ids_u = ids.astype(np.uint64, copy=False)
    bucket = np.remainder(ids_u, np.uint64(bins)).astype(np.int64)
    # A second bit of the stable biological ID supplies the projection sign.
    sign = np.where(((ids_u >> np.uint64(9)) & np.uint64(1)) == 0, 1.0, -1.0)
    projected = np.zeros(bins, dtype=np.float32)
    np.add.at(projected, bucket, counts.astype(np.float32, copy=False) * sign)
    return projected


def decode_readout(features: np.ndarray, weights: np.ndarray, bias: float = 0.0) -> float:
    features = np.asarray(features, dtype=np.float32)
    weights = np.asarray(weights, dtype=np.float32)
    if features.shape != weights.shape or features.ndim != 1:
        raise ValueError("features and weights must be aligned 1-D arrays")
    raw = float(np.dot(features, weights) + float(bias))
    return float(np.clip(raw, 0.0, 1.0))


def _reset_brain(brain: Any) -> None:
    """Reset DOOMFLY Brain state without reloading the 1+ GB graph."""
    brain.cursor = 0
    brain.v.fill(-52)
    brain.g.fill(0)
    brain.drive.fill(0)
    brain.refractory.fill(0)
    brain.queue.fill(0)
    brain.queue_count.fill(0)
    brain.counts.fill(0)
    brain.luminance.fill(0)
    brain.active.fill(0)
    brain.active_flag.fill(0)
    initial = np.unique(np.r_[brain.retina, brain.lamina, brain.sugar]).astype(np.int32, copy=False)
    brain.active[: len(initial)] = initial
    brain.active_flag[initial] = 1
    brain.nactive[0] = len(initial)
    brain.total_spikes = 0
    brain.sim_ms = 0


def load_doomfly_brain(doomfly_root: Path, graph_path: Path):
    doomfly_root = Path(doomfly_root).resolve()
    graph_path = Path(graph_path).resolve()
    if not graph_path.exists():
        raise FileNotFoundError(f"MaleCNS graph not found: {graph_path}")
    if not (doomfly_root / "doom" / "engine.py").exists():
        raise FileNotFoundError(f"Pinned DOOMFLY checkout not found: {doomfly_root}")

    sys.path.insert(0, str(doomfly_root))
    try:
        engine = importlib.import_module("doom.engine")
        return engine.Brain(str(graph_path))
    finally:
        try:
            sys.path.remove(str(doomfly_root))
        except ValueError:
            pass


@dataclass
class Readout:
    weights: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    bias: float
    bins: int
    validation_mae: float | None = None

    @classmethod
    def load(cls, path: Path) -> "Readout":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"Flytegral MaleCNS readout not found: {path}. Run brain_runtime/train_readout.py first."
            )
        with np.load(path, allow_pickle=False) as data:
            weights = np.asarray(data["weights"], dtype=np.float32)
            mean = np.asarray(data["mean"], dtype=np.float32)
            scale = np.asarray(data["scale"], dtype=np.float32)
            bias = float(np.asarray(data["bias"]).item())
            bins = int(np.asarray(data["bins"]).item())
            validation_mae = float(np.asarray(data["validation_mae"]).item()) if "validation_mae" in data else None
        if weights.shape != (bins,) or mean.shape != (bins,) or scale.shape != (bins,):
            raise ValueError("readout arrays do not match bins")
        scale = np.where(scale > 1e-6, scale, 1.0).astype(np.float32)
        return cls(weights, mean, scale, bias, bins, validation_mae)

    def predict(self, projected_rates: np.ndarray) -> float:
        normalized = (np.asarray(projected_rates, dtype=np.float32) - self.mean) / self.scale
        return decode_readout(normalized, self.weights, self.bias)


class MaleCNSRuntime:
    def __init__(self, brain: Any, readout: Readout, *, frame_ms: float = 20.0, repeats: int = 4):
        self.brain = brain
        self.readout = readout
        self.frame_ms = float(frame_ms)
        self.repeats = int(repeats)
        if self.frame_ms <= 0 or self.repeats < 1:
            raise ValueError("frame_ms and repeats must be positive")

    def estimate(self, stimulus: dict[str, Any]) -> dict[str, Any]:
        width = int(stimulus.get("width", 0))
        height = int(stimulus.get("height", 0))
        values = np.asarray(stimulus.get("luminance", []), dtype=np.float32)
        if width < 2 or height < 2 or values.size != width * height:
            raise ValueError("stimulus dimensions do not match luminance length")
        if not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1):
            raise ValueError("stimulus luminance must be finite and in [0, 1]")
        raster = values.reshape(height, width)
        receptor_luminance = sample_retina(raster, self.brain.uv)

        _reset_brain(self.brain)
        total = np.zeros(self.brain.n, dtype=np.int64)
        elapsed_wall = 0.0
        for _ in range(self.repeats):
            counts, elapsed = self.brain.step(receptor_luminance, self.frame_ms)
            total += counts
            elapsed_wall += float(elapsed)

        sim_ms = self.frame_ms * self.repeats
        rates = total.astype(np.float32) * (1000.0 / sim_ms)
        projected = project_counts(rates, self.brain.ids, bins=self.readout.bins)
        slider = self.readout.predict(projected)
        validation_mae = self.readout.validation_mae
        confidence = None if validation_mae is None else float(np.clip(1.0 - 2.0 * validation_mae, 0.0, 1.0))

        return {
            "sliderPosition": slider,
            "confidence": confidence,
            "trace": [0.5, float(np.clip(0.5 + (slider - 0.5) * 0.55, 0, 1)), slider],
            "telemetry": {
                "neurons": int(self.brain.n),
                "retinaMapped": int(len(self.brain.retina)),
                "totalSpikes": int(total.sum()),
                "activeNeurons": int(np.count_nonzero(total)),
                "simMs": sim_ms,
                "wallSeconds": elapsed_wall,
                "readoutBins": self.readout.bins,
            },
        }


def runtime_from_paths(doomfly_root: Path, graph_path: Path, readout_path: Path) -> MaleCNSRuntime:
    brain = load_doomfly_brain(doomfly_root, graph_path)
    readout = Readout.load(readout_path)
    return MaleCNSRuntime(brain, readout)


def describe_runtime(runtime: MaleCNSRuntime) -> str:
    payload = {
        "neurons": int(runtime.brain.n),
        "retinaMapped": int(len(runtime.brain.retina)),
        "readoutBins": runtime.readout.bins,
        "validationMae": runtime.readout.validation_mae,
    }
    return json.dumps(payload, sort_keys=True)
