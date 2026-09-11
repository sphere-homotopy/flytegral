"""Deterministic Flytegral task/raster generation and scalar readout fitting."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable

import numpy as np

UINT32_MASK = 0xFFFF_FFFF
UINT32_SCALE = float(0x1_0000_0000)


def _imul(a: int, b: int) -> int:
    return (a * b) & UINT32_MASK


def create_rng(seed: int) -> Callable[[], float]:
    """Port of src/math.js createRng for integer seeds."""
    state = int(seed) & UINT32_MASK

    def random() -> float:
        nonlocal state
        state = (state + 0x6D2B79F5) & UINT32_MASK
        value = state
        value = _imul(value ^ (value >> 15), value | 1)
        value ^= (value + _imul(value ^ (value >> 7), value | 61)) & UINT32_MASK
        value &= UINT32_MASK
        return ((value ^ (value >> 14)) & UINT32_MASK) / UINT32_SCALE

    return random


def evaluate_polynomial(coefficients, x: float) -> float:
    acc = 0.0
    for coefficient in reversed(coefficients):
        acc = acc * x + coefficient
    return float(acc)


def integrate_polynomial(coefficients, a: float, b: float) -> float:
    total = 0.0
    for degree, coefficient in enumerate(coefficients):
        exponent = degree + 1
        total += coefficient / exponent * (b**exponent - a**exponent)
    return float(total)


def _choose(rng, values):
    return values[math.floor(rng() * len(values))]


def _quantized_coefficient(rng, max_abs=2.25, step=0.25) -> float:
    units = math.floor(max_abs / step + 0.5)
    return float((math.floor(rng() * (units * 2 + 1)) - units) * step)


def _answer_range(target: float) -> list[float]:
    raw_radius = max(4.0, abs(target) * 1.75 + 1.5)
    radius = math.ceil(raw_radius * 2.0) / 2.0
    return [-radius, radius]


def generate_problem(seed: int, coefficient_max_abs: float = 2.25) -> dict:
    rng = create_rng(seed)
    coefficients = [_quantized_coefficient(rng, coefficient_max_abs) for _ in range(4)]
    if all(coefficient == 0 for coefficient in coefficients[1:]):
        coefficients[1 + math.floor(rng() * 3)] = _choose(rng, [-1.0, -0.5, 0.5, 1.0])

    endpoints = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
    start_index = math.floor(rng() * (len(endpoints) - 2))
    possible_end_indexes = list(range(start_index + 2, len(endpoints)))
    end_index = _choose(rng, possible_end_indexes)
    interval = [endpoints[start_index], endpoints[end_index]]
    target = integrate_polynomial(coefficients, *interval)

    return {
        "seed": int(seed),
        "coefficients": coefficients,
        "interval": interval,
        "target": target,
        "answerRange": _answer_range(target),
        "graphDomain": [-2.5, 2.5],
    }


def answer_to_slider(value: float, answer_range) -> float:
    low, high = answer_range
    if not high > low:
        raise ValueError("answer range must have positive width")
    return float(np.clip((value - low) / (high - low), 0.0, 1.0))


def _js_round_positive(value: float) -> int:
    return math.floor(value + 0.5)


def _graph_extents(problem: dict):
    x_min, x_max = problem["graphDomain"]
    max_abs = 1.0
    for index in range(161):
        x = x_min + index / 160 * (x_max - x_min)
        max_abs = max(max_abs, abs(evaluate_polynomial(problem["coefficients"], x)))
    return x_min, x_max, -max_abs * 1.14, max_abs * 1.14


def _draw_point(buffer, width, height, x, y, luminance, radius=1):
    px = _js_round_positive(x)
    py = _js_round_positive(y)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            sx, sy = px + dx, py + dy
            if sx < 0 or sy < 0 or sx >= width or sy >= height:
                continue
            distance = math.hypot(dx, dy)
            if distance > radius + 0.25:
                continue
            buffer[sy, sx] = min(float(buffer[sy, sx]), luminance + distance * 0.08)


def render_graph_raster(problem: dict, *, width: int = 64, height: int = 40, padding: int = 3) -> np.ndarray:
    if width < 16 or height < 12:
        raise ValueError("raster dimensions are too small")
    raster = np.ones((height, width), dtype=np.float32)
    x_min, x_max, y_min, y_max = _graph_extents(problem)
    inner_width = width - padding * 2 - 1
    inner_height = height - padding * 2 - 1

    def x_to_px(x):
        return padding + ((x - x_min) / (x_max - x_min)) * inner_width

    def y_to_px(y):
        return padding + ((y_max - y) / (y_max - y_min)) * inner_height

    zero_y = y_to_px(0)
    a, b = problem["interval"]
    left = max(0, math.ceil(x_to_px(a)))
    right = min(width - 1, math.floor(x_to_px(b)))
    for px in range(left, right + 1):
        x = x_min + ((px - padding) / inner_width) * (x_max - x_min)
        curve_y = y_to_px(evaluate_polynomial(problem["coefficients"], x))
        start = max(0, math.ceil(min(zero_y, curve_y)))
        end = min(height - 1, math.floor(max(zero_y, curve_y)))
        if start <= end:
            raster[start : end + 1, px] = np.minimum(raster[start : end + 1, px], 0.86)

    zero_row = _js_round_positive(zero_y)
    if 0 <= zero_row < height:
        raster[zero_row, padding : width - padding] = np.minimum(
            raster[zero_row, padding : width - padding], 0.72
        )
    if x_min <= 0 <= x_max:
        zero_column = _js_round_positive(x_to_px(0))
        raster[padding : height - padding, zero_column] = np.minimum(
            raster[padding : height - padding, zero_column], 0.72
        )

    steps = max(width * 5, 240)
    for index in range(steps + 1):
        x = x_min + index / steps * (x_max - x_min)
        _draw_point(
            raster,
            width,
            height,
            x_to_px(x),
            y_to_px(evaluate_polynomial(problem["coefficients"], x)),
            0.08,
            1,
        )

    for boundary in (a, b):
        px = _js_round_positive(x_to_px(boundary))
        for py in range(padding, height - padding, 2):
            raster[py, px] = min(float(raster[py, px]), 0.46)

    # Browser encoder serializes values rounded to four decimal places.
    return np.round(np.clip(raster, 0.0, 1.0), 4).astype(np.float32)


@dataclass
class RidgeModel:
    weights: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    bias: float

    def predict(self, x) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        normalized = (x - self.mean) / self.scale
        return normalized @ self.weights + self.bias


def fit_ridge(x, y, *, alpha: float = 1e-2) -> RidgeModel:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 1 or len(x) != len(y) or len(x) < 2:
        raise ValueError("x must be 2-D and aligned with 1-D y")
    if alpha < 0 or not np.isfinite(alpha):
        raise ValueError("alpha must be finite and nonnegative")

    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale = np.where(scale > 1e-8, scale, 1.0)
    z = (x - mean) / scale
    bias = float(y.mean())
    centered_y = y - bias
    gram = z.T @ z
    rhs = z.T @ centered_y
    weights = np.linalg.solve(gram + np.eye(z.shape[1]) * alpha, rhs)
    return RidgeModel(
        weights=weights.astype(np.float32),
        mean=mean.astype(np.float32),
        scale=scale.astype(np.float32),
        bias=bias,
    )
