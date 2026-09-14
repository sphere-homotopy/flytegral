from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class FlyState:
    x: float
    y: float
    heading: float
    speed: float
    step_count: int


@dataclass(frozen=True, slots=True)
class StepResult:
    observation: np.ndarray
    state: FlyState
    reward: float
    terminated: bool
    truncated: bool
    success: bool
    collision: bool
