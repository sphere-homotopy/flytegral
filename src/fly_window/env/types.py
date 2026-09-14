from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class FlyState:
    x: float
    y: float
    heading: float
    speed: float
    step_count: int
