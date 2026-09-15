from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any


@dataclass(frozen=True, slots=True)
class ReleaseSegment:
    kind: str
    seed: int | None
    start_frame: int
    end_frame: int

    @property
    def frame_count(self) -> int:
        return self.end_frame - self.start_frame


@dataclass(frozen=True, slots=True)
class ReleaseTimeline:
    fps: int
    segments: tuple[ReleaseSegment, ...]

    @property
    def total_frames(self) -> int:
        return self.segments[-1].end_frame if self.segments else 0

    @property
    def duration_seconds(self) -> float:
        return self.total_frames / self.fps


@dataclass(frozen=True, slots=True)
class HeldoutMetric:
    successes: int
    episode_count: int
    success_rate: float

    @property
    def display(self) -> str:
        percentage = round(self.success_rate * 100)
        return f"{self.successes}/{self.episode_count} held-out successes ({percentage}%)"


def build_release_timeline(*, fps: int = 30) -> ReleaseTimeline:
    if fps <= 0:
        raise ValueError("fps must be positive")

    slots = (
        ("hook", None, 3),
        ("development-untrained", 9000, 6),
        ("development-midpoint", 9000, 6),
        ("development-final", 9000, 6),
        ("heldout", 10000, 3),
        ("heldout", 10001, 3),
        ("heldout", 10002, 3),
        ("end-card", None, 2),
    )
    segments: list[ReleaseSegment] = []
    cursor = 0
    for kind, seed, seconds in slots:
        frames = int(seconds * fps)
        segments.append(
            ReleaseSegment(
                kind=kind,
                seed=seed,
                start_frame=cursor,
                end_frame=cursor + frames,
            )
        )
        cursor += frames
    return ReleaseTimeline(fps=fps, segments=tuple(segments))


def heldout_metric(evaluation: dict[str, Any], model_label: str) -> HeldoutMetric:
    if evaluation.get("held_out_seed_start") != 10000 or evaluation.get("held_out_seed_end") != 10099:
        raise ValueError("evaluation must use held-out seeds 10000..10099")

    models = evaluation.get("models")
    if not isinstance(models, list):
        raise ValueError("evaluation must contain a models list")
    matches = [model for model in models if model.get("model_label") == model_label]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one model summary for {model_label!r}")

    summary = matches[0]
    episode_count = int(summary.get("episode_count", -1))
    if episode_count != 100:
        raise ValueError("held-out metric must contain exactly 100 episodes")
    successes = int(summary.get("successes", -1))
    if not 0 <= successes <= episode_count:
        raise ValueError("held-out successes must lie in [0, 100]")
    success_rate = float(summary.get("success_rate", math.nan))
    expected_rate = successes / episode_count
    if not math.isfinite(success_rate) or not math.isclose(
        success_rate, expected_rate, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError("held-out success_rate must equal successes / episode_count")

    return HeldoutMetric(
        successes=successes,
        episode_count=episode_count,
        success_rate=success_rate,
    )
