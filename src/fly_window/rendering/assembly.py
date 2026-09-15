from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fly_window.rendering.media import RELEASE_MEDIA_SPEC
from fly_window.rendering.release import HeldoutMetric, build_release_timeline, heldout_metric


BIOLOGICAL_MODEL_LABEL = "oracle-imitation-biological"


@dataclass(frozen=True, slots=True)
class ReleaseAssemblySegment:
    kind: str
    seed: int | None
    source_name: str
    duration_seconds: float


def build_release_assembly_plan() -> tuple[ReleaseAssemblySegment, ...]:
    timeline = build_release_timeline(fps=RELEASE_MEDIA_SPEC.fps)
    source_names = (
        "hook.png",
        "development-untrained.mp4",
        "development-midpoint.mp4",
        "development-final.mp4",
        "heldout-10000.mp4",
        "heldout-10001.mp4",
        "heldout-10002.mp4",
        "end-card.png",
    )
    if len(timeline.segments) != len(source_names):
        raise RuntimeError("release timeline and source plan have different lengths")

    return tuple(
        ReleaseAssemblySegment(
            kind=segment.kind,
            seed=segment.seed,
            source_name=source_name,
            duration_seconds=segment.frame_count / timeline.fps,
        )
        for segment, source_name in zip(timeline.segments, source_names, strict=True)
    )


def validated_release_metric(evaluation: dict[str, Any]) -> HeldoutMetric:
    release_gate = evaluation.get("release_gate")
    if not isinstance(release_gate, dict) or release_gate.get("passed") is not True:
        raise ValueError("release gate did not pass")
    return heldout_metric(evaluation, BIOLOGICAL_MODEL_LABEL)
