from __future__ import annotations

from dataclasses import dataclass

from fly_window.rendering.release import build_release_timeline


@dataclass(frozen=True, slots=True)
class ReleaseMediaSpec:
    width: int
    height: int
    fps: int
    video_codec: str
    pixel_format: str
    brain_panel_width: int
    brain_panel_margin: int

    @property
    def duration_seconds(self) -> float:
        return build_release_timeline(fps=self.fps).duration_seconds


RELEASE_MEDIA_SPEC = ReleaseMediaSpec(
    width=1600,
    height=900,
    fps=30,
    video_codec="libx264",
    pixel_format="yuv420p",
    brain_panel_width=520,
    brain_panel_margin=22,
)
