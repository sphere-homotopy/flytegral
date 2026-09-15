from __future__ import annotations

from PIL import Image, ImageDraw

import render_cinematic_demo_v3 as v3

from fly_window.rendering.media import RELEASE_MEDIA_SPEC


base = v3.base
spec = RELEASE_MEDIA_SPEC

base.WIDTH = spec.width
base.HEIGHT = spec.height
base.FPS = spec.fps

_original_brain_static_layer = base._brain_static_layer
_original_overlay_brain = base._overlay_brain


def _brain_static_layer(projected, valid, size=None):
    panel_height = spec.height - 2 * spec.brain_panel_margin
    return _original_brain_static_layer(
        projected,
        valid,
        size=(spec.brain_panel_width, panel_height),
    )


def _overlay_brain(frame: Image.Image, static: Image.Image, projected, brain):
    separator_x = spec.width - spec.brain_panel_width - 2 * spec.brain_panel_margin
    veil = Image.new("RGBA", frame.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(veil, "RGBA")
    draw.rectangle(
        (separator_x, 0, spec.width, spec.height),
        fill=(1, 4, 8, 184),
    )
    draw.rectangle(
        (separator_x, 0, separator_x + 2, spec.height),
        fill=(152, 174, 201, 42),
    )
    frame.alpha_composite(veil)
    _original_overlay_brain(frame, static, projected, brain)


base._brain_static_layer = _brain_static_layer
base._overlay_brain = _overlay_brain


if __name__ == "__main__":
    base.main()
