from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import imageio_ffmpeg


WIDTH = 1280
HEIGHT = 720
ROOM_W = 10.0
ROOM_H = 6.0
WINDOW_CENTER_Y = 3.0
WINDOW_WIDTH = 1.5


def _font(size: int):
    for name in ("DejaVuSans.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def _panel_transform(rect):
    x0, y0, x1, y1 = rect
    sx = (x1 - x0) / ROOM_W
    sy = (y1 - y0) / ROOM_H

    def point(x: float, y: float):
        return (x0 + x * sx, y1 - y * sy)

    return point


def _draw_panel(draw, rect, trajectory, frame_index, title):
    x0, y0, x1, y1 = rect
    point = _panel_transform(rect)
    draw.rectangle(rect, outline=(20, 20, 20), width=4, fill=(247, 247, 245))

    wy0 = WINDOW_CENTER_Y - WINDOW_WIDTH / 2
    wy1 = WINDOW_CENTER_Y + WINDOW_WIDTH / 2
    _, py1 = point(ROOM_W, wy1)
    _, py0 = point(ROOM_W, wy0)
    draw.line([(x1, py1), (x1, py0)], fill=(255, 205, 35), width=10)

    frames = trajectory["frames"]
    idx = min(frame_index, len(frames) - 1)
    current = frames[idx]
    trail = frames[: idx + 1]
    if len(trail) > 1:
        draw.line([point(f["x"], f["y"]) for f in trail], fill=(95, 95, 95), width=2)

    px, py = point(current["x"], current["y"])
    r = 7
    draw.ellipse((px - r, py - r, px + r, py + r), fill=(25, 25, 25))
    heading = current["heading"]
    hx = px + 18 * math.cos(heading)
    hy = py - 18 * math.sin(heading)
    draw.line([(px, py), (hx, hy)], fill=(220, 55, 45), width=3)

    draw.text((x0, y0 - 42), title, font=_font(28), fill=(15, 15, 15))
    status = "EXIT" if current["done"] and trajectory["success"] else f"step {current['step']}"
    draw.text((x0 + 8, y1 + 10), status, font=_font(20), fill=(20, 20, 20))


def _frame(before, after, frame_index, seed, trained_steps):
    image = Image.new("RGB", (WIDTH, HEIGHT), (235, 235, 232))
    draw = ImageDraw.Draw(image)
    draw.text((42, 24), "Teaching a connectome-constrained fly to find an open window", font=_font(34), fill=(15, 15, 15))
    draw.text((42, 72), f"MaleCNS v1.0 • fixed demo seed {seed}", font=_font(22), fill=(55, 55, 55))

    left = (45, 150, 615, 492)
    right = (665, 150, 1235, 492)
    _draw_panel(draw, left, before, frame_index, "BEFORE TRAINING")
    _draw_panel(draw, right, after, frame_index, f"AFTER {trained_steps:,} ENV STEPS")

    draw.text((42, 570), "Same room. Same starting seed. Same published connectome.", font=_font(24), fill=(25, 25, 25))
    draw.text((42, 610), "Trainable: visual encoder + descending readout. Connectome topology stays fixed.", font=_font(20), fill=(55, 55, 55))
    draw.text((42, 650), "Dynamics, sensory encoding and learning rule are modeled.", font=_font(18), fill=(85, 85, 85))
    return np.asarray(image, dtype=np.uint8)


def main():
    parser = argparse.ArgumentParser(description="Render before/after Fly Window trajectories to MP4.")
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--hold-frames", type=int, default=10)
    args = parser.parse_args()

    before_payload = json.loads(args.before.read_text(encoding="utf-8"))
    after_payload = json.loads(args.after.read_text(encoding="utf-8"))
    if before_payload["demo_seeds"] != after_payload["demo_seeds"]:
        raise ValueError("before/after demo seeds must match")

    before_by_seed = {t["seed"]: t for t in before_payload["trajectories"]}
    after_by_seed = {t["seed"]: t for t in after_payload["trajectories"]}
    trained_steps = int(after_payload["checkpoint_environment_steps"])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio_ffmpeg.write_frames(
        str(args.output),
        (WIDTH, HEIGHT),
        fps=args.fps,
        codec="libx264",
        pix_fmt_in="rgb24",
        pix_fmt_out="yuv420p",
        ffmpeg_log_level="warning",
        output_params=["-crf", "20", "-movflags", "+faststart"],
    )
    writer.send(None)
    try:
        for seed in before_payload["demo_seeds"]:
            before = before_by_seed[seed]
            after = after_by_seed[seed]
            max_steps = max(before["steps"], after["steps"])
            sampled = list(range(0, max_steps + 1, args.stride))
            if sampled[-1] != max_steps:
                sampled.append(max_steps)
            for step in sampled:
                writer.send(_frame(before, after, step, seed, trained_steps).tobytes())
            final_frame = _frame(before, after, max_steps, seed, trained_steps).tobytes()
            for _ in range(args.hold_frames):
                writer.send(final_frame)
    finally:
        writer.close()

    print(args.output)


if __name__ == "__main__":
    main()
