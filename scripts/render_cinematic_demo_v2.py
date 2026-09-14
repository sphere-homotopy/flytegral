from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
import tempfile
import wave

import imageio_ffmpeg
import mujoco
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from fly_window.rendering.cinematic import env_to_world, yaw_quaternion

WIDTH = 1280
HEIGHT = 720
FPS = 24
WINDOW_HALF_Y = 0.075


def _font(size: int, bold: bool = False):
    candidates = [
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            pass
    return ImageFont.load_default()


def _write_scene(flybody_dir: Path) -> Path:
    scene = flybody_dir / "fly_window_cinematic_v2.xml"
    scene.write_text(
        f'''<mujoco model="fly window cinematic v2">
  <include file="fruitfly.xml"/>
  <visual>
    <global offwidth="{WIDTH}" offheight="{HEIGHT}" azimuth="315" elevation="-18"/>
    <quality shadowsize="4096" offsamples="4"/>
    <headlight ambient="0.10 0.11 0.13" diffuse="0.26 0.27 0.29" specular="0.04 0.04 0.05"/>
  </visual>
  <statistic meansize="0.03" extent="1.15" center="0 0 0.13"/>
  <asset>
    <texture name="room_sky" type="skybox" builtin="gradient" rgb1="0.025 0.035 0.055" rgb2="0.004 0.006 0.010" width="256" height="256"/>
    <material name="floor_m" rgba="0.055 0.064 0.075 1" roughness="0.76"/>
    <material name="wall_m" rgba="0.075 0.084 0.095 1" roughness="0.92"/>
    <material name="frame_m" rgba="0.030 0.034 0.041 1" roughness="0.62"/>
    <material name="outside_m" rgba="0.66 0.82 1.0 1" emission="0.48"/>
  </asset>
  <worldbody>
    <geom name="room_floor" type="box" pos="0 0 -0.012" size="0.53 0.33 0.012" material="floor_m" contype="0" conaffinity="0"/>
    <geom name="room_back" type="box" pos="0 0.315 0.16" size="0.53 0.015 0.16" material="wall_m" contype="0" conaffinity="0"/>

    <!-- The near/front side is intentionally open for the cinematic camera. -->
    <geom name="right_lower_y" type="box" pos="0.515 -0.195 0.16" size="0.015 0.12 0.16" material="wall_m" contype="0" conaffinity="0"/>
    <geom name="right_upper_y" type="box" pos="0.515 0.195 0.16" size="0.015 0.12 0.16" material="wall_m" contype="0" conaffinity="0"/>
    <geom name="window_sill" type="box" pos="0.515 0 0.022" size="0.015 {WINDOW_HALF_Y} 0.022" material="frame_m" contype="0" conaffinity="0"/>
    <geom name="window_header" type="box" pos="0.515 0 0.278" size="0.015 {WINDOW_HALF_Y} 0.042" material="frame_m" contype="0" conaffinity="0"/>
    <geom name="window_frame_low" type="box" pos="0.507 -{WINDOW_HALF_Y} 0.15" size="0.010 0.007 0.106" material="frame_m" contype="0" conaffinity="0"/>
    <geom name="window_frame_high" type="box" pos="0.507 {WINDOW_HALF_Y} 0.15" size="0.010 0.007 0.106" material="frame_m" contype="0" conaffinity="0"/>
    <geom name="outside_glow" type="box" pos="0.538 0 0.15" size="0.004 0.068 0.102" material="outside_m" contype="0" conaffinity="0"/>

    <light name="window_light" pos="0.41 0 0.22" dir="-1 0 -0.16" directional="true" diffuse="1.05 1.08 1.16" specular="0.14 0.15 0.17" castshadow="true"/>
    <light name="fill_light" pos="-0.32 -0.24 0.28" dir="0.55 0.35 -0.30" directional="true" diffuse="0.17 0.19 0.24" specular="0.04 0.04 0.05"/>
  </worldbody>
</mujoco>\n''',
        encoding="utf-8",
    )
    return scene


def _load_soma_projection(annotations_path: Path, graph_npz: Path):
    graph = np.load(graph_npz)
    body_ids = graph["body_ids"].astype(np.int64)
    annotations = pd.read_feather(annotations_path, columns=["bodyId", "somaLocation"])
    soma_by_body: dict[int, np.ndarray] = {}
    for body, location in zip(annotations["bodyId"], annotations["somaLocation"], strict=True):
        if location is None or (isinstance(location, float) and np.isnan(location)):
            continue
        array = np.asarray(location, dtype=np.float64)
        if array.shape == (3,) and np.all(np.isfinite(array)):
            soma_by_body[int(body)] = array

    coords = np.full((len(body_ids), 3), np.nan, dtype=np.float64)
    for index, body in enumerate(body_ids):
        location = soma_by_body.get(int(body))
        if location is not None:
            coords[index] = location

    valid = np.flatnonzero(np.all(np.isfinite(coords), axis=1))
    centered = coords[valid] - np.nanmedian(coords[valid], axis=0)
    scales = np.nanpercentile(np.abs(centered), 92, axis=0)
    scales[scales == 0] = 1.0
    normalized = centered / scales
    _, _, vt = np.linalg.svd(normalized, full_matrices=False)
    projected = normalized @ vt[:2].T

    # Stabilize the visual orientation: wider axis horizontal, upper lobe slightly higher.
    if np.nanstd(projected[:, 0]) < np.nanstd(projected[:, 1]):
        projected = projected[:, ::-1]
    if np.nanmedian(projected[:, 1]) > 0:
        projected[:, 1] *= -1

    qx = np.nanpercentile(np.abs(projected[:, 0]), 98)
    qy = np.nanpercentile(np.abs(projected[:, 1]), 98)
    projected[:, 0] /= qx if qx > 0 else 1.0
    projected[:, 1] /= qy if qy > 0 else 1.0

    xy = np.full((len(body_ids), 2), np.nan, dtype=np.float64)
    xy[valid] = projected
    return xy, valid, graph


def _point_to_pixel(point: np.ndarray, box: tuple[int, int, int, int]) -> tuple[float, float]:
    x = box[0] + (np.clip(point[0], -1, 1) + 1) * 0.5 * (box[2] - box[0])
    y = box[3] - (np.clip(point[1], -1, 1) + 1) * 0.5 * (box[3] - box[1])
    return float(x), float(y)


def _brain_static_layer(projected: np.ndarray, valid: np.ndarray, size=(340, 238)) -> Image.Image:
    w, h = size
    layer = Image.new("RGBA", size, (5, 9, 15, 222))
    draw = ImageDraw.Draw(layer, "RGBA")
    draw.rounded_rectangle((0, 0, w - 1, h - 1), radius=18, fill=(5, 9, 15, 222), outline=(135, 153, 180, 50), width=1)
    draw.text((17, 13), "MaleCNS live activity", font=_font(17, True), fill=(235, 239, 244, 238))
    draw.text((17, 36), "57,407 neurons | soma positions", font=_font(11), fill=(144, 158, 178, 205))
    box = (14, 58, w - 14, h - 12)

    # A clean soma cloud reads much better than a dense projected edge hairball.
    stride = max(1, len(valid) // 6200)
    for index in valid[::stride]:
        p = projected[index]
        if not np.all(np.isfinite(p)):
            continue
        x, y = _point_to_pixel(p, box)
        draw.ellipse((x - 0.75, y - 0.75, x + 0.75, y + 0.75), fill=(119, 154, 194, 72))
    return layer


def _overlay_brain(frame: Image.Image, static: Image.Image, projected: np.ndarray, brain: dict | None):
    panel = static.copy()
    if brain:
        indices = brain.get("indices", [])
        values = brain.get("values", [])
        w, h = panel.size
        box = (14, 58, w - 14, h - 12)
        glow = Image.new("RGBA", panel.size, (0, 0, 0, 0))
        cores = Image.new("RGBA", panel.size, (0, 0, 0, 0))
        gd = ImageDraw.Draw(glow, "RGBA")
        cd = ImageDraw.Draw(cores, "RGBA")
        max_value = max((abs(float(v)) for v in values), default=1.0) or 1.0
        for index, value in zip(indices, values, strict=True):
            if index >= len(projected) or not np.all(np.isfinite(projected[index])):
                continue
            x, y = _point_to_pixel(projected[index], box)
            strength = min(1.0, abs(float(value)) / max_value)
            color = (255, 166, 64) if float(value) >= 0 else (81, 197, 255)
            radius = 1.8 + 3.3 * strength
            gd.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(*color, int(60 + 115 * strength)))
            core = 0.8 + 1.25 * strength
            cd.ellipse((x - core, y - core, x + core, y + core), fill=(255, 247, 226, int(150 + 90 * strength)))
        panel.alpha_composite(glow.filter(ImageFilter.GaussianBlur(radius=4.0)))
        panel.alpha_composite(cores)
    frame.alpha_composite(panel, (WIDTH - panel.width - 22, 22))


def _camera_for(frame: dict) -> mujoco.MjvCamera:
    fly = np.asarray(env_to_world(frame["x"], frame["y"]), dtype=float)
    window = np.asarray([0.50, 0.0, 0.15], dtype=float)
    distance_to_window = float(np.linalg.norm(fly[:2] - window[:2]))
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = 0.78 * fly + 0.22 * window
    camera.lookat[2] = 0.13
    camera.distance = float(np.clip(0.48 + 0.16 * distance_to_window, 0.48, 0.62))
    camera.azimuth = 315.0
    camera.elevation = -18.0
    return camera


def _style_model(model: mujoco.MjModel):
    custom_prefixes = ("room_", "right_", "window_", "outside_", "fill_", "window_light")
    for geom_id in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
        if name.startswith(custom_prefixes):
            continue
        rgba = model.geom_rgba[geom_id]
        # Keep the red compound eyes; make the rest of FlyBody less toy-orange.
        if rgba[0] > 0.55 and rgba[1] > 0.20 and rgba[1] < 0.75 and not (rgba[0] > 0.75 and rgba[1] < 0.18):
            model.geom_rgba[geom_id, :3] = np.asarray([0.27, 0.18, 0.10], dtype=np.float32)


def _render_frame(renderer, model, data, raw_frame: dict, static_brain, projected, label: str, seed: int) -> np.ndarray:
    data.qpos[:3] = env_to_world(raw_frame["x"], raw_frame["y"])
    data.qpos[3:7] = yaw_quaternion(raw_frame["heading"])
    mujoco.mj_forward(model, data)
    renderer.update_scene(data, _camera_for(raw_frame))
    rgb = renderer.render()
    image = Image.fromarray(rgb).convert("RGBA")

    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")
    draw.rounded_rectangle((22, 22, 303, 102), radius=16, fill=(5, 8, 13, 176), outline=(255, 255, 255, 20), width=1)
    draw.text((40, 35), label, font=_font(23, True), fill=(246, 246, 243, 245))
    draw.text((40, 70), f"seed {seed} | step {raw_frame['step']}", font=_font(14), fill=(170, 181, 195, 230))
    if raw_frame.get("collision"):
        draw.rounded_rectangle((30, HEIGHT - 73, 156, HEIGHT - 28), radius=13, fill=(113, 33, 26, 198))
        draw.text((51, HEIGHT - 62), "COLLISION", font=_font(15, True), fill=(255, 231, 220, 248))
    image.alpha_composite(overlay)
    _overlay_brain(image, static_brain, projected, raw_frame.get("brain"))
    return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _audio_for_timeline(timeline: list[dict | None], fps: int, output: Path):
    sample_rate = 44100
    total = max(1, int(math.ceil(len(timeline) / fps * sample_rate)))
    frame_positions = np.minimum((np.arange(total) * fps / sample_rate).astype(int), len(timeline) - 1)
    speeds = np.asarray([0.0 if item is None else float(item.get("speed", 0.0)) for item in timeline], dtype=np.float64)
    speed = speeds[frame_positions]
    norm = np.clip(speed / 3.0, 0.0, 1.0)

    # Drosophila-like wing-buzz: fundamental around 200 Hz with rough harmonics and small modulation.
    frequency = 195.0 + 72.0 * norm
    phase = 2.0 * np.pi * np.cumsum(frequency) / sample_rate
    t = np.arange(total) / sample_rate
    buzz = (0.070 + 0.065 * norm) * (
        np.sin(phase) + 0.42 * np.sin(2.01 * phase + 0.2) + 0.18 * np.sin(3.03 * phase + 0.5)
    )
    buzz *= 0.84 + 0.16 * np.sin(2.0 * np.pi * 13.0 * t)

    rng = np.random.default_rng(20260914)
    audio = buzz.copy()
    last_collision_frame = -10_000
    cooldown = max(1, int(round(0.18 * fps)))
    for frame_index, item in enumerate(timeline):
        if not item or not item.get("collision") or frame_index - last_collision_frame < cooldown:
            continue
        last_collision_frame = frame_index
        start = int(frame_index / fps * sample_rate)
        length = min(int(0.18 * sample_rate), total - start)
        if length <= 0:
            continue
        tt = np.arange(length) / sample_rate
        thump = 0.52 * np.sin(2.0 * np.pi * 68.0 * tt) * np.exp(-24.0 * tt)
        thump += 0.12 * rng.normal(size=length) * np.exp(-34.0 * tt)
        audio[start:start + length] += thump

    pcm = np.asarray(np.clip(np.tanh(audio * 1.25), -1.0, 1.0) * 32767, dtype=np.int16)
    with wave.open(str(output), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())


def render(args):
    payload = json.loads(args.trajectory_json.read_text(encoding="utf-8"))
    projected, valid, _graph = _load_soma_projection(args.annotations, args.graph_npz)
    static_brain = _brain_static_layer(projected, valid)

    scene_path = _write_scene(args.flybody_dir)
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    _style_model(model)
    data = mujoco.MjData(model)
    if model.nkey:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    else:
        mujoco.mj_resetData(model, data)
    renderer = mujoco.Renderer(model, height=HEIGHT, width=WIDTH)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="fly-window-cinematic-v2-") as temp_dir:
        temp = Path(temp_dir)
        silent = temp / "silent.mp4"
        wav = temp / "audio.wav"
        writer = imageio_ffmpeg.write_frames(
            str(silent),
            (WIDTH, HEIGHT),
            fps=args.fps,
            codec="libx264",
            pix_fmt_in="rgb24",
            pix_fmt_out="yuv420p",
            ffmpeg_log_level="warning",
            output_params=["-crf", "19", "-preset", "medium", "-movflags", "+faststart"],
        )
        writer.send(None)
        timeline: list[dict | None] = []
        try:
            for trajectory in payload["trajectories"]:
                frames = trajectory["frames"]
                sampled = list(range(0, len(frames), args.stride))
                if sampled[-1] != len(frames) - 1:
                    sampled.append(len(frames) - 1)
                for index in sampled:
                    raw = frames[index]
                    writer.send(
                        _render_frame(renderer, model, data, raw, static_brain, projected, args.label, int(trajectory["seed"])).tobytes()
                    )
                    timeline.append(raw)
                for _ in range(args.hold_frames):
                    writer.send(
                        _render_frame(renderer, model, data, frames[-1], static_brain, projected, args.label, int(trajectory["seed"])).tobytes()
                    )
                    timeline.append(None)
        finally:
            writer.close()
            renderer.close()

        _audio_for_timeline(timeline, args.fps, wav)
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        subprocess.run(
            [ffmpeg, "-y", "-i", str(silent), "-i", str(wav), "-c:v", "copy", "-c:a", "aac", "-b:a", "160k", "-shortest", str(args.output)],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    print(json.dumps({"output": str(args.output), "frames": len(timeline), "fps": args.fps, "duration_seconds": len(timeline) / args.fps}, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Render Fly Window trajectories with FlyBody, MaleCNS activity, and audio.")
    parser.add_argument("--trajectory-json", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--graph-npz", type=Path, required=True)
    parser.add_argument("--flybody-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--fps", type=int, default=FPS)
    parser.add_argument("--hold-frames", type=int, default=10)
    args = parser.parse_args()
    if args.stride <= 0 or args.fps <= 0 or args.hold_frames < 0:
        parser.error("stride/fps must be positive and hold-frames non-negative")
    render(args)


if __name__ == "__main__":
    main()
