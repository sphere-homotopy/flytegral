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
WINDOW_WORLD_Y_HALF = 0.075
WINDOW_BOTTOM = 0.045
WINDOW_TOP = 0.235


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
    scene = flybody_dir / "fly_window_cinematic.xml"
    scene.write_text(
        f'''<mujoco model="fly window cinematic">
  <include file="fruitfly.xml"/>
  <visual>
    <global offwidth="{WIDTH}" offheight="{HEIGHT}" azimuth="135" elevation="-20"/>
    <quality shadowsize="4096" offsamples="4"/>
    <headlight ambient="0.18 0.17 0.16" diffuse="0.42 0.40 0.38" specular="0.08 0.08 0.08"/>
  </visual>
  <statistic meansize="0.03" extent="1.15" center="0 0 0.13"/>
  <asset>
    <texture name="room_sky" type="skybox" builtin="gradient" rgb1="0.035 0.045 0.06" rgb2="0.008 0.009 0.012" width="256" height="256"/>
    <material name="floor_m" rgba="0.095 0.085 0.075 1" roughness="0.9"/>
    <material name="wall_m" rgba="0.16 0.14 0.12 1" roughness="0.95"/>
    <material name="frame_m" rgba="0.055 0.045 0.04 1" roughness="0.55"/>
    <material name="outside_m" rgba="0.72 0.88 1.0 1" emission="0.85"/>
  </asset>
  <worldbody>
    <geom name="room_floor" type="box" pos="0 0 -0.012" size="0.53 0.33 0.012" material="floor_m" contype="0" conaffinity="0"/>
    <geom name="room_left" type="box" pos="-0.515 0 0.15" size="0.015 0.315 0.15" material="wall_m" contype="0" conaffinity="0"/>
    <geom name="room_back" type="box" pos="0 0.315 0.15" size="0.53 0.015 0.15" material="wall_m" contype="0" conaffinity="0"/>
    <geom name="room_front" type="box" pos="0 -0.315 0.15" size="0.53 0.015 0.15" material="wall_m" contype="0" conaffinity="0"/>
    <geom name="right_lower_y" type="box" pos="0.515 -0.195 0.15" size="0.015 0.12 0.15" material="wall_m" contype="0" conaffinity="0"/>
    <geom name="right_upper_y" type="box" pos="0.515 0.195 0.15" size="0.015 0.12 0.15" material="wall_m" contype="0" conaffinity="0"/>
    <geom name="window_sill" type="box" pos="0.515 0 0.0225" size="0.015 {WINDOW_WORLD_Y_HALF} 0.0225" material="frame_m" contype="0" conaffinity="0"/>
    <geom name="window_header" type="box" pos="0.515 0 0.2675" size="0.015 {WINDOW_WORLD_Y_HALF} 0.0325" material="frame_m" contype="0" conaffinity="0"/>
    <geom name="window_frame_low" type="box" pos="0.505 -{WINDOW_WORLD_Y_HALF} 0.14" size="0.012 0.008 0.10" material="frame_m" contype="0" conaffinity="0"/>
    <geom name="window_frame_high" type="box" pos="0.505 {WINDOW_WORLD_Y_HALF} 0.14" size="0.012 0.008 0.10" material="frame_m" contype="0" conaffinity="0"/>
    <geom name="outside_glow" type="box" pos="0.545 0 0.14" size="0.006 0.072 0.092" material="outside_m" contype="0" conaffinity="0"/>
    <light name="window_light" pos="0.40 0 0.20" dir="-1 0 -0.12" directional="true" diffuse="1.15 1.12 1.05" specular="0.18 0.18 0.18" castshadow="true"/>
    <light name="fill_light" pos="-0.25 -0.18 0.28" dir="0.5 0.25 -0.35" directional="true" diffuse="0.20 0.22 0.28" specular="0.05 0.05 0.07"/>
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
    scales = np.nanpercentile(np.abs(centered), 90, axis=0)
    scales[scales == 0] = 1.0
    normalized = centered / scales
    _, _, vt = np.linalg.svd(normalized, full_matrices=False)
    projected = normalized @ vt[:2].T
    qx = np.nanpercentile(np.abs(projected[:, 0]), 98)
    qy = np.nanpercentile(np.abs(projected[:, 1]), 98)
    qx = qx if qx > 0 else 1.0
    qy = qy if qy > 0 else 1.0
    projected[:, 0] /= qx
    projected[:, 1] /= qy

    xy = np.full((len(body_ids), 2), np.nan, dtype=np.float64)
    xy[valid] = projected
    return xy, valid, graph


def _brain_static_layer(projected: np.ndarray, valid: np.ndarray, graph, size=(390, 285)) -> Image.Image:
    w, h = size
    layer = Image.new("RGBA", size, (7, 10, 16, 224))
    draw = ImageDraw.Draw(layer, "RGBA")
    draw.rounded_rectangle((0, 0, w - 1, h - 1), radius=22, fill=(7, 10, 16, 224), outline=(130, 145, 170, 55), width=1)
    draw.text((20, 15), "MaleCNS · live activity", font=_font(18, True), fill=(235, 238, 242, 235))
    draw.text((20, 40), "57,407 neurons · real soma coordinates", font=_font(13), fill=(150, 160, 177, 210))
    box = (18, 66, w - 18, h - 16)

    def to_pixel(point):
        x = box[0] + (np.clip(point[0], -1, 1) + 1) * 0.5 * (box[2] - box[0])
        y = box[3] - (np.clip(point[1], -1, 1) + 1) * 0.5 * (box[3] - box[1])
        return (float(x), float(y))

    src = graph["src_idx"].astype(np.int64)
    dst = graph["dst_idx"].astype(np.int64)
    weights = graph["weights"].astype(np.int64)
    edge_count = min(700, len(weights))
    if edge_count:
        chosen = np.argpartition(weights, -edge_count)[-edge_count:]
        for edge in chosen:
            a, b = int(src[edge]), int(dst[edge])
            if np.all(np.isfinite(projected[a])) and np.all(np.isfinite(projected[b])):
                draw.line((*to_pixel(projected[a]), *to_pixel(projected[b])), fill=(86, 114, 151, 18), width=1)

    stride = max(1, len(valid) // 8500)
    for index in valid[::stride]:
        x, y = to_pixel(projected[index])
        draw.ellipse((x - 1, y - 1, x + 1, y + 1), fill=(125, 155, 195, 72))
    return layer


def _overlay_brain(frame: Image.Image, static: Image.Image, projected: np.ndarray, brain: dict | None):
    panel = static.copy()
    if brain:
        indices = brain.get("indices", [])
        values = brain.get("values", [])
        w, h = panel.size
        box = (18, 66, w - 18, h - 16)
        glow = Image.new("RGBA", panel.size, (0, 0, 0, 0))
        cores = Image.new("RGBA", panel.size, (0, 0, 0, 0))
        gd = ImageDraw.Draw(glow, "RGBA")
        cd = ImageDraw.Draw(cores, "RGBA")
        max_value = max((abs(float(v)) for v in values), default=1.0) or 1.0
        for index, value in zip(indices, values, strict=True):
            if index >= len(projected) or not np.all(np.isfinite(projected[index])):
                continue
            p = projected[index]
            x = box[0] + (np.clip(p[0], -1, 1) + 1) * 0.5 * (box[2] - box[0])
            y = box[3] - (np.clip(p[1], -1, 1) + 1) * 0.5 * (box[3] - box[1])
            strength = min(1.0, abs(float(value)) / max_value)
            if float(value) >= 0:
                color = (255, 155, 48)
            else:
                color = (65, 195, 255)
            radius = 3.0 + 5.0 * strength
            gd.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(*color, int(90 + 120 * strength)))
            core = 1.2 + 1.8 * strength
            cd.ellipse((x - core, y - core, x + core, y + core), fill=(255, 245, 220, int(170 + 80 * strength)))
        glow = glow.filter(ImageFilter.GaussianBlur(radius=5.5))
        panel.alpha_composite(glow)
        panel.alpha_composite(cores)
    frame.alpha_composite(panel, (WIDTH - panel.width - 24, 24))


def _camera_for(frame: dict) -> mujoco.MjvCamera:
    fly = np.asarray(env_to_world(frame["x"], frame["y"]), dtype=float)
    window = np.asarray([0.5, 0.0, 0.14], dtype=float)
    distance_to_window = float(np.linalg.norm(fly[:2] - window[:2]))
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = 0.68 * fly + 0.32 * window
    camera.distance = float(np.clip(0.38 + 0.50 * distance_to_window, 0.42, 0.72))
    camera.azimuth = 138.0
    camera.elevation = -24.0
    return camera


def _style_model(model: mujoco.MjModel):
    custom_prefixes = ("room_", "right_", "window_", "outside_", "fill_", "window_light")
    for geom_id in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
        if name.startswith(custom_prefixes):
            continue
        rgba = model.geom_rgba[geom_id]
        # FlyBody's stock orange is useful structurally but too toy-like for the cinematic shot.
        if rgba[0] > 0.55 and rgba[1] > 0.20 and rgba[1] < 0.75:
            model.geom_rgba[geom_id, :3] = np.asarray([0.23, 0.16, 0.10], dtype=np.float32)


def _render_frame(renderer, model, data, raw_frame: dict, static_brain, projected, label: str, seed: int) -> np.ndarray:
    world = env_to_world(raw_frame["x"], raw_frame["y"])
    data.qpos[:3] = world
    data.qpos[3:7] = yaw_quaternion(raw_frame["heading"])
    mujoco.mj_forward(model, data)
    renderer.update_scene(data, _camera_for(raw_frame))
    rgb = renderer.render()
    image = Image.fromarray(rgb).convert("RGBA")

    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")
    draw.rounded_rectangle((24, 24, 320, 112), radius=18, fill=(6, 8, 12, 180), outline=(255, 255, 255, 24), width=1)
    draw.text((43, 39), label, font=_font(25, True), fill=(246, 245, 241, 245))
    draw.text((43, 76), f"fixed seed {seed}  ·  step {raw_frame['step']}", font=_font(15), fill=(174, 181, 192, 235))
    if raw_frame.get("collision"):
        draw.rounded_rectangle((38, HEIGHT - 88, 190, HEIGHT - 38), radius=14, fill=(120, 36, 26, 205))
        draw.text((60, HEIGHT - 75), "COLLISION", font=_font(17, True), fill=(255, 232, 220, 250))
    image.alpha_composite(overlay)
    _overlay_brain(image, static_brain, projected, raw_frame.get("brain"))
    return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _audio_for_timeline(timeline: list[dict | None], fps: int, output: Path):
    sample_rate = 44100
    duration = len(timeline) / fps
    total = max(1, int(math.ceil(duration * sample_rate)))
    frame_positions = np.minimum((np.arange(total) * fps / sample_rate).astype(int), len(timeline) - 1)
    speeds = np.asarray([0.0 if item is None else float(item.get("speed", 0.0)) for item in timeline], dtype=np.float64)
    speed = speeds[frame_positions]
    norm = np.clip(speed / 3.0, 0.0, 1.0)
    frequency = 175.0 + 95.0 * norm
    phase = 2.0 * np.pi * np.cumsum(frequency) / sample_rate
    t = np.arange(total) / sample_rate
    buzz = (0.075 + 0.085 * norm) * (np.sin(phase) + 0.33 * np.sin(2.02 * phase) + 0.14 * np.sin(3.01 * phase))
    buzz *= 0.82 + 0.18 * np.sin(2.0 * np.pi * 17.0 * t)

    rng = np.random.default_rng(20260914)
    audio = buzz
    for frame_index, item in enumerate(timeline):
        if not item or not item.get("collision"):
            continue
        start = int(frame_index / fps * sample_rate)
        length = min(int(0.16 * sample_rate), total - start)
        if length <= 0:
            continue
        tt = np.arange(length) / sample_rate
        thump = 0.48 * np.sin(2.0 * np.pi * 74.0 * tt) * np.exp(-26.0 * tt)
        thump += 0.16 * rng.normal(size=length) * np.exp(-38.0 * tt)
        audio[start:start + length] += thump
    audio = np.tanh(audio * 1.3)
    pcm = np.asarray(np.clip(audio, -1.0, 1.0) * 32767, dtype=np.int16)
    with wave.open(str(output), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())


def render(args):
    data = json.loads(args.trajectory_json.read_text(encoding="utf-8"))
    projected, valid, graph = _load_soma_projection(args.annotations, args.graph_npz)
    static_brain = _brain_static_layer(projected, valid, graph)

    scene_path = _write_scene(args.flybody_dir)
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    _style_model(model)
    mjdata = mujoco.MjData(model)
    if model.nkey:
        mujoco.mj_resetDataKeyframe(model, mjdata, 0)
    else:
        mujoco.mj_resetData(model, mjdata)
    renderer = mujoco.Renderer(model, height=HEIGHT, width=WIDTH)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="fly-window-cinematic-") as temp_dir:
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
            for trajectory in data["trajectories"]:
                frames = trajectory["frames"]
                sampled = list(range(0, len(frames), args.stride))
                if sampled[-1] != len(frames) - 1:
                    sampled.append(len(frames) - 1)
                for index in sampled:
                    raw = frames[index]
                    writer.send(_render_frame(renderer, model, mjdata, raw, static_brain, projected, args.label, int(trajectory["seed"])).tobytes())
                    timeline.append(raw)
                for _ in range(args.hold_frames):
                    writer.send(_render_frame(renderer, model, mjdata, frames[-1], static_brain, projected, args.label, int(trajectory["seed"])).tobytes())
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
    parser = argparse.ArgumentParser(description="Render recorded Fly Window trajectories as a cinematic FlyBody video.")
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
