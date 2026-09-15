from __future__ import annotations

import math

import mujoco
import numpy as np

import render_cinematic_demo_v2 as base


def _camera_for(frame: dict) -> mujoco.MjvCamera:
    fly = np.asarray(base.env_to_world(frame["x"], frame["y"]), dtype=float)
    window = np.asarray([0.50, 0.0, 0.15], dtype=float)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = 0.72 * fly + 0.28 * window
    camera.lookat[2] = 0.13
    camera.distance = 0.98
    camera.azimuth = 45.0
    camera.elevation = -16.0
    return camera


def _joint_value(model: mujoco.MjModel, joint_name: str, phase_value: float, amplitude: float) -> tuple[int, float] | None:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if joint_id < 0:
        return None
    qpos_addr = int(model.jnt_qposadr[joint_id])
    limited = bool(model.jnt_limited[joint_id])
    if limited:
        low, high = (float(v) for v in model.jnt_range[joint_id])
        center = 0.5 * (low + high)
        half = 0.5 * (high - low)
        value = center + amplitude * half * phase_value
    else:
        value = amplitude * phase_value
    return qpos_addr, value


def _animate_wings(model: mujoco.MjModel, data: mujoco.MjData, step: int) -> None:
    # The physical wingbeat is ~200 Hz, far above a 24 fps video.  We render an
    # intentionally aliased visual flap so the viewer can see that the fly is flying,
    # while the audio carries the high-frequency buzz.
    phase = 2.0 * math.pi * (float(step) % 7.0) / 7.0
    channels = {
        "yaw": (math.sin(phase), 0.46),
        "roll": (math.sin(phase + 0.5 * math.pi), 0.32),
        "pitch": (math.sin(phase + 0.85), 0.24),
    }
    for side in ("left", "right"):
        for axis, (signal, amplitude) in channels.items():
            result = _joint_value(model, f"wing_{axis}_{side}", signal, amplitude)
            if result is not None:
                qpos_addr, value = result
                data.qpos[qpos_addr] = value


_original_render_frame = base._render_frame


def _render_frame(renderer, model, data, raw_frame, static_brain, projected, label, seed):
    _animate_wings(model, data, int(raw_frame.get("step", 0)))
    return _original_render_frame(renderer, model, data, raw_frame, static_brain, projected, label, seed)


base._camera_for = _camera_for
base._render_frame = _render_frame


if __name__ == "__main__":
    base.main()
