import math

import numpy as np

from fly_window.rendering.cinematic import env_to_world, yaw_quaternion


def test_env_to_world_preserves_room_geometry():
    assert env_to_world(0.0, 0.0) == (-0.5, -0.3, 0.12)
    assert env_to_world(10.0, 6.0) == (0.5, 0.3, 0.12)
    assert env_to_world(10.0, 3.0) == (0.5, 0.0, 0.12)


def test_yaw_quaternion_is_unit_and_rotates_about_vertical_axis():
    q = np.asarray(yaw_quaternion(math.pi / 2), dtype=float)
    assert np.isclose(np.linalg.norm(q), 1.0)
    assert np.allclose(q, [math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)])
