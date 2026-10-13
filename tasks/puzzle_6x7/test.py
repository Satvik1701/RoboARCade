"""Tests of the puzzle_6x7 task."""
import mujoco
import numpy as np
import pytest
from core.sim_env import SimEnv


@pytest.fixture(scope="module")
def puzzle_6x7():
    env = SimEnv("puzzle_6x7", seed=0)
    env.reset(seed=0)
    return env


def press(env, button):
    adr = env.model.joint(f"buttonbox_joint_{button}").qposadr[0]
    env.data.qpos[adr] = 0.0
    mujoco.mj_forward(env.model, env.data)
    env.dynamics.control_step(env)
    before = env.data.userdata[:42].astype(np.uint8)
    env.data.qpos[adr] = -0.024
    mujoco.mj_forward(env.model, env.data)
    env.dynamics.control_step(env)
    after = env.data.userdata[:42].astype(np.uint8)
    env.data.qpos[adr] = 0.0
    mujoco.mj_forward(env.model, env.data)
    env.dynamics.control_step(env)
    return set(np.flatnonzero(before ^ after))


def test_6x7_language_camera_convention_and_corner_edge_toggles(puzzle_6x7):
    env = puzzle_6x7
    env.reset(seed=0)
    assert env.language_instruction.endswith("Make every button red.")

    intrinsic = env.get_camera_intrinsics("agentview")
    world_to_camera = np.linalg.inv(env.get_camera_extrinsics("agentview"))
    screen_x = []
    for i in (0, 6):
        camera = world_to_camera @ np.r_[env.data.site(f"btntop_{i}").xpos, 1.0]
        pixel = intrinsic @ camera[:3]
        screen_x.append(float(pixel[0] / pixel[2]))
    assert screen_x[0] < screen_x[1], "+y columns must read left-to-right in agentview"

    assert press(env, 0) == {0, 1, 7}
    assert press(env, 3) == {2, 3, 4, 10}
    assert press(env, 17) == {10, 16, 17, 18, 24}
    assert press(env, 41) == {34, 40, 41}
