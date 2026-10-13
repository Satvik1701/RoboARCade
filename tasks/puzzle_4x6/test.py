"""Tests of the puzzle_4x6 task."""
import numpy as np
import pytest
from tasks.puzzle_4x6.dynamics import RGBA, toggle_matrix
from core.sim_env import SimEnv
import mujoco
from core.grade_episode import describe, make_checker
from helpers import (
    _pack_env,
    _press,
)


@pytest.fixture(scope="module")
def puzzle_4x6():
    env = SimEnv("puzzle_4x6", seed=0)
    env.reset(seed=0)
    return env


def board(env):
    return env.data.userdata[:24].astype(int).tolist()


def test_reset_redeals_deterministically_and_speaks_goal(puzzle_4x6):
    env = puzzle_4x6
    env.reset(seed=7)
    first = (board(env), env.language_instruction)
    env.reset(seed=7)
    assert (board(env), env.language_instruction) == first
    assert first[1].endswith("Make every button red.")
    env.reset(seed=8)
    assert board(env) != first[0]


def test_static_pack_description_says_goal_is_dealt():
    env = SimEnv("puzzle_4x6", seed=0)
    assert describe(env.spec["success"], env.task) == [
        (0, "Lights Out 4x6: every button red")]


def test_reset_paints_dealt_init_in_first_rendered_frame(puzzle_4x6):
    env = puzzle_4x6
    env.reset(seed=0)
    init = board(env)
    assert any(init)
    for i, state in enumerate(init):
        assert tuple(env.model.geom(f"btngeom_{i}").rgba) == pytest.approx(RGBA[state])

    rgb = env.observe_camera(depth=False)["agentview_image"]
    intrinsic = env.get_camera_intrinsics("agentview")
    world_to_camera = np.linalg.inv(env.get_camera_extrinsics("agentview"))
    screen_x = []
    for i, state in enumerate(init):
        world = np.r_[env.data.site(f"btntop_{i}").xpos, 1.0]
        camera = world_to_camera @ world
        pixel = intrinsic @ camera[:3]
        x, y = np.rint(pixel[:2] / pixel[2]).astype(int)
        patch = rgb[y - 2:y + 3, x - 2:x + 3].mean(axis=(0, 1))
        assert bool(patch[2] > patch[0]) is bool(state), (i, state, patch)
        screen_x.append(x)

    assert screen_x[0] < screen_x[5]


def test_lights_out_press_toggles_plus_shape():
    env = _pack_env("puzzle_4x6")
    before = board(env)
    _press(env, env.dynamics.control_step, "buttonbox_joint_8")
    changed = [i for i, (a, b) in enumerate(zip(before, board(env))) if a != b]
    assert changed == [2, 7, 8, 9, 14]
    for i in (0, 8):
        assert tuple(env.model.geom(f"btngeom_{i}").rgba) == pytest.approx(RGBA[board(env)[i]])


def test_lights_out_hold_is_one_toggle():
    env = _pack_env("puzzle_4x6")
    before = board(env)
    adr = env.model.joint("buttonbox_joint_0").qposadr[0]
    for _ in range(10):
        env.data.qpos[adr] = -0.024
        mujoco.mj_forward(env.model, env.data)
        env.dynamics.control_step(env)
    changed = [i for i, (a, b) in enumerate(zip(before, board(env))) if a != b]
    assert changed == [0, 1, 6]


def test_lights_out_solve_then_overshoot_and_replay_from_saved_state():
    env = _pack_env("puzzle_4x6")
    check, _ = make_checker(env.spec["success"], env.task)
    solution = (0, 5, 8, 9, 14, 15, 18, 23)
    press_bits = np.zeros(24, dtype=np.uint8)
    press_bits[list(solution)] = 1
    env.data.userdata[:24] = toggle_matrix(4, 6) @ press_bits % 2
    assert check(env) is False
    for i in solution:
        _press(env, env.dynamics.control_step, f"buttonbox_joint_{i}")
    assert check(env) is True
    solved = env.sim_state()
    _press(env, env.dynamics.control_step, "buttonbox_joint_0")
    assert check(env) is False
    assert board(env)[0] == 1
    env.reset_to_sim_state(solved)
    assert check(env) is True
    assert tuple(env.model.geom("btngeom_0").rgba) == pytest.approx(RGBA[0])
