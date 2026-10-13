"""Tests of the square task."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R
from core.sim_env import SimEnv
from helpers import (
    POS_TOL,
)
import mujoco
from core.grade_episode import make_checker
from helpers import (
    set_free_body,
)


@pytest.mark.parametrize("deg", [0.0, 0.2, 45.0])
@pytest.mark.parametrize("dist", [0.0, 0.0005, 0.002, 0.01, 0.20])
def test_small_waypoints_do_not_break_the_pacer(dist, deg):
    env = SimEnv(task="square", seed=0)
    env.reset(seed=0, randomize_init=False)
    env.max_len = 10**9
    arm = env.arms[0]
    home = env.observe_proprio(arm)
    tgt = home.eef_pos + np.array([dist, 0.0, 0.0])
    rot = R.from_euler("z", deg, degrees=True) * R.from_euler("xyz", home.eef_euler)
    env.move_to(tgt, rot.as_euler("xyz"), 1.0)
    err = float(np.linalg.norm(env.observe_proprio(arm).eef_pos - tgt))
    assert err < POS_TOL, f"{dist * 1000:.1f} mm / {deg} deg waypoint landed {err * 1000:.1f} mm off"


@pytest.mark.parametrize("dist", [0.0005, 0.0010, 0.0015])
def test_sub_arrival_translation_actually_moves(dist):
    env = SimEnv(task="square", seed=0)
    env.reset(seed=0, randomize_init=False)
    env.max_len = 10**9
    arm = env.arms[0]
    start = env.observe_proprio(arm)
    steps_before = env.num_step
    target = start.eef_pos + np.array([dist, 0.0, 0.0])

    env.move_to(target, start.eef_euler, 1.0)

    delta = env.observe_proprio(arm).eef_pos - start.eef_pos
    assert env.num_step > steps_before, f"{dist * 1000:.1f} mm correction was a zero-step no-op"
    assert delta[0] > 0.5 * dist, (
        f"{dist * 1000:.1f} mm correction moved only {delta[0] * 1000:.2f} mm toward target")


def test_resending_reset_pose_is_zero_step():
    env = SimEnv(task="square", seed=0)
    env.reset(seed=0, randomize_init=False)
    env.max_len = 10**9
    arm = env.arms[0]
    start = env.observe_proprio(arm)
    steps_before = env.num_step

    env.move_to(start.eef_pos, start.eef_euler, 1.0)

    assert env.num_step == steps_before
    np.testing.assert_array_equal(env.observe_proprio(arm).eef_pos, start.eef_pos)


def test_square_success_tree():
    env = SimEnv("square", seed=0)
    env.reset(seed=0)
    check, hold = make_checker(env.spec["success"])
    assert hold == 1
    assert check(env) is False

    tree = {c["pred"]: c for c in env.spec["success"]["of"]}
    peg = tree["within_dist"]["b"]
    nut = tree["within_dist"]["a"]
    peg_pos = env.data.body(peg).xpos.copy()
    set_free_body(env, nut, [peg_pos[0], peg_pos[1], tree["z_below"]["z_below"] - 0.02])
    mujoco.mj_forward(env.model, env.data)
    assert check(env) is True

    set_free_body(env, nut, [peg_pos[0] + 0.05, peg_pos[1], tree["z_below"]["z_below"] - 0.02])
    mujoco.mj_forward(env.model, env.data)
    assert check(env) is False
