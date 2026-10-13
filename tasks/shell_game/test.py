"""Tests of the shell_game task."""
import mujoco
import numpy as np
from core.grade_episode import make_checker
from helpers import (
    ROYGBIV,
    _pack_env,
    _press,
)
import logging
import pytest
from core.sim_env import SimEnv


def _rainbow(env, order):
    for i, n in enumerate(order):
        env.data.qpos[env._obj_qadr[n]:env._obj_qadr[n] + 3] = [0.18, -0.27 + 0.09 * i, 0.8167]
    mujoco.mj_forward(env.model, env.data)


def _shell(env, i, xy):
    a = env._obj_qadr[f"shell_{i}"]
    env.data.qpos[a:a + 3] = [*xy, 0.80]


def _cover(env, check):
    for i, n in enumerate(ROYGBIV):
        _shell(env, i, env.data.body(n).xpos[:2])
    mujoco.mj_forward(env.model, env.data)
    check(env)


def _uncover(env, check, order):
    for i, n in enumerate(order):
        _shell(env, ROYGBIV.index(n), [-0.30, -0.27 + 0.09 * i])
        mujoco.mj_forward(env.model, env.data)
        check(env)


def test_shell_game_needs_both_press_stages_then_the_rainbow():
    env = _pack_env("shell_game")
    check, _ = make_checker(env.spec["success"], env.task)
    _rainbow(env, ROYGBIV)
    assert check(env) is False
    _press(env, check, "buttonbox_joint_0")
    assert check(env) is False
    for _ in range(4):
        _press(env, check, "buttonbox_joint_1")
    assert check(env) is False
    _press(env, check, "buttonbox_joint_1")
    _rainbow(env, ROYGBIV)
    _cover(env, check)
    _uncover(env, check, ROYGBIV)
    check(env)
    assert check(env) is True


def test_shell_game_wrong_order_fails():
    env = _pack_env("shell_game")
    check, _ = make_checker(env.spec["success"], env.task)
    _press(env, check, "buttonbox_joint_0")
    for _ in range(5):
        _press(env, check, "buttonbox_joint_1")
    _rainbow(env, list(reversed(ROYGBIV)))
    check(env)
    assert check(env) is False


def _to_the_row(env, check):
    env.reset(seed=0)
    _press(env, check, "buttonbox_joint_0")
    _press(env, check, "buttonbox_joint_1", times=5)
    _rainbow(env, ROYGBIV)


def test_shell_game_row_never_uncovered_fails():
    env = _pack_env("shell_game")
    check, _ = make_checker(env.spec["success"], env.task)
    _to_the_row(env, check)
    _cover(env, check)
    assert check(env) is False
    assert check(env) is False


def test_shell_game_uncover_out_of_turn_fails():
    env = _pack_env("shell_game")
    check, _ = make_checker(env.spec["success"], env.task)
    _to_the_row(env, check)
    _cover(env, check)
    swapped = ROYGBIV[:5] + [ROYGBIV[6], ROYGBIV[5]]
    _uncover(env, check, swapped)
    check(env)
    assert check(env) is False


def test_shell_game_reveal_survives_a_shell_parked_alongside(env=None):
    env = _pack_env("shell_game")
    check, _ = make_checker(env.spec["success"], env.task)
    _to_the_row(env, check)
    _cover(env, check)
    _uncover(env, check, ROYGBIV)
    _shell(env, 6, env.data.body("cube_red").xpos[:2] + np.array([0.0, 0.03]))
    mujoco.mj_forward(env.model, env.data)
    check(env)
    assert check(env) is True


logging.getLogger("core.sim_env").setLevel(logging.ERROR)

TASK = "shell_game"
CUBES = ["cube_red", "cube_orange", "cube_yellow", "cube_green",
         "cube_blue", "cube_indigo", "cube_violet"]
GREEN, BLUE = "buttonbox_joint_0", "buttonbox_joint_1"


@pytest.fixture(scope="module")
def env():
    e = SimEnv(TASK, seed=7)
    e.reset(seed=7)
    return e


def _steps(e, seconds):
    for _ in range(round(seconds / e.model.opt.timestep)):
        e._step_physics()
        e._check(e)


def _push(e, joint):
    adr = e.model.joint(joint).qposadr[0]
    for _ in range(round(0.06 / e.model.opt.timestep)):
        e.data.qpos[adr] = -0.02
        e._step_physics()
        e._check(e)
    _steps(e, 0.08)


def _row(e):
    return sorted(CUBES, key=lambda n: float(e.data.body(n).xpos[1]))


def test_deal_is_seeded_and_permuted(env):
    assert _row(env) != CUBES, "seed 7 dealt the solved rainbow — the deal did nothing"
    e2 = SimEnv(TASK, seed=7); e2.reset(seed=7)
    assert _row(e2) == _row(env), "same seed must deal the same row"
    e3 = SimEnv(TASK, seed=8); e3.reset(seed=8)
    assert _row(e3) != _row(env), "a different seed dealt the identical row"


def test_green_covers_and_release_is_inert(env):
    _push(env, GREEN)
    _steps(env, 1.6)
    cover_x = float(env.spec["scene_script_gen"]["cover_x"])
    for i in range(7):
        p = env.data.body(f"shell_{i}").xpos
        assert abs(float(p[0]) - cover_x) < 0.01, f"shell_{i} never covered its slot"
    before = _row(env)
    _steps(env, 0.8)
    assert _row(env) == before, "the row moved with no button pressed"


def test_each_blue_press_swaps_exactly_two(env):
    prev = _row(env)
    for k in range(5):
        _push(env, BLUE)
        _steps(env, 1.8)
        cur = _row(env)
        moved = sum(a != b for a, b in zip(prev, cur))
        assert moved == 2, f"blue press {k+1} changed {moved} positions, not 2"
        prev = cur


def test_wrong_row_then_rainbow(env):
    def place(order):
        for i, n in enumerate(order):
            a = env.model.joint(env.model.body(n).jntadr[0]).qposadr[0]
            env.data.qpos[a:a + 3] = [0.18, -0.27 + 0.09 * i, 0.8167]
            env.data.qpos[a + 3:a + 7] = [1, 0, 0, 0]
        mujoco.mj_forward(env.model, env.data)
        _steps(env, 0.4)

    place(list(reversed(CUBES)))
    assert not bool(env._check(env)), "VIBGYOR graded as ROYGBIV"
    place(CUBES)
    assert bool(env._check(env)), "ROYGBIV row after both press stages must grade success"


@pytest.fixture()
def scripted():
    e = SimEnv("shell_game", seed=0)
    e.reset(seed=0)
    e.max_len = 10 ** 9
    return e


def dwell(env, seconds):
    t = float(env.data.time)
    while float(env.data.time) - t < seconds:
        env.apply_action(np.zeros(3), np.zeros(3), 1.0)


def press(env, joint):
    adr = env.model.joint(joint).qposadr[0]
    t = float(env.data.time)
    while float(env.data.time) - t < 0.06:
        env.data.qpos[adr] = -0.024
        env.apply_action(np.zeros(3), np.zeros(3), 1.0)
    dwell(env, 0.1)


def test_script_waits_for_its_trigger(scripted):
    env = scripted
    dwell(env, 8.0)
    for i in range(7):
        assert env.data.body(f"shell_{i}").xpos[0] == pytest.approx(-0.22, abs=2e-3)


def test_cover_runs_on_the_press_and_release_is_inert(scripted):
    env = scripted
    press(env, "buttonbox_joint_0")
    dwell(env, 1.8)
    for i in range(7):
        assert env.data.body(f"shell_{i}").xpos[0] == pytest.approx(-0.04, abs=2e-3)
    row0 = sorted(float(env.data.body(f"shell_{i}").xpos[1]) for i in range(7))
    dwell(env, 4.0)
    row1 = sorted(float(env.data.body(f"shell_{i}").xpos[1]) for i in range(7))
    assert np.allclose(row0, row1, atol=2e-3)


def test_each_press_edge_runs_one_swap_and_the_last_hands_back(scripted):
    env = scripted
    press(env, "buttonbox_joint_0")
    dwell(env, 1.8)
    for k in range(5):
        before = {i: float(env.data.body(f"shell_{i}").xpos[1]) for i in range(7)}
        press(env, "buttonbox_joint_1")
        dwell(env, 2.0)
        after = {i: float(env.data.body(f"shell_{i}").xpos[1]) for i in range(7)}
        moved = sum(abs(before[i] - after[i]) > 0.02 for i in range(7))
        assert moved == 2, f"press {k + 1} moved {moved} shells, not the swapped pair"
    env.data.qpos[env._obj_qadr["shell_0"] + 2] = 0.95
    mujoco.mj_forward(env.model, env.data)
    dwell(env, 0.5)
    assert env.data.body("shell_0").xpos[2] < 0.9, "script re-pinned a released shell"


def test_pieces_stay_pinned_until_their_last_leg(scripted):
    env = scripted
    press(env, "buttonbox_joint_0")
    dwell(env, 1.8)
    pinned = next(b for b, segs in env.dynamics.script.items()
                  if b.startswith("shell_") and env.dynamics.epoch(env, segs[-1]) is None)
    adr = env._obj_qadr[pinned]
    env.data.qpos[adr + 2] = 0.95
    mujoco.mj_forward(env.model, env.data)
    dwell(env, 0.3)
    assert env.data.body(pinned).xpos[2] < 0.85, f"{pinned} should be script-held"


def test_script_does_not_sag_between_control_steps(scripted):
    env = scripted
    press(env, "buttonbox_joint_0")
    t0 = env.dynamics.pressed["buttonbox_joint_0"][0]
    dwell(env, max(0.0, t0 + 0.35 - float(env.data.time)))
    for _ in range(5):
        z0 = env.data.body("shell_0").xpos[2]
        env.apply_action(np.zeros(3), np.zeros(3), 1.0)
        t = float(env.data.time) - t0 - env.model.opt.timestep
        want = 0.800 + (0.865 - 0.800) * (t - 0.3) / 0.4
        got = env.data.body("shell_0").xpos[2]
        assert got > z0, f"shell fell during the carry: {z0:.4f} -> {got:.4f}"
        assert abs(got - want) < 1e-4, f"t={t:.2f} off the ramp by {(got - want) * 1000:.2f} mm"


def test_a_reset_rearms_the_schedule(scripted):
    env = scripted
    press(env, "buttonbox_joint_0")
    dwell(env, 2.0)
    row = sorted(float(env.data.body(n).xpos[1])
                 for n in env.spec["scene_script_gen"]["cubes"])
    env.reset(seed=0)
    assert env.dynamics.pressed == {}
    assert float(env.data.time) - env.episode_t0 == pytest.approx(0.0, abs=1e-9)
    for i in range(7):
        assert env.data.body(f"shell_{i}").xpos[0] == pytest.approx(-0.22, abs=2e-3)
    row2 = sorted(float(env.data.body(n).xpos[1])
                  for n in env.spec["scene_script_gen"]["cubes"])
    assert np.allclose(row, row2, atol=1e-6), "same seed must re-deal the same hand"


def test_trigger_leg_outranks_clock_legs(scripted):
    env = scripted
    body, btn = "shell_0", "buttonbox_joint_0"
    env.dynamics.build(env, [
        {"body": body, "from": 0.5, "to": 1.5, "pos": [-0.16, 0.10, 0.80]},
        {"body": body, "from": 3.0, "to": 4.0, "pos": [-0.16, -0.27, 0.80]},
        {"body": body, "after": [btn], "from": 0.0, "to": 1e9, "pos": None},
    ])
    dwell(env, 1.6)
    assert env.data.body(body).xpos[1] == pytest.approx(0.10, abs=5e-3)
    press(env, btn)
    y = float(env.data.body(body).xpos[1])
    dwell(env, 4.0)
    assert env.data.body(body).xpos[1] == pytest.approx(y, abs=5e-3), \
        "a clock leg that opened after the press stole the body from the stop hold"


def _put(env, body, pos):
    j = env.model.joint(env.model.body(body).jntadr[0])
    env.data.qpos[j.qposadr[0]:j.qposadr[0] + 3] = pos
    env.data.qvel[j.dofadr[0]:j.dofadr[0] + 6] = 0.0
    mujoco.mj_forward(env.model, env.data)


def test_press_count_needs_a_release_between_presses():
    env = _pack_env("shell_game")
    check, _ = make_checker({"pred": "press_count", "joint": "buttonbox_joint_0", "count": 1})
    j = env.model.joint("buttonbox_joint_0")
    adr, lo = j.qposadr[0], env.model.jnt_range[j.id][0]
    for _ in range(20):
        env.data.qpos[adr] = lo
        mujoco.mj_forward(env.model, env.data)
        assert check(env) is True
    _press(env, check, "buttonbox_joint_0")
    assert check(env) is False


def test_lateral_order_needs_the_gap_and_the_row():
    env = _pack_env("shell_game")
    _put(env, "cube_red", [0.02, -0.18, 0.8167])
    _put(env, "cube_orange", [0.02, -0.06, 0.8167])
    fwd = {"pred": "lateral_order", "first": "cube_red", "second": "cube_orange",
           "axis": "y", "min_gap": 0.05, "max_gap": 0.20, "align_tol": 0.05}
    check, _ = make_checker(fwd, "shell_game")
    assert check(env) is True
    back, _ = make_checker({**fwd, "first": "cube_orange", "second": "cube_red"}, "shell_game")
    assert back(env) is False
    _put(env, "cube_orange", [0.30, -0.06, 0.8167])
    assert check(env) is False
