"""Tests of the serve_warm_drinks task.

"""

import mujoco
from core.grade_episode import make_checker
from helpers import (
    _pack_env,
)



def test_press_edge_latches_only_with_live_conditions():
    env = _pack_env("shell_game")
    key = env.model.joint("buttonbox_joint_0")
    door = env.model.joint("buttonbox_joint_1")
    key_adr, key_lo = key.qposadr[0], env.model.jnt_range[key.id][0]
    door_adr, door_lo = door.qposadr[0], env.model.jnt_range[door.id][0]
    closed = {"pred": "joint_in", "joint": "buttonbox_joint_1", "lo": -0.001,
              "hi": 0.001}
    check, _ = make_checker({"pred": "sequence", "of": [
        {"pred": "all", "of": [closed,
                                  {"pred": "press_edge", "joint": "buttonbox_joint_0"}]},
        closed,
    ]}, "serve_warm_drinks")

    for door_q in (door_lo, 0.0):
        env.data.qpos[door_adr] = door_q
        mujoco.mj_forward(env.model, env.data)
        assert check(env) is False

    env.data.qpos[door_adr] = door_lo
    for key_q in (key_lo, key_lo, 0.0):
        env.data.qpos[key_adr] = key_q
        mujoco.mj_forward(env.model, env.data)
        assert check(env) is False
    env.data.qpos[door_adr] = 0.0
    mujoco.mj_forward(env.model, env.data)
    assert check(env) is False

    env.data.qpos[key_adr] = key_lo
    mujoco.mj_forward(env.model, env.data)
    assert check(env) is False
    env.data.qpos[key_adr] = 0.0
    mujoco.mj_forward(env.model, env.data)
    assert check(env) is True
