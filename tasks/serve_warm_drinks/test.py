"""Tests of the serve_warm_drinks task.

"""


from types import SimpleNamespace
import json

import mujoco
import numpy as np
import pytest

from core.grade_episode import PressEdge, make_checker
from core.tasks import TASKS, load_scene
from helpers import _pack_env


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


@pytest.fixture(scope="module")
def button_model():
    spec = load_scene(TASKS / "serve_warm_drinks")
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    root = data.body("microwave_1_main")
    rotation = root.xmat.reshape(3, 3)
    finger = spec.worldbody.add_body(
        name="test_finger",
        pos=root.xpos + rotation @ [0.13, -0.1365, 0.16],
        gravcomp=1,
    )
    finger.add_joint(
        name="test_slide",
        type=mujoco.mjtJoint.mjJNT_SLIDE,
        axis=rotation @ [0.0, 1.0, 0.0],
        damping=1,
    )
    finger.add_geom(
        name="test_tip",
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[0.008, 0, 0],
        mass=0.02,
        contype=1,
        conaffinity=1,
        friction=[1, 0.3, 0.0001],
        solref=[0.004, 1],
        solimp=[0.998, 0.998, 0.001, 0.5, 2],
    )
    model = spec.compile()
    model.opt.timestep = 0.00025
    return model, rotation.copy(), model.body("test_finger").pos.copy()


@pytest.mark.parametrize(
    "offset",
    [(0, 0), (-0.008, -0.003), (-0.008, 0.003), (0.008, -0.003), (0.008, 0.003)],
)
def test_physical_press_and_spring_return(button_model, offset):
    model, rotation, origin = button_model
    model.body("test_finger").pos[:] = origin + rotation @ [offset[0], 0, offset[1]]
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    env = SimpleNamespace(model=model, data=data)
    edge = PressEdge("microwave_1_start_key")
    key = int(model.joint("microwave_1_start_key").qposadr[0])
    finger = model.joint("test_slide")
    qa, va = int(finger.qposadr[0]), int(finger.dofadr[0])
    tip = model.geom("test_tip").id
    cap = model.geom("microwave_1_start_key_collision").id
    count, stroke = 0, 0.0
    # A contact-driven probe, not a forced button qpos or joint force.
    for _ in range(2):
        for i in range(4000):
            target = min(0.026, i * model.opt.timestep * 0.04)
            data.qfrc_applied[va] = np.clip(
                600 * (target - data.qpos[qa]) - 3 * data.qvel[va], -2, 2
            )
            mujoco.mj_step(model, data)
            count += edge.update(env)
            stroke = max(stroke, -float(data.qpos[key]))
        assert any({c.geom1, c.geom2} == {tip, cap} for c in data.contact)
        assert all(
            cap in (c.geom1, c.geom2) for c in data.contact if tip in (c.geom1, c.geom2)
        ), "panel blocks fingertip travel"
        for _ in range(2000):
            data.qfrc_applied[va] = np.clip(
                -600 * data.qpos[qa] - 3 * data.qvel[va], -2, 2
            )
            mujoco.mj_step(model, data)
            count += edge.update(env)
        assert abs(data.qpos[key]) < 0.0001
    assert 0.0059 < stroke < 0.0061
    assert count == 2, "one edge per press, rearmed only after release"


@pytest.mark.parametrize(
    "mug_inside,door_closed", [(True, True), (True, False), (False, True)]
)
def test_real_warm_stage_requires_fresh_press_and_live_conditions(
    button_model, mug_inside, door_closed
):
    model, _, _ = button_model
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    env = SimpleNamespace(model=model, data=data)
    task = json.loads((TASKS / "serve_warm_drinks" / "task.json").read_text())
    check, _ = make_checker(task["success"]["of"][0], "serve_warm_drinks")
    mug = model.body("white_yellow_mug_1_main")
    qa = int(model.jnt_qposadr[mug.jntadr[0]])
    root = data.body("microwave_1_main")
    data.qpos[qa : qa + 3] = root.xpos + root.xmat.reshape(3, 3) @ [-0.04, 0, 0.03]
    data.qpos[qa + 3 : qa + 7] = [1, 0, 0, 0]
    if not mug_inside:
        data.qpos[qa] += 0.4
    door = int(model.joint("microwave_1_microjoint").qposadr[0])
    key = int(model.joint("microwave_1_start_key").qposadr[0])
    data.qpos[door] = 0 if door_closed else -0.5
    mujoco.mj_forward(model, data)
    assert not check(env)
    data.qpos[key] = -0.003
    mujoco.mj_forward(model, data)
    assert bool(check(env)) == (mug_inside and door_closed)
    # Closing the door while holding an already-pressed button must not start it.
    data.qpos[door] = 0
    mujoco.mj_forward(model, data)
    assert not check(env)
    data.qpos[key] = 0
    mujoco.mj_forward(model, data)
    assert not check(env)
    data.qpos[key] = -0.003
    mujoco.mj_forward(model, data)
    assert bool(check(env)) == mug_inside


def test_visible_panel_has_clearance_behind_entire_button(button_model):
    model, _, _ = button_model
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    root = data.body("microwave_1_main")
    rotation = root.xmat.reshape(3, 3)
    panel = model.geom("microwave_1_g0").id
    # Trace through the cap footprint. The first visible panel surface must be
    # behind the cap's back at full travel, including its soft-stop tolerance.
    for x in np.linspace(0.1101, 0.1499, 5):
        for z in np.linspace(0.1481, 0.1719, 5):
            origin = root.xpos + rotation @ [x, -0.14, z]
            distance = mujoco.mj_rayMesh(
                model, data, panel, origin, rotation @ [0.0, 1.0, 0.0]
            )
            assert distance < 0 or -0.14 + distance > -0.1044
