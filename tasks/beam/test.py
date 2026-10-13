"""Tests of the beam task."""
import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R
from core.grade_episode import make_checker, nodes
from core.tasks import TASKS
from core.sim_env import SimEnv
from core import grade_episode as predicates
from helpers import (
    _pack_env,
)


def test_fabrica_pose_within():
    if not (TASKS / "beam" / "task.json").exists():
        pytest.skip("fabrica/beam pack not materialized")
    env = SimEnv("beam", seed=0)
    env.reset(seed=0)
    success = env.spec["success"]
    check, _ = make_checker(success)
    spec = next(n for n in nodes(success) if n.get("pred") == "pose_within")
    assert check(env) is False

    qadr = env.model.joint(env.model.body(spec["obj"]).jntadr[0]).qposadr[0]
    ref = env.data.body(spec["rel_to"])
    goal_p = ref.xpos + ref.xmat.reshape(3, 3) @ np.asarray(spec["pos"])
    goal_q = np.empty(4)
    mujoco.mju_mulQuat(goal_q, ref.xquat, np.asarray(spec["quat"], dtype=float))
    env.data.qpos[qadr : qadr + 3] = goal_p
    env.data.qpos[qadr + 3 : qadr + 7] = goal_q
    mujoco.mj_forward(env.model, env.data)
    assert check(env) is True

    env.data.qpos[qadr] += 0.01
    mujoco.mj_forward(env.model, env.data)
    assert check(env) is False

    env.data.qpos[qadr] -= 0.01
    env.data.qpos[qadr + 3 : qadr + 7] = R.from_euler("z", 90, degrees=True).as_quat()[[3, 0, 1, 2]]
    mujoco.mj_forward(env.model, env.data)
    assert check(env) is False


def test_touching_part_survives_missing_sleep_contacts(monkeypatch):
    env = _pack_env("jenga_tower")
    block = "jenga_1"
    table = "play_table"
    check, _ = make_checker({"pred": "touching_part", "a": block, "b": table})
    assert check(env)
    monkeypatch.setattr(predicates, "_contact_body_pairs", lambda _env: iter(()))
    assert check(env)


def test_pose_within_lateral_limit_uses_reference_frame():
    from types import SimpleNamespace
    spec = dict(pred="pose_within", obj="part", rel_to="assembly", pos=[0, 0, 0],
                quat=[1, 0, 0, 0], pos_tol=0.005, pos_tol_xy=0.0006,
                rot_tol_deg=6, sym=2)
    rotation = R.from_euler("y", 60, degrees=True)
    ref = SimpleNamespace(xpos=np.array([0.2, 0.3, 0.4]), xmat=rotation.as_matrix(),
                          xquat=rotation.as_quat()[[3, 0, 1, 2]])

    def at(offset, angle=0):
        orient = rotation * R.from_euler("z", angle, degrees=True)
        part = SimpleNamespace(xpos=ref.xpos + ref.xmat @ np.asarray(offset),
                               xmat=orient.as_matrix(), xquat=orient.as_quat()[[3, 0, 1, 2]])
        return SimpleNamespace(data=SimpleNamespace(body=lambda name: ref if name == "assembly" else part))

    check = predicates._build(spec)
    assert check(at([0.00059, 0, 0.0012]))
    assert check(at([0, 0, 0.0049]))
    assert not check(at([0.00061, 0, 0]))
    assert not check(at([0.00045, 0.00045, 0]))
    assert not check(at([0, 0, 0.0051]))
    assert check(at([0, 0, 0], 5.9))
    assert not check(at([0, 0, 0], 6.1))
    assert check(at([0, 0, 0], 180))
    assert predicates._build({k: v for k, v in spec.items() if k != "pos_tol_xy"})(at([0.001, 0, 0]))


def test_fabrica_pose_within_axis_only_frees_the_spin():
    from types import SimpleNamespace
    spec = {"pred": "pose_within", "obj": "part0", "pos": [0.0, 0.0, 0.0], "quat": [1.0, 0.0, 0.0, 0.0],
            "pos_tol": 0.005, "rot_tol_deg": 15, "axis": [0.0, 0.0, 1.0]}

    def env_with(quat):
        m = np.empty(9)
        mujoco.mju_quat2Mat(m, np.asarray(quat, float))
        body = SimpleNamespace(xquat=np.asarray(quat, float), xmat=m, xpos=np.zeros(3))
        return SimpleNamespace(data=SimpleNamespace(body=lambda name: body), model=None)

    spin = R.from_euler("z", 90, degrees=True).as_quat()[[3, 0, 1, 2]]
    tilt = R.from_euler("x", 30, degrees=True).as_quat()[[3, 0, 1, 2]]
    check = predicates._build(spec)
    assert check(env_with([1.0, 0.0, 0.0, 0.0])) is True
    assert check(env_with(spin)) is True
    assert check(env_with(tilt)) is False
    assert predicates._build({**spec, "axis": None})(env_with(spin)) is False
    assert "spin free" in predicates.describe({"pred": "all", "of": [spec]})[1][1]
