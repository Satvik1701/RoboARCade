"""All eight tools physically seat and ride in their matching pockets.

These are asset-fit checks using prescribed poses, not agent rollout tests.
"""

import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from core.grade_episode import make_checker
from core.sim_env import SimEnv


def tool_goals(node):
    goals = {}
    if node["pred"] == "pose_within":
        goals[node["obj"]] = node
    for child in node.get("of", []):
        goals.update(tool_goals(child))
    return goals


@pytest.fixture(scope="module")
def fit_env():
    return SimEnv("pack_tools_into_toolboxes", seed=1000000)


@pytest.mark.parametrize("yaw", [0, 90, 180, 270])
def test_every_tool_seats_and_stays_in_moving_box(fit_env, yaw):
    env = fit_env
    env.reset(seed=1000000 + yaw // 90)
    model, data = env.model, env.data
    goals = tool_goals(env.spec["success"])
    assert len(goals) == 8
    env.dynamics.spawn_schedule = ()
    for i, y in enumerate([-0.26, 0.26]):
        a = env._obj_qadr[f"toolbox{i}"]
        data.qpos[a : a + 3] = [0.065, y, 0.8401]
        data.qpos[a + 3 : a + 7] = Rotation.from_euler(
            "z", yaw, degrees=True
        ).as_quat()[[3, 0, 1, 2]]
    mujoco.mj_forward(model, data)

    def place_tools(dz):
        for name, goal in goals.items():
            box = data.body(goal["rel_to"])
            a = env._obj_qadr[name]
            data.qpos[a : a + 3] = (
                box.xpos + box.xmat.reshape(3, 3) @ np.array(goal["pos"]) + [0, 0, dz]
            )
            mujoco.mju_mulQuat(
                data.qpos[a + 3 : a + 7], box.xquat, np.array(goal["quat"])
            )
        data.qvel[:] = 0
        mujoco.mj_forward(model, data)

    # Check the entire descent, not just the final grader tolerance.
    for dz in np.linspace(0.05, 0, 51):
        place_tools(dz)
        for contact in data.contact:
            a, b = [
                model.body(int(model.geom_bodyid[g])).name
                for g in (contact.geom1, contact.geom2)
            ]
            if (a in goals and b == goals[a]["rel_to"]) or (
                b in goals and a == goals[b]["rel_to"]
            ):
                assert contact.dist > -0.0001, f"{a}/{b}: blocked insertion at {dz:g} m"

    place_tools(0.005)
    mujoco.mj_step(model, data, nstep=round(2 / model.opt.timestep))

    def check_seated():
        for name, goal in goals.items():
            predicate = {
                "pred": "all",
                "of": [goal, {"pred": "in_bbox", "a": name, "b": goal["rel_to"]}],
            }
            assert make_checker(predicate)[0](
                env
            ), f"{name} is not seated at box yaw {yaw}"
            tool, box = model.body(name).id, model.body(goal["rel_to"]).id
            assert any(
                {int(model.geom_bodyid[c.geom1]), int(model.geom_bodyid[c.geom2])}
                == {tool, box}
                for c in data.contact
            ), f"{name} has no supporting box contact"

    check_seated()
    before = {name: data.body(name).xpos.copy() for name in ("toolbox0", "toolbox1")}
    for _ in range(round(3 / model.opt.timestep)):
        env._step_physics()
    check_seated()
    for name, start in before.items():
        assert start[1] - data.body(name).xpos[1] == pytest.approx(0.008 * 3, abs=0.003)
