"""Tests of the pack_tools_into_toolboxes task."""


from scipy.spatial.transform import Rotation
import mujoco
import numpy as np
import pytest

from core.grade_episode import make_checker
from core.sim_env import SimEnv


@pytest.fixture()
def toolbox_conveyor():
    e = SimEnv("pack_tools_into_toolboxes", seed=0)
    e.reset(seed=0)
    e.max_len = 10 ** 9
    return e


def _put_tool_past_conveyor_end(env, name):
    axis = np.asarray(env.dynamics.cfg["axis"], float)
    axis /= np.linalg.norm(axis)
    a, dof = env._obj_qadr[name], env._obj_dofadr[name]
    kind = name[:-1]
    pose = np.asarray(env.dynamics.cfg["spawn"]["queue"]["poses"][f"{kind}1"], float).copy()
    pose[:3] += axis * (float(env.dynamics.cfg["end"]) + 0.005 - pose[:3] @ axis)
    pose[2] -= 0.0005
    env.data.qpos[a:a + 7] = pose
    env.data.qvel[dof:dof + 6] = 0.0
    mujoco.mj_forward(env.model, env.data)
    belt = set(map(int, env._subtree_geoms(env.dynamics.cfg["body"])))
    tool = set(map(int, env._subtree_geoms(name)))
    assert any((int(c.geom1) in belt and int(c.geom2) in tool)
               or (int(c.geom2) in belt and int(c.geom1) in tool)
               for c in env.data.contact[:env.data.ncon])


def test_toolbox_conveyor_remove_is_one_way(toolbox_conveyor):
    env, name = toolbox_conveyor, "hammer1"
    park = env.dynamics.exit_poses[name].copy()
    distance = next(d for d, body, _pose, _park in env.dynamics.spawn_schedule
                    if body == name)
    span = float(env.dynamics.cfg["end"]) - float(env.dynamics.cfg["start"])
    env.data.time = env.episode_t0 + (distance + span + 0.01) / env.dynamics.cfg["speed"]
    _put_tool_past_conveyor_end(env, name)

    env._step_physics()
    np.testing.assert_allclose(env.data.qpos[env._obj_qadr[name]:env._obj_qadr[name] + 7],
                               park, atol=2e-3)
    for _ in range(50):
        env._step_physics()
    np.testing.assert_allclose(env.data.qpos[env._obj_qadr[name]:env._obj_qadr[name] + 3],
                               park[:3], atol=2e-3)


def test_toolbox_conveyor_seated_rider_leaves_with_box(toolbox_conveyor):
    env = toolbox_conveyor
    env.dynamics.cfg["speed"] = 0.01
    axis = np.asarray(env.dynamics.cfg["axis"], float)
    axis /= np.linalg.norm(axis)
    tray, tool = "toolbox0", "tape_measure1"
    ta, td = env._obj_qadr[tray], env._obj_dofadr[tray]
    tray_pose = env.model.qpos0[ta:ta + 7].copy()
    tray_pose[:3] += axis * (float(env.dynamics.cfg["end"]) - 0.050
                             - float(tray_pose[:3] @ axis))
    tray_pose[2] += 0.005
    env.data.qpos[ta:ta + 7] = tray_pose
    env.data.qvel[td:td + 6] = 0.0

    pa, pd = env._obj_qadr[tool], env._obj_dofadr[tool]
    tool_pose = np.asarray(
        env.dynamics.cfg["spawn"]["queue"]["poses"][tool], float).copy()
    tool_pose[:2] = tray_pose[:2]
    tool_pose[:3] += axis * 0.075
    env.data.qpos[pa:pa + 7] = tool_pose
    env.data.qvel[pd:pd + 6] = 0.0
    mujoco.mj_forward(env.model, env.data)
    assert tool in env.dynamics.carry_group(env, tray, set())
    belt = set(map(int, env._subtree_geoms(env.dynamics.cfg["body"])))
    root = env.model.body_rootid
    riding = set()
    for con in env.data.contact[:env.data.ncon]:
        g1, g2 = int(con.geom1), int(con.geom2)
        if (g1 in belt) != (g2 in belt):
            riding.add(int(root[env.model.geom_bodyid[g2 if g1 in belt else g1]]))
    assert env.model.body(tool).id not in riding
    assert env.model.body(tray).id not in riding

    rel = env.data.qpos[pa:pa + 3] - env.data.qpos[ta:ta + 3]
    env.dynamics.physics_step(env)
    force = env.data.xfrc_applied[env.model.body(tool).id, :3]
    assert float(force @ axis) == pytest.approx(0.0, abs=1e-12)
    np.testing.assert_allclose(
        env.data.qpos[pa:pa + 3] - env.data.qpos[ta:ta + 3], rel, atol=1e-9)
    assert np.linalg.norm(env.data.qpos[pa:pa + 3]
                          - env.dynamics.exit_poses[tool][:3]) > 1.0

    env.data.qpos[ta + 2] -= 0.005
    rel = env.data.qpos[pa:pa + 3] - env.data.qpos[ta:ta + 3]
    shift = axis * 0.055
    env.data.qpos[ta:ta + 3] += shift
    env.data.qpos[pa:pa + 3] += shift
    mujoco.mj_forward(env.model, env.data)
    before = float(env.data.qpos[ta:ta + 3] @ axis)
    unparked = env.checkpoint()
    env.dynamics.physics_step(env)

    assert float(env.data.qpos[ta:ta + 3] @ axis) == pytest.approx(before + 10.0, abs=0.01)
    np.testing.assert_allclose(
        env.data.qpos[pa:pa + 3] - env.data.qpos[ta:ta + 3], rel, atol=1e-9)
    assert {tray, tool} <= set(env.dynamics.off_stage)
    assert not env.scene_over
    gone = env.data.qpos[ta:ta + 7].copy()
    for _ in range(40):
        env._step_physics()
    np.testing.assert_allclose(env.data.qpos[ta:ta + 7], gone, atol=1e-5)

    other = "toolbox1"
    oa, od = env._obj_qadr[other], env._obj_dofadr[other]
    pose = env.model.qpos0[oa:oa + 7].copy()
    pose[:3] += axis * (float(env.dynamics.cfg["end"]) + 0.03 - float(pose[:3] @ axis))
    pose[2] -= 0.0005
    env.data.qpos[oa:oa + 7] = pose
    env.data.qvel[od:od + 6] = 0.0
    mujoco.mj_forward(env.model, env.data)
    env.dynamics.physics_step(env)
    assert other in env.dynamics.off_stage and env.scene_over
    parked_geoms = np.concatenate([env._subtree_geoms(n) for n in (tray, tool, other)])
    assert not env.model.geom_contype[parked_geoms].any()

    env.restore(unparked)
    assert not env.dynamics.off_stage and not env.scene_over
    np.testing.assert_array_equal(env.model.geom_contype, env.dynamics.contype)

    env.dynamics.physics_step(env)
    env.reset(seed=1)
    np.testing.assert_array_equal(env.model.geom_contype, env.dynamics.contype)
    fresh = SimEnv("pack_tools_into_toolboxes", seed=1)
    fresh.reset(seed=1)
    for name in ("toolbox0", "toolbox1"):
        np.testing.assert_allclose(env.data.body(name).xpos, fresh.data.body(name).xpos, atol=1e-6)


def test_toolbox_conveyor_static_rail_rider_does_not_leave_with_box(
        toolbox_conveyor):
    env = toolbox_conveyor
    env.dynamics.spawn_schedule = ()
    axis = np.asarray(env.dynamics.cfg["axis"], float)
    axis /= np.linalg.norm(axis)
    tray, tool = "toolbox0", "tape_measure0"

    ta, td = env._obj_qadr[tray], env._obj_dofadr[tray]
    tray_pose = env.model.qpos0[ta:ta + 7].copy()
    tray_pose[:3] += axis * (float(env.dynamics.cfg["end"]) + 0.005
                             - float(tray_pose[:3] @ axis))
    env.data.qpos[ta:ta + 7] = tray_pose
    env.data.qvel[td:td + 6] = 0.0

    pa, pd = env._obj_qadr[tool], env._obj_dofadr[tool]
    rail = env.model.geom("rail_mid_collision").id
    rail_top = float(env.model.geom_pos[rail][2] + env.model.geom_size[rail][2])
    tool_pose = env.model.qpos0[pa:pa + 7].copy()
    tool_pose[:3] = [-0.100, tray_pose[1], rail_top + 0.01654]
    env.data.qpos[pa:pa + 7] = tool_pose
    env.data.qvel[pd:pd + 6] = 0.0
    mujoco.mj_forward(env.model, env.data)
    assert any(rail in (int(contact.geom1), int(contact.geom2))
               for contact in env.data.contact[:env.data.ncon])
    assert tool not in env.dynamics.carry_group(env, tray, set())

    tool_before = env.data.qpos[pa:pa + 3].copy()
    tray_before = float(env.data.qpos[ta:ta + 3] @ axis)
    env.dynamics.physics_step(env)

    np.testing.assert_allclose(env.data.qpos[pa:pa + 3], tool_before, atol=1e-9)
    assert float(env.data.qpos[ta:ta + 3] @ axis) == pytest.approx(tray_before + 10.0, abs=0.01)
    assert tool not in env.dynamics.off_stage


def test_toolbox_conveyor_does_not_remove_held_rider(toolbox_conveyor):
    env, name, arm = toolbox_conveyor, "pliers0", toolbox_conveyor.arms[0]
    env.dynamics.cfg["speed"] = 0.0
    a, dof = env._obj_qadr[name], env._obj_dofadr[name]
    env.data.qpos[a:a + 7] = env.model.qpos0[a:a + 7]
    env.data.qpos[a + 1] = 0.0
    env.data.qvel[dof:dof + 6] = 0.0
    mujoco.mj_forward(env.model, env.data)
    euler = env._home_euler[arm]
    for height, grip in ((0.12, 1), (0.006, 1), (0.006, 0)):
        env.move_to(env.data.body(name).xpos.copy() + [0.0, 0.0, height],
                    euler, grip, arm=arm)

    belt = set(map(int, env._subtree_geoms(env.dynamics.cfg["body"])))
    body = env.model.body(name).id
    root = env.model.body_rootid
    belt_contact = robot_contact = False
    for con in env.data.contact[:env.data.ncon]:
        g1, g2 = int(con.geom1), int(con.geom2)
        b1 = int(root[env.model.geom_bodyid[g1]])
        b2 = int(root[env.model.geom_bodyid[g2]])
        if body not in (b1, b2):
            continue
        belt_contact |= g1 in belt or g2 in belt
        other = b2 if b1 == body else b1
        robot_contact |= env.model.body(other).name.startswith(env.robot_prefix)
    assert belt_contact and robot_contact

    axis = np.asarray(env.dynamics.cfg["axis"], float)
    coord = float(env.data.qpos[a:a + 3] @ axis)
    env.dynamics.cfg["end"] = coord - 0.001
    before = env.data.qpos[a:a + 3].copy()
    env._step_physics()
    assert np.linalg.norm(env.data.qpos[a:a + 3] - before) < 0.01
    assert np.linalg.norm(env.data.qpos[a:a + 3]
                          - env.dynamics.exit_poses[name][:3]) > 1.0


def test_toolbox_conveyor_case_belt_runs_at_its_own_speed(toolbox_conveyor):
    env = toolbox_conveyor
    env.dynamics.spawn_schedule = ()
    env.dynamics.cfg["slab_speeds"] = {"belt_cases_collision": 0.010}
    before = {b: float(env.data.body(b).xpos[1]) for b in ("pliers0", "toolbox0")}
    for _ in range(8000):
        env._step_physics()
    travel = {b: before[b] - float(env.data.body(b).xpos[1]) for b in before}
    assert travel["pliers0"] == pytest.approx(0.030, abs=0.003)
    assert travel["toolbox0"] == pytest.approx(0.020, abs=0.003)


def test_toolbox_conveyor_wrap_remains_the_default(toolbox_conveyor):
    env, name = toolbox_conveyor, "hammer0"
    env.dynamics.cfg.pop("exit")
    env.dynamics.deal(env)
    env.dynamics.cfg["start"] = -0.2
    _put_tool_past_conveyor_end(env, name)

    env._step_physics()
    axis = np.asarray(env.dynamics.cfg["axis"], float)
    coord = float(env.data.qpos[env._obj_qadr[name]:env._obj_qadr[name] + 3] @ axis)
    assert coord == pytest.approx(env.dynamics.cfg["start"], abs=0.01)


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
