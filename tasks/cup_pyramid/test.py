"""Tests of the cup_pyramid task."""
import json
import xml.etree.ElementTree as ET
import mujoco
import numpy as np
import pytest
from core.tasks import TASKS, load_scene
from core.sim_env import SimEnv
from core.grade_episode import make_checker
from helpers import (
    _pack_env,
    set_free_body,
    zero_free_qvel,
)


def test_spd_speed_stacks_pack_is_three_randomized_red_cups():
    pack = TASKS / "cup_pyramid"
    spec = json.loads((pack / "task.json").read_text())
    cups = ["cup_slot_b", "cup_slot_c", "cup_slot_e"]
    assert spec["category"] == "game"
    assert spec["objects"] == cups
    assert spec["termination"]["max_len"] == 5000
    assert spec["language"] == "Turn the three cups upside down and build a pyramid."
    assert spec["success"]["pred"] == "hold" and spec["success"]["steps"] == 20
    assert not any(dash in spec["language"] for dash in ("\u2013", "\u2014", " -- "))

    rnd = spec["randomization"]
    assert rnd["full_rand"] is True
    assert rnd["groups"] == [[cup] for cup in cups]
    assert rnd["lift_reach"] == {cup: 0.14 for cup in cups}
    assert "placement_regions" not in rnd
    rim_diameter = 2 * 0.0431538
    assert rnd["min_sep"] == [
        [cups[i], cups[j], rim_diameter]
        for i in range(len(cups)) for j in range(i + 1, len(cups))
    ]
    assert rnd["grasp_clearance"] == 0.02

    root = ET.parse(pack / "scene.xml").getroot()
    starts = {
        "cup_slot_b": ("0.71783 -0.50314 0.83861",
                       "0.38060 0.00003 -0.00004 -0.92474"),
        "cup_slot_c": ("0.74727 -0.20242 0.83861",
                       "0.99901 0.00005 0.00001 -0.04448"),
        "cup_slot_e": ("0.63930 0.50417 0.83861",
                       "0.33893 0.00001 0.00005 0.94081"),
    }
    for cup, (expected_pos, expected_quat) in starts.items():
        body = next(body for body in root.find("worldbody").iter("body")
                    if body.get("name") == cup)
        assert body.get("pos") == expected_pos
        assert body.get("quat") == expected_quat
        material = next(material for material in root.find("asset").iter("material")
                        if material.get("name", "").startswith(cup + "__"))
        assert "texture" not in material.attrib
        assert material.get("rgba") == "0.80 0.08 0.08 1"
        assert material.get("specular") == "0.05"

    model = load_scene(pack).compile()
    for cup in cups:
        root_id = model.body(cup).id
        subtree = []
        for body_id in range(model.nbody):
            ancestor = body_id
            while ancestor and ancestor != root_id:
                ancestor = int(model.body_parentid[ancestor])
            if ancestor == root_id:
                subtree.append(body_id)
        assert sum(float(model.body_mass[body_id]) for body_id in subtree) == pytest.approx(0.050)
        collision_geoms = [gid for gid in range(model.ngeom)
                           if int(model.geom_bodyid[gid]) in subtree
                           and int(model.geom_group[gid]) == 3
                           and int(model.geom_contype[gid]) != 0]
        assert collision_geoms
        assert all(np.allclose(model.geom_solref[gid], [0.004, 1.0])
                   and int(model.geom_priority[gid]) == 1
                   and np.allclose(model.geom_solimp[gid][:2], [0.998, 0.998])
                   for gid in collision_geoms)

    success = spec["success"]
    assert success["pred"] == "hold" and success["steps"] == 20
    assert any(node["pred"] == "robot_clear" and node["bodies"] == cups
               for node in success["of"]["of"])

    env = SimEnv("cup_pyramid", seed=0)
    env.reset(seed=0, randomize_init=True)
    for cup in cups:
        pos = env.data.body(cup).xpos.copy()
        assert cup not in env._placement_regions
        assert len(env.legal_cells([cup]))
        assert env._in_pc_crop([cup])
        assert any(env.reachable(pos, arm) for arm in env.arms)
        assert any(env.reachable(pos + [0, 0, rnd["lift_reach"][cup]], arm)
                   for arm in env.arms)
        assert any(env.ctl.reaches(arm, pos, env._home_euler[arm])[0]
                   for arm in env.arms)
        assert env.data.body(cup).xmat.reshape(3, 3)[2, 2] > 0.99


def test_spd_pyramid_needs_a_supported_arrangement_held_for_a_second():
    env = _pack_env("cup_pyramid")
    cups = ["cup_slot_b", "cup_slot_c", "cup_slot_e"]

    def put(body, pos):
        set_free_body(env, body, pos)
        qadr = env.model.joint(env.model.body(body).jntadr[0]).qposadr[0]
        env.data.qpos[qadr + 3:qadr + 7] = [0, 1, 0, 0]
        zero_free_qvel(env, body)

    check, hold = make_checker(env.spec["success"])
    assert hold == 20
    assert check(env) is False
    for body, pos in zip(cups, ([0.70, 0.0, 0.87], [0.80, 0.0, 0.87],
                                [0.75, 0.0, 0.95])):
        put(body, pos)
    mujoco.mj_forward(env.model, env.data)
    for _ in range(20):
        for _ in range(env._substeps):
            env._step_physics()
        assert check(env) is False

    for body, pos in zip(cups, ([0.70, 0.0, 0.839], [0.80, 0.0, 0.839],
                                [0.75, 0.0, 0.940])):
        put(body, pos)
    mujoco.mj_forward(env.model, env.data)
    assert check(env) is True


def test_axis_dot_reads_orientation_and_abs_ignores_which_end():
    env = _pack_env("shell_game")
    up, _ = make_checker({"pred": "axis_dot", "obj": "cube_red", "min_dot": 0.94})
    either, _ = make_checker({"pred": "axis_dot", "obj": "cube_red", "min_dot": 0.94, "abs": True})
    flat, _ = make_checker({"pred": "axis_dot", "obj": "cube_red", "min_dot": 0.0,
                            "max_dot": 0.15, "abs": True})
    a = env._obj_qadr["cube_red"]
    env.data.qpos[a + 3:a + 7] = [1, 0, 0, 0]
    mujoco.mj_forward(env.model, env.data)
    assert up(env) and either(env) and not flat(env)
    env.data.qpos[a + 3:a + 7] = [0, 1, 0, 0]
    mujoco.mj_forward(env.model, env.data)
    assert not up(env) and either(env)
    env.data.qpos[a + 3:a + 7] = [0.7071, 0, 0.7071, 0]
    mujoco.mj_forward(env.model, env.data)
    assert flat(env) and not either(env)
