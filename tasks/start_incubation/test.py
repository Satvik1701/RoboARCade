"""Tests of the start_incubation task."""
from pathlib import Path
import json
import xml.etree.ElementTree as ET
import numpy as np
from core.grade_episode import nodes
from core.tasks import ASSETS, TASKS, robot_kinematics
import mujoco
from core.sim_env import SimEnv
from helpers import (
    set_free_body,
)

import pytest


def test_autobio_start_incubation_has_physical_wells_and_ordered_return():
    pack = TASKS / "start_incubation"
    spec = json.loads((pack / "task.json").read_text())
    root = ET.parse(pack / "scene.xml").getroot()

    block = next(body for body in root.find("worldbody").iter("body")
                 if body.get("name") == "thermoblock")
    geoms = list(block.findall("geom"))
    assert len(geoms) == 3110
    assert all(g.get("name", "").startswith("thermoblock_") for g in geoms)
    assert all(g.get("class") == "collision" for g in geoms)
    assert len([g for g in geoms if "_level_" in g.get("name", "")]) == 2688
    deck_geoms = [g for g in geoms if g.get("name", "").startswith(
        "thermoblock_deck_")]
    assert len(deck_geoms) == 417
    assert len([g for g in deck_geoms if "_lip_" in g.get("name", "")]) == 384
    machine_sites = [s for s in block.findall("site")
                     if s.get("name", "").startswith("machine_well_")]
    assert len(machine_sites) == 24 and len({s.get("name") for s in machine_sites}) == 24

    rack = next(body for body in root.find("worldbody").iter("body")
                if body.get("name") == "tube_rack")
    rack_sites = [s for s in rack.iter("site")
                  if s.get("name", "").startswith("rack_hole_")]
    assert len(rack_sites) == 60 and len({s.get("name") for s in rack_sites}) == 60
    rack_deck = [g for g in rack.iter("geom")
                 if g.get("name", "").startswith("tube_rack_deck_rack_hole_")]
    assert len(rack_deck) == 2880
    assert all(g.get("mass") == "0" for g in rack_deck)

    vortex_root = ET.parse(
        TASKS / "vortex_sample" / "scene.xml").getroot()
    vortex_rack = next(body for body in vortex_root.find("worldbody").iter("body")
                       if body.get("name") == "tube_rack")
    vortex_deck = [g for g in vortex_rack.iter("geom")
                   if g.get("name", "").startswith("tube_rack_deck_rack_hole_")]
    assert [{**g.attrib, "friction": None} for g in vortex_deck] == [
        {**g.attrib, "friction": None} for g in rack_deck]
    assert {g.get("friction") for g in rack_deck} == {"1 0.002 0.0001"}

    rnd = spec["randomization"]
    assert spec["objects"] == ["tube_rack", "tube_1", "tube_2"]
    assert rnd["groups"] == [["tube_rack", "tube_1", "tube_2"]]
    assert rnd["slots"]["parent"] == "tube_rack"
    assert rnd["slots"]["members"] == ["tube_1", "tube_2"]
    assert len(rnd["slots"]["poses"]) == 60
    assert len({tuple(pose) for pose in rnd["slots"]["poses"]}) == 60
    rack_site_pos = [np.fromstring(site.get("pos"), sep=" ") for site in rack_sites]
    expected_slots = [np.r_[pos[:2], pos[2] + 0.002, 0.0]
                      for pos in rack_site_pos]
    assert all(np.allclose(pose, expected, atol=1e-9)
               for pose, expected in zip(rnd["slots"]["poses"], expected_slots))
    assert rnd["full_rand"] is True
    assert rnd["placement_regions"]["tube_rack"] == [{
        "x": [-0.15, -0.03], "y": [-0.26, -0.16],
        "yaw_offset_deg": [-15.0, 15.0]}]
    assert set(s.get("name") for s in machine_sites + rack_sites) <= set(rnd["reach_sites"])
    assert not any(dash in spec["language"] for dash in ("-", "\u2013", "\u2014"))

    success = spec["success"]
    assert success["pred"] == "sequence" and len(success["of"]) == 4
    assert success["invariant"]["pred"] == "all"
    for i, stage in enumerate(success["of"]):
        counts = [node for node in nodes(stage) if node["pred"] == "press_count"]
        assert len(counts) == 1 and counts[0]["count"] == (0 if i == 0 else 1)
    for stage in success["of"][:2]:
        contacts = {(node.get("a"), node.get("b")) for node in nodes(stage)
                    if node["pred"] == "touching_part"}
        assert contacts == {("tube_1", "thermoblock"), ("tube_2", "thermoblock")}
    final = list(nodes(success["of"][-1]))
    assert any(node["pred"] == "eef_near_start" for node in final)
    assert any(node["pred"] == "gripper_open" for node in final)
    assert any(node["pred"] == "robot_clear"
               and node["bodies"] == ["tube_1", "tube_2", "tube_rack", "mixer"]
               for node in final)


def test_in_start_site_requires_the_hole_the_layout_dealt():
    from core.grade_episode import _build, _in_site_geom
    env = SimEnv("start_incubation", seed=1)
    env.reset(seed=1)
    for _ in range(int(0.05 / env.model.opt.timestep)):
        env._step_physics()
    leaf = next(n for n in nodes(env.spec["success"]) if n.get("pred") == "in_start_site" and n["obj"] == "tube_1")
    start = {obj: next((s for s in leaf["sites"] if _in_site_geom(env, obj, s, np.asarray(leaf["size"]))), None)
             for obj in ("tube_1", "tube_2")}
    assert None not in start.values() and start["tube_1"] != start["tube_2"]
    check = _build(leaf, "start_incubation")
    assert check(env) is True
    other = next(s for s in leaf["sites"] if s not in start.values())
    home, there = env.data.site(start["tube_1"]).xpos.copy(), env.data.site(other).xpos.copy()
    pos = env.data.body("tube_1").xpos.copy()
    set_free_body(env, "tube_1", pos + (there - home))
    mujoco.mj_forward(env.model, env.data)
    assert check(env) is False
    set_free_body(env, "tube_1", pos)
    mujoco.mj_forward(env.model, env.data)
    assert check(env) is True


def robot_model(name):
    return mujoco.MjModel.from_xml_path(str(ASSETS / "robots" / name / robot_kinematics(name)["mjcf"]))


@pytest.fixture(scope="module")
def robots():
    return robot_model("franka_panda"), robot_model("franka_panda_low_torsion")


def test_low_torsion_robot_changes_only_torsional_coefficients(robots):
    old, new = robots
    changed = [name for name in dir(old) if not name.startswith("_")
               and isinstance(getattr(old, name), np.ndarray)
               and not np.array_equal(getattr(old, name), getattr(new, name))]
    assert changed == ["geom_friction"]
    np.testing.assert_array_equal(old.geom_friction[:, [0, 2]], new.geom_friction[:, [0, 2]])
    fingers = np.isin(old.geom_bodyid, [old.body("left_finger").id, old.body("right_finger").id])
    np.testing.assert_array_equal(new.geom_friction[:, 1],
                                  np.minimum(old.geom_friction[:, 1], np.where(fingers, .01, .002)))
    assert np.max(old.geom_friction[:, 1]) == .3


def test_start_incubation_caps_every_torsional_coefficient():
    m = SimEnv("start_incubation").model
    fingers = np.isin(m.geom_bodyid, [m.body("robot0_left_finger").id, m.body("robot0_right_finger").id])
    assert np.max(m.geom_friction[~fingers, 1]) == .002
    assert np.max(m.geom_friction[fingers, 1]) == .01
    assert m.npair == 0


@pytest.mark.parametrize("case,name", [("launch", "tube_1"), ("one_pad", "tube_2")])
def test_recorded_incubation_contact_failures(case, name):
    states = np.load(Path(__file__).parent / "incubation_contact_states.npz")
    env = SimEnv("start_incubation")
    m, d = env.model, env.data
    env.reset_to_sim_state(states[case])
    body = m.body(name).id
    dof = env._obj_dofadr[name]
    inertia_rot = np.empty(9)
    mujoco.mju_quat2Mat(inertia_rot, m.body_iquat[body])
    z0, energies = float(d.xipos[body, 2]), []
    for step in range(801):
        mujoco.mj_forward(m, d)
        mujoco.mj_subtreeVel(m, d)
        velocity = d.subtree_linvel[body]
        omega = inertia_rot.reshape(3, 3).T @ d.qvel[dof+3:dof+6]
        energies.append(.5*m.body_mass[body]*(velocity @ velocity)
                        + .5*np.dot(m.body_inertia[body], omega*omega)
                        + m.body_mass[body]*9.81*(d.xipos[body, 2]-.8))
        if step < 800:
            mujoco.mj_step(m, d)
    assert max(energies)-energies[0] < .0001
    assert d.xipos[body, 2] < z0-.02
    for contact in d.contact:
        if body not in (m.geom_bodyid[contact.geom1], m.geom_bodyid[contact.geom2]):
            continue
        other = contact.geom1 if m.geom_bodyid[contact.geom2] == body else contact.geom2
        assert env.finger_root(int(m.geom_bodyid[other])) is None


def test_recorded_tube_seats_after_release():
    states = np.load(Path(__file__).parent / "incubation_contact_states.npz")
    env = SimEnv("start_incubation")
    env.reset_to_sim_state(states["insertion_jam"])
    for _ in range(round(1/env.model.opt.timestep)):
        mujoco.mj_step(env.model, env.data)
    tip = env.data.body("tube_2").xpos
    floor = env.data.site("machine_well_r1_c1").xpos
    assert abs(tip[2]-floor[2]) < .001
    assert np.linalg.norm(tip[:2]-floor[:2]) < .001


@pytest.mark.parametrize("heading", [0, 90, 180, 270])
def test_round_tip_topples_between_holes(heading):
    from scipy.spatial.transform import Rotation
    env = SimEnv("start_incubation")
    m, d = env.model, env.data
    mujoco.mj_forward(m, d)
    b = m.body("tube_2").id
    a = int(m.jnt_qposadr[int(m.body(b).jntadr[0])])
    centre = (d.site("machine_well_r1_c1").xpos +
              d.site("machine_well_r2_c2").xpos)/2
    deck = max(d.geom_xpos[g, 2]+m.geom_size[g, 2] for g in range(m.ngeom)
               if m.geom(g).name.startswith("thermoblock_deck_row_strip"))
    axis = np.array([np.cos(np.radians(heading)), np.sin(np.radians(heading)), 0.])
    d.qpos[a:a+3] = [centre[0], centre[1], deck+.001]
    d.qpos[a+3:a+7] = Rotation.from_rotvec(axis*np.radians(.1)).as_quat()[[3, 0, 1, 2]]
    mujoco.mj_forward(m, d)
    for _ in range(round(3/m.opt.timestep)):
        mujoco.mj_step(m, d)
    assert d.xmat[b].reshape(3, 3)[2, 2] < np.cos(np.radians(45))
    assert m.body_mass[b] == pytest.approx(.007810109237082784, abs=1e-14)
