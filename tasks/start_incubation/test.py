"""Tests of the start_incubation task."""


from pathlib import Path
from types import SimpleNamespace
import json
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest

from core.grade_episode import PressEdge, nodes
from core.sim_env import SimEnv
from core.tasks import ASSETS, TASKS, load_scene, robot_kinematics
from helpers import set_free_body
from tasks.start_incubation.dynamics import Dynamics, LATCHES, SEGMENTS


def test_autobio_start_incubation_has_physical_wells_and_ordered_return():
    pack = TASKS / "start_incubation"
    spec = json.loads((pack / "task.json").read_text())
    root = ET.parse(pack / "scene.xml").getroot()

    block = next(body for body in root.find("worldbody").iter("body")
                 if body.get("name") == "thermoblock")
    geoms = list(block.findall("geom"))
    assert geoms and all(g.get("name", "").startswith("thermoblock_surface_") for g in geoms)
    assert all(g.get("class") == "collision" and g.get("type") == "mesh" for g in geoms)
    assert not any(g.get("type") == "box" for g in block.findall("geom"))
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
    deck = -np.inf
    for g in range(m.ngeom):
        if not m.geom(g).name.startswith("thermoblock_surface_"): continue
        mesh = m.geom_dataid[g]
        start, count = m.mesh_vertadr[mesh], m.mesh_vertnum[mesh]
        vertices = d.geom_xpos[g]+m.mesh_vert[start:start+count]@d.geom_xmat[g].reshape(3, 3).T
        deck = max(deck, vertices[:, 2].max())
    axis = np.array([np.cos(np.radians(heading)), np.sin(np.radians(heading)), 0.])
    d.qpos[a:a+3] = [centre[0], centre[1], deck+.001]
    d.qpos[a+3:a+7] = Rotation.from_rotvec(axis*np.radians(.1)).as_quat()[[3, 0, 1, 2]]
    mujoco.mj_forward(m, d)
    for _ in range(round(3/m.opt.timestep)):
        mujoco.mj_step(m, d)
    assert d.xmat[b].reshape(3, 3)[2, 2] < np.cos(np.radians(45))
    assert m.body_mass[b] == pytest.approx(.007810109237082784, abs=1e-14)


@pytest.fixture(scope='module')
def aligned_model():
    return load_scene(TASKS/'start_incubation').compile()


def ray(model, data, origin, group):
    mask = np.zeros(6, dtype=np.uint8); mask[group] = 1
    hit = np.array([-1], dtype=np.int32)
    distance = mujoco.mj_ray(model, data, origin, np.array([0., 0., -1.]), mask, True, -1, hit)
    assert distance >= 0
    return origin[2]-distance


@pytest.mark.parametrize('row,col', [(r, c) for r in range(4) for c in range(6)])
def test_well_visual_floor_matches_contact_surface_and_tube_seats(aligned_model, row, col):
    m = aligned_model; d = mujoco.MjData(m); mujoco.mj_forward(m, d)
    center = d.site(f'machine_well_r{row}_c{col}').xpos.copy()
    # Avoid exact shared triangle vertices, where the source mesh has tiny seams.
    for dx, dy in ((.0003, .0002), (-.0003, -.0002)):
        point = center + [dx, dy, .04]
        visual, contact = ray(m, d, point, 2), ray(m, d, point, 3)
        assert abs(visual-center[2]) < .0001
        assert 0 <= contact-visual < .0002
    body = m.body('tube_2').id
    a = int(m.jnt_qposadr[m.body_jntadr[body]])
    d.qpos[a:a+3] = center+[0, 0, .001]
    d.qpos[a+3:a+7] = [1, 0, 0, 0]
    mujoco.mj_forward(m, d)
    for _ in range(round(5/m.opt.timestep)): mujoco.mj_step(m, d)
    assert abs(d.xpos[body, 2]-center[2]) < .001
    assert np.linalg.norm(d.xpos[body, :2]-center[:2]) < .001
    assert d.xmat[body].reshape(3, 3)[2, 2] > np.cos(np.radians(12))
    block = m.body('thermoblock').id
    assert any({int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])} == {body, block}
               for c in d.contact)


def test_button_caps_and_colliders_have_identical_poses_and_sizes(aligned_model):
    m = aligned_model; d = mujoco.MjData(m); mujoco.mj_forward(m, d)
    for j in range(m.njnt):
        name = m.joint(j).name
        if not name.startswith('button_'): continue
        stem = name.removesuffix('_key')
        visual, contact = m.geom(stem+'_cap').id, m.geom(stem+'_contact').id
        np.testing.assert_array_equal(d.geom_xpos[visual], d.geom_xpos[contact])
        np.testing.assert_array_equal(d.geom_xmat[visual], d.geom_xmat[contact])
        np.testing.assert_array_equal(m.geom_size[visual], m.geom_size[contact])


@pytest.fixture(scope="module")
def model():
    return load_scene(TASKS / "start_incubation").compile()


@pytest.fixture
def env(model):
    e = SimpleNamespace(model=model, data=mujoco.MjData(model),
                        spec=json.loads((TASKS / "start_incubation/task.json").read_text()))
    e.dynamics = Dynamics(e)
    return e


def advance(env, seconds):
    env.data.time += seconds
    env.dynamics.physics_step(env)


def press(env, name, release=True):
    q = env.model.joint(f"button_{name}_key").qposadr[0]
    env.data.qpos[q] = -.003
    env.dynamics.physics_step(env)
    if release:
        env.data.qpos[q] = 0
        env.dynamics.physics_step(env)


def test_program_countdown_temperature_pause_and_completion(env):
    d = env.dynamics
    assert d.readout(env)["actual_celsius"] == 24
    press(env, "prog_1")
    advance(env, 5)
    r = d.readout(env)
    assert r["running"] and r["remaining_seconds"] == 145
    assert r["actual_celsius"] == 29
    press(env, "start_stop")
    advance(env, 10)
    assert d.readout(env)["remaining_seconds"] == 145
    assert d.readout(env)["actual_celsius"] == 27
    press(env, "start_stop")
    advance(env, 145)
    r = d.readout(env)
    assert r["done"] and not r["running"] and r["remaining_seconds"] == 0
    assert r["actual_celsius"] == 45


def test_held_buttons_do_not_repeat_and_presets_keep_edits(env):
    press(env, "time_up", release=False)
    for _ in range(10): advance(env, .1)
    assert env.dynamics.readout(env)["duration_seconds"] == 165
    env.data.qpos[env.model.joint("button_time_up_key").qposadr[0]] = 0
    advance(env, 0)
    press(env, "temp_up")
    press(env, "speed_up")
    press(env, "prog_2")
    assert env.dynamics.readout(env)["program"] == 2
    assert env.dynamics.readout(env)["duration_seconds"] == 150
    press(env, "prog_1")
    r = env.dynamics.readout(env)
    assert (r["duration_seconds"], r["target_celsius"], r["target_rpm"]) == (165, 46, 50)
    advance(env, 1)
    assert env.dynamics.readout(env)["actual_rpm"] == 50
    press(env, "start_stop")
    advance(env, .025)
    assert env.dynamics.readout(env)["actual_rpm"] == pytest.approx(25)
    advance(env, 1)
    assert env.dynamics.readout(env)["actual_rpm"] == 0


def test_short_is_momentary_and_menu_edits_selected_field(env):
    press(env, "menu_enter")
    press(env, "menu_right")  # Temperature.
    press(env, "menu_up")
    assert env.dynamics.readout(env)["target_celsius"] == 46
    press(env, "menu_down")
    press(env, "menu_left")  # Time.
    press(env, "menu_down")
    assert env.dynamics.readout(env)["duration_seconds"] == 135
    press(env, "short", release=False)
    advance(env, .5)
    assert env.dynamics.readout(env)["actual_rpm"] == 500
    assert env.dynamics.readout(env)["remaining_seconds"] == 135
    env.data.qpos[env.model.joint("button_short_key").qposadr[0]] = 0
    advance(env, 0)
    advance(env, .5)
    assert env.dynamics.readout(env)["actual_rpm"] == 0


def test_restore_reproduces_display_and_press_latches(env):
    press(env, "speed_up")
    press(env, "prog_1", release=False)
    advance(env, 3.5)
    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    state = np.empty(mujoco.mj_stateSize(env.model, spec))
    mujoco.mj_getState(env.model, env.data, state, spec)
    readout = env.dynamics.readout(env)
    colors = env.model.geom_rgba.copy()
    env.dynamics.reset(env)
    mujoco.mj_setState(env.model, env.data, state, spec)
    env.dynamics.restore(env)
    assert env.dynamics.readout(env) == readout
    np.testing.assert_array_equal(env.model.geom_rgba, colors)
    advance(env, .5)  # The still-held Program 1 key must not restart the timer.
    assert env.dynamics.readout(env)["remaining_seconds"] == 146


def test_digits_match_machine_state_and_setting_bounds(env):
    press(env, "prog_1")
    advance(env, 61)
    # 01:29 and 45/45 are actual segment states, not text in a screenshot.
    for field, text in [("time", "0129"), ("temp", "4545")]:
        for ids, char in zip(env.dynamics.digits[field], text):
            active = ''.join(letter for letter, gid in zip('abcdefg', ids)
                             if env.model.geom_rgba[gid, 0] < .5)
            assert active == SEGMENTS[int(char)]
    for field in range(3):
        s = env.dynamics.state(env)
        for _ in range(450): env.dynamics.adjust(s, field, -1)
        assert env.dynamics.settings(s)[field] == env.spec['incubation_panel']['limits'][field][0]
        for _ in range(450): env.dynamics.adjust(s, field, 1)
        assert env.dynamics.settings(s)[field] == env.spec['incubation_panel']['limits'][field][1]


def test_contact_press_starts_program_without_writing_key_pose():
    spec = load_scene(TASKS / "start_incubation")
    probe = spec.worldbody.add_body(mocap=True)
    probe.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[.003, 0, 0])
    m = spec.compile()
    e = SimpleNamespace(model=m, data=mujoco.MjData(m),
                        spec=json.loads((TASKS / "start_incubation/task.json").read_text()))
    e.dynamics = Dynamics(e)
    mujoco.mj_forward(m, e.data)
    b = m.body('button_prog_1_key').id
    centre = e.data.xpos[b].copy()
    normal = e.data.xmat[b].reshape(3, 3)[:, 2]
    for offset in np.linspace(.015, .003, 120):
        e.data.mocap_pos[0] = centre+normal*offset
        for _ in range(5):
            e.dynamics.physics_step(e)
            mujoco.mj_step(m, e.data)
    assert e.dynamics.readout(e)['running']
    assert 0 < e.dynamics.readout(e)['remaining_seconds'] < 150


BUTTONS = (
    "prog_1", "prog_2", "prog_3", "prog_4", "prog_5",
    "menu_enter", "menu_up", "menu_down", "menu_left", "menu_right",
    "short", "start_stop", "time_up", "time_down", "temp_up", "temp_down",
    "speed_up", "speed_down",
)


@pytest.fixture(scope="module")
def panel_model():
    spec = load_scene(TASKS / "start_incubation")
    probe = spec.worldbody.add_body(name="test_probe", mocap=True)
    probe.add_geom(name="test_tip", type=mujoco.mjtGeom.mjGEOM_SPHERE,
                   size=[.003, 0, 0], contype=1, conaffinity=1)
    return spec.compile()


@pytest.mark.parametrize("button", BUTTONS)
def test_contact_press_and_release(panel_model, button):
    m, d = panel_model, mujoco.MjData(panel_model)
    name = f"button_{button}"
    joint = m.joint(name + "_key")
    d.mocap_pos[0] = [0, 0, 3]
    mujoco.mj_forward(m, d)
    body = m.body(name + "_key").id
    centre = d.xpos[body].copy()
    normal = d.xmat[body].reshape(3, 3)[:, 2]
    edge = PressEdge(name + "_key")
    env = SimpleNamespace(model=m, data=d)
    pair = {m.geom("test_tip").id, m.geom(name + "_contact").id}
    presses, touched = 0, False
    # Move a physical probe into the key, without writing the key's qpos or force.
    for offset in np.linspace(.015, .003, 120):
        d.mocap_pos[0] = centre + normal * offset
        for _ in range(5):
            mujoco.mj_step(m, d)
            presses += int(edge.update(env))
            touched |= any({c.geom1, c.geom2} == pair for c in d.contact)
    assert touched
    assert presses == 1
    assert d.qpos[joint.qposadr[0]] < -.0015
    for other in BUTTONS:
        if other != button:
            assert abs(d.qpos[m.joint(f"button_{other}_key").qposadr[0]]) < .0015
    d.mocap_pos[0] = centre + normal * .025
    for _ in range(500):
        mujoco.mj_step(m, d)
        assert not edge.update(env)
    assert d.qpos[joint.qposadr[0]] > -.00075
    assert not edge.down
