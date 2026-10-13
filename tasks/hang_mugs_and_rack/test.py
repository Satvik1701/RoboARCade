"""Tests of the hang_mugs_and_rack task."""
import json
import xml.etree.ElementTree as ET
import numpy as np
import pytest
from core.tasks import CATEGORY_REAL_WORLD, TASKS, asset_dir
from core.sim_env import SimEnv
import mujoco
from scipy.spatial.transform import Rotation as R
from core.grade_episode import make_checker, nodes
from helpers import (
    _pack_env,
    set_free_body,
    zero_free_qvel,
)


def test_spd_kitchen_full_randomizes_only_retained_upright_objects():
    task = "hang_mugs_and_rack"
    pack = TASKS / task
    spec = json.loads((pack / "task.json").read_text())
    mugs = [f"mug_slot_{letter}" for letter in "ad"]
    plates = [f"plate_slot_{letter}" for letter in "cd"]
    loose = mugs + plates
    removed = {"mug_slot_b", "mug_slot_c", "plate_slot_a", "plate_slot_b"}
    rnd = spec["randomization"]

    assert spec["category"] == CATEGORY_REAL_WORLD
    assert spec["language"] == (
        "Hang each mug on the mug tree, and stand each plate in the dish rack.")
    assert not removed & set(spec["objects"])
    assert rnd["full_rand"] is True
    assert rnd["groups"] == [["rack_slot", "mug_tree_slot"],
                             *[[body] for body in loose]]
    assert set(rnd["placement_regions"]) == {"rack_slot", *mugs}
    assert rnd["placement_regions"]["rack_slot"] == [{
        "x": [0.794995, 0.794995], "y": [-0.00002, -0.00002],
        "yaw_offset_deg": [0, 0],
    }]
    for mug in mugs:
        assert rnd["placement_regions"][mug] == [{
            "x": [0.50, 0.75], "y": [-0.15, 0.15], "yaw_offset_deg": [-180, 180]}]
    assert rnd["region"] == {"x": [0, 0], "y": [0, 0],
                             "yaw": [-np.pi, np.pi]}
    assert rnd["lift_reach"] == {body: 0.1 for body in loose}
    assert len(rnd["min_sep"]) == 6
    assert {distance for _a, _b, distance in rnd["min_sep"]} == {0.1}
    assert "grasp_clearance" not in rnd

    scene_text = (pack / "scene.xml").read_text()
    root = ET.fromstring(scene_text)
    assert all(name not in scene_text for name in removed)
    rack = next(body for body in root.iter("body") if body.get("name") == "rack_slot")
    assert np.allclose([float(value) for value in rack.get("pos").split()],
                       [0.77, -0.33, 0.81836])
    rack_object = next(body for body in root.iter("body")
                       if body.get("name") == "rack_slot__object")
    rack_collisions = [geom for geom in rack_object.findall("geom")
                       if geom.get("class") == "rack_slot__collision"]
    assert len(rack_collisions) == 71
    z_bounds = [
        (float(geom.get("pos").split()[2]) - float(geom.get("size").split()[2]),
         float(geom.get("pos").split()[2]) + float(geom.get("size").split()[2]))
        for geom in rack_collisions]
    assert min(lo for lo, _hi in z_bounds) == pytest.approx(-0.03915197)
    assert max(hi for _lo, hi in z_bounds) == pytest.approx(0.08857858)
    assert len([geom for geom in rack_object.findall("geom")
                if geom.get("class") == "rack_slot__visual"]) == 1
    assert not any(path.name.startswith(tuple(removed))
                   or path.name == "plate_texture.png"
                   for path in (asset_dir(pack) / "assets").iterdir())

    env = SimEnv(task, seed=0)
    env.reset(seed=0, randomize_init=False)
    fixtures = {
        name: np.r_[env.data.body(name).xpos, env.data.body(name).xquat].copy()
        for name in ("rack_slot", "mug_tree_slot")}
    env.reset(seed=0, randomize_init=True)
    assert all(np.array_equal(
        np.r_[env.data.body(name).xpos, env.data.body(name).xquat], pose)
        for name, pose in fixtures.items())
    assert all(env.data.body(name).xmat.reshape(3, 3)[2, 2] > 0.99
               for name in loose)


def test_spd_kitchen_pad_pairs_carry_the_rolling_term():
    from core.sim_env import SimEnv

    env = SimEnv("hang_mugs_and_rack", seed=0)
    m = env.model
    root = lambda g: m.body(int(m.body_rootid[int(m.geom_bodyid[g])])).name or ""
    fingers = [g for g in range(m.ngeom) if "finger" in (m.body(int(m.geom_bodyid[g])).name or "")
               and (m.geom_contype[g] or m.geom_conaffinity[g])]
    plates = [g for g in range(m.ngeom) if root(g).startswith("plate_slot") and m.geom_contype[g]]
    pad = [p for p in range(m.npair) if m.pair_geom1[p] in plates]
    assert len(pad) == len(fingers) * len(plates)
    assert {(m.pair_geom1[p], m.pair_geom2[p]) for p in pad} == {(p, f) for p in plates for f in fingers}
    assert all(m.pair_dim[p] == 6 and abs(m.pair_friction[p][3] - 0.05) < 1e-9 for p in pad)
    assert all(m.geom_friction[g][2] < 0.01 for g in plates)


def test_spd_kitchen_requires_fixture_contact_and_accepts_scripted_goals():
    env = _pack_env("hang_mugs_and_rack")

    def put(body, fixture, offset, quat=None):
        pos = env.data.body(fixture).xpos.copy()
        pos += [offset[0], offset[1], offset[2] - pos[2]]
        set_free_body(env, body, pos)
        if quat is not None:
            adr = env._obj_qadr[body]
            env.data.qpos[adr + 3:adr + 7] = quat
        zero_free_qvel(env, body)

    assert make_checker(env.spec["success"])[0](env) is False
    for letter in "ad":
        put(f"mug_slot_{letter}", "mug_tree_slot", (0.16, 0.0, 1.01))
    for letter in "cd":
        put(f"plate_slot_{letter}", "rack_slot", (0.0, 0.0, 1.07),
            [np.sqrt(0.5), np.sqrt(0.5), 0, 0])
    mujoco.mj_forward(env.model, env.data)
    for group in env.spec["success"]["of"][:2]:
        for goal in group["of"]:
            noncontact = {"pred": "all", "of": [
                node for node in goal["of"] if node["pred"] != "touching_part"]}
            assert make_checker(noncontact)[0](env) is True
    assert make_checker(env.spec["success"])[0](env) is False

    env.reset(seed=0, randomize_init=False)
    mug_goals = [(-0.12, -0.05, 0.87), (0.0, 0.04, 0.87)]
    plate_goals = [
        ((-0.000183, 0.020832, 0.882042),
         [0.692568412, -0.721351940, -0.000191331, -0.000579983]),
        ((0.000113, 0.103942, 0.885389),
         [0.696072258, -0.717971459, 0.000626397, 0.000060499]),
    ]
    for letter, offset in zip("ad", mug_goals):
        put(f"mug_slot_{letter}", "mug_tree_slot", offset)
    for letter, (offset, quat) in zip("cd", plate_goals):
        put(f"plate_slot_{letter}", "rack_slot", offset, quat)
    mujoco.mj_forward(env.model, env.data)
    assert make_checker(env.spec["success"])[0](env) is True


@pytest.mark.parametrize("angle,expected", [
    (0, False), (45.9, False), (46.1, True), (90, True),
    (133.9, True), (134.1, False), (180, False),
])
def test_spd_kitchen_plates_on_edge_either_face(angle, expected):
    env = _pack_env("hang_mugs_and_rack")
    for node in nodes(env.spec["success"]):
        if node["pred"] != "axis_dot":
            continue
        a = env._obj_qadr[node["obj"]]
        env.data.qpos[a + 3:a + 7] = np.roll(
            R.from_euler("x", angle, degrees=True).as_quat(), 1)
        mujoco.mj_forward(env.model, env.data)
        assert make_checker(node)[0](env) is expected
