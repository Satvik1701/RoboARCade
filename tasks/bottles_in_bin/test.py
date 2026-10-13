"""Tests of the bottles_in_bin task."""
import json
import xml.etree.ElementTree as ET
from core.tasks import CATEGORY_REAL_WORLD, TASKS
import mujoco
from core.grade_episode import make_checker
from core.sim_env import SimEnv
from helpers import (
    set_free_body,
    zero_free_qvel,
)


def test_spd_bottle_randomization_materials_and_contact_are_explicit():
    pack = TASKS / "bottles_in_bin"
    spec = json.loads((pack / "task.json").read_text())
    bottles = [f"bottle_slot_{letter}" for letter in "abcd"]
    rnd = spec["randomization"]

    assert spec["category"] == CATEGORY_REAL_WORLD
    assert spec["metadata"]["bottles"] == {
        "container": "500 ml PET bottle",
        "fill_state": "about one-quarter full",
        "mass_kg": 0.124,
    }
    assert rnd["full_rand"] is True
    assert rnd["groups"] == [["bin_slot"], *[[bottle] for bottle in bottles]]
    assert rnd["placement_regions"] == {"bin_slot": [{"x": [0.725, 0.925], "y": [-0.10, 0.10],
                                                      "yaw_offset_deg": [-180, 180]}]}
    assert rnd["lift_reach"] == {bottle: 0.12 for bottle in bottles}

    root = ET.parse(pack / "scene.xml").getroot()
    collisions = [default.find("geom") for default in root.iter("default")
                  if (default.get("class") or "").startswith("bottle_slot_")
                  and default.get("class").endswith("__collision")]
    assert len(collisions) == 4
    assert all(geom is not None and geom.get("priority") == "2"
               and geom.get("condim") == "6"
               and geom.get("friction") == "0.5 0.005 0"
               for geom in collisions)
    pairs = root.findall("contact/pair")
    assert len(pairs) == 338 and all(pair.get("solref") == "0.004 0.5" and pair.get("condim") == "6"
                                     and pair.get("friction") == "0.5 0.5 0.005 0.001 0.001"
                                     and pair.get("geom1").startswith("bottle_slot_")
                                     and (pair.get("geom2") == "table_plane" or pair.get("geom2").startswith("bin_col"))
                                     for pair in pairs)

    bodies = {body.get("name"): body for body in root.iter("body") if body.get("name")}
    materials = {material.get("name"): material for material in root.iter("material")}
    assigned = []
    rgba = []
    for bottle in bottles:
        visuals = [geom for geom in bodies[bottle + "__object"].findall("geom")
                   if geom.get("type") == "mesh" and geom.get("group") == "1"
                   and (geom.get("mesh") or "").startswith(bottle + "__model_normalized_")]
        names = {geom.get("material") for geom in visuals}
        assert len(names) == 1
        name = names.pop()
        assert name.startswith(bottle + "__opaque_")
        material = materials[name]
        color = [float(value) for value in material.get("rgba").split()]
        assert color[3] == 1.0
        assert float(material.get("specular")) <= 0.15
        assert float(material.get("shininess")) <= 0.10
        assigned.append(name)
        rgba.append(tuple(color))
    assert len(set(assigned)) == len(bottles)
    assert len(set(rgba)) == len(bottles)


def test_in_site_box_rejects_robot_contact_with_child_body(monkeypatch):
    env = SimEnv("bottles_in_bin", seed=0)
    env.reset(seed=0)
    leaf = env.spec["success"]["of"][0]
    obj = leaf["obj"]
    set_free_body(env, obj, env.data.site(leaf["site"]).xpos)
    zero_free_qvel(env, obj)
    mujoco.mj_forward(env.model, env.data)

    root = env.model.body(obj).id
    child = env.model.body(obj + "__object").id
    assert env.model.body_rootid[child] == root
    finger = next(iter(env.finger_roots(env.arms[0])))
    monkeypatch.setattr(env, "contact_body_pairs", lambda: [(finger, child)])
    assert env.robot_clear([obj]) is False

    check, _ = make_checker(leaf)
    assert check(env) is False
