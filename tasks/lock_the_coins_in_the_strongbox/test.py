"""Tests of the lock_the_coins_in_the_strongbox task."""
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R
from core.sim_env import SimEnv
import json
import pytest
from core import tasks
from core.sim_env import PHYSICS_DT


LOCKBOX_TASK = "lock_the_coins_in_the_strongbox"


def _lockbox_lid_contact(one_pad=False):
    env = SimEnv(task=LOCKBOX_TASK, seed=0)
    env.reset(seed=0, randomize_init=False)
    env.max_len = 10**9
    arm = env.arms[0]
    lid = env.model.body("lid").id
    target = env.data.geom("lid_grip").xpos.copy()
    case_rot = R.from_matrix(env.data.body("case").xmat.reshape(3, 3))
    down = (case_rot * R.from_euler("z", np.pi).inv()
            * R.from_euler("z", np.deg2rad(75))
            * R.from_euler("xyz", env._home_euler[arm])).as_euler("xyz")
    env.move_to(target + [0, 0, 0.10], down, 1)
    env.move_to(target, down, 1)
    env.move_to(target, down, 0)
    if one_pad:
        env.data.qpos[env.model.joint("j_lid").qposadr[0]] = 0.0
        mujoco.mj_forward(env.model, env.data)
    fingers = set()
    for contact in env.data.contact[: env.data.ncon]:
        for finger, other in ((contact.geom1, contact.geom2),
                              (contact.geom2, contact.geom1)):
            body = int(env.model.geom_bodyid[finger])
            root = env.finger_root(body, arm)
            if (env._ancestor_in(int(env.model.geom_bodyid[other]), {lid})
                    and root is not None):
                fingers.add(root)
    lid_geoms = {g for g in range(env.model.ngeom)
                 if env._ancestor_in(int(env.model.geom_bodyid[g]), {lid})}
    return env, arm, fingers, lid_geoms


def test_closed_two_pad_grasp_exempts_articulated_lid():
    env, arm, fingers, lid_geoms = _lockbox_lid_contact()
    assert fingers == env.finger_roots(arm)
    assert env.ctl._held_payload_geoms(arm) == lid_geoms


def test_closed_one_pad_brush_does_not_exempt_articulated_lid():
    env, arm, fingers, _ = _lockbox_lid_contact(one_pad=True)
    assert len(fingers) == 1
    assert env.ctl._held_payload_geoms(arm) == set()


def test_open_jaw_does_not_exempt_articulated_lid():
    env, arm, fingers, _ = _lockbox_lid_contact()
    assert fingers == env.finger_roots(arm)
    env.curr_gripper_open[arm] = 1
    assert env.ctl._held_payload_geoms(arm) == set()


TASK = "lock_the_coins_in_the_strongbox"
PACK = tasks.resolve_task(TASK)
SCENE = PACK / "scene.xml"
LANGUAGE = "Unlock the chest with the key, put four coins inside, and lock it again."
ROT_HI = 1.5708
BOLT_TRAVEL = 0.030
LID_OPEN = 1.7453
CASE_POS = np.array([0.739, 0.055, 0.77921])
OUTSIDE_FACE_X = 0.076
LOCK_X = 0.064
LOCK_Z = 0.115
KEY_SEATED_X = 0.075
KEY_SEATED_QUAT = np.array([0.707107, 0.0, 0.707107, 0.0])


@pytest.fixture(scope="module")
def mechanism():
    model = tasks.load_scene(PACK).compile()
    model.opt.timestep = PHYSICS_DT
    return model, mujoco.MjData(model)


def _qpos(model, name):
    return model.joint(name).qposadr[0]


def _dof(model, name):
    return model.joint(name).dofadr[0]


def _settle(model, data, steps=800):
    for _ in range(round(steps * .002 / model.opt.timestep)):
        mujoco.mj_step(model, data)


def _quat_mul(left, right):
    w1, x1, y1, z1 = left
    w2, x2, y2, z2 = right
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def _seat_key(model, data, rotor):
    mujoco.mj_forward(model, data)
    case = data.body("case")
    key_qpos = model.jnt_qposadr[model.body("key").jntadr[0]]
    local_pos = np.array([KEY_SEATED_X, 0.0, LOCK_Z])
    data.qpos[key_qpos : key_qpos + 3] = (
        case.xpos + case.xmat.reshape(3, 3) @ local_pos)
    turn = np.array([np.cos(rotor / 2), np.sin(rotor / 2), 0.0, 0.0])
    data.qpos[key_qpos + 3 : key_qpos + 7] = _quat_mul(
        case.xquat, _quat_mul(turn, KEY_SEATED_QUAT))
    data.qpos[_qpos(model, "j_depth_pin")] = .010


def _prepare(model, data, *, rotor, lid, plunger, key_out=True):
    mujoco.mj_resetData(model, data)
    data.qpos[_qpos(model, "j_rotor")] = rotor
    data.qpos[_qpos(model, "j_bolt_slide")] = BOLT_TRAVEL * rotor / ROT_HI
    data.qpos[_qpos(model, "j_lid")] = lid
    data.qpos[_qpos(model, "j_plunger")] = plunger
    data.qpos[_qpos(model, "j_depth_pin")] = .010
    if key_out:
        data.qfrc_applied[_dof(model, "j_depth_pin")] = .12
    if key_out:
        key_qpos = model.jnt_qposadr[model.body("key").jntadr[0]]
        data.qpos[key_qpos] += 0.20
    else:
        _seat_key(model, data, rotor)
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)
    _settle(model, data)


def _apply_joint_torque(model, data, joint, torque, steps=1600):
    address = _dof(model, joint)
    qpos = _qpos(model, joint)
    low = high = float(data.qpos[qpos])
    for _ in range(round(steps * .002 / model.opt.timestep)):
        data.qfrc_applied[address] = torque
        mujoco.mj_step(model, data)
        value = float(data.qpos[qpos])
        low, high = min(low, value), max(high, value)
    data.qfrc_applied[:] = 0
    return low, high, float(data.qpos[qpos])


def test_treasure_chest_declaration_and_generated_visuals():
    task = json.loads((PACK / "task.json").read_text())
    manifest = json.loads((PACK / "manifest.json").read_text())
    scene = SCENE.read_text()

    assert task["language"] == LANGUAGE
    assert task["category"] == "real_world"
    assert [(robot["name"], robot["model"]) for robot in task["robots"]] == [
        ("move", "i2rt_yam"), ("hold", "i2rt_yam")]
    assert {camera.get("robot") for camera in task["cameras"]
            if camera["role"] == "wrist"} == {"move", "hold"}
    assert task["termination"]["max_len"] == 5000
    assert task["table_z"] == pytest.approx(CASE_POS[2])
    assert task["objects"] == ["key"] + [f"coin{i}" for i in range(6)]
    assert "invariant" not in task["success"]
    assert len(task["success"]["of"]) == 4
    seated = task["success"]["of"][0]
    assert {k: seated[k] for k in ("pred", "obj", "rel_to", "pos_tol", "rot_tol_deg")} == {
        "pred": "pose_within", "obj": "key", "rel_to": "case",
        "pos_tol": 0.006, "rot_tol_deg": 8.0,
    }
    assert seated["pos"] == pytest.approx([KEY_SEATED_X, 0.0, LOCK_Z])
    assert seated["quat"] == pytest.approx(KEY_SEATED_QUAT)
    deposit = task["success"]["of"][2]
    assert deposit["pred"] == "count"
    assert deposit["min"] == 4
    assert [leaf["obj"] for leaf in deposit["of"]] == [f"coin{i}" for i in range(6)]
    assert task["randomization"]["groups"][:2] == [["case"], ["key"]]
    assert task["randomization"]["full_rand"] is True
    assert task["randomization"]["placement_regions"]["key"][0] == {
        "x": [0.52, 0.72], "y": [-0.34, -0.10], "yaw_offset_deg": [-180.0, 180.0]}
    assert task["randomization"]["lift_reach"] == {
        "key": 0.12, **{f"coin{i}": 0.14 for i in range(6)}}
    assert "thumb-screw" not in task["language"]
    assert "j_screw" not in scene
    assert "j_lid" in scene
    assert "j_bolt_slide" in scene
    assert "j_plunger" in scene
    assert scene.count('name="lid_shell_') == 12
    assert scene.count('name="lid_wood_plank_') == 12
    assert scene.count('name="lid_brass_arch_0_') == 12
    assert scene.count('name="lid_brass_arch_1_') == 12
    assert 'name="front_brass_lock_plate"' not in scene
    assert 'name="keyway_silhouette_stem"' not in scene
    assert 'name="keyhole_border_side_1"' in scene
    assert 'name="lid_endcap_1"' in scene
    assert 'name="lid_endcap_0"' in scene
    assert 'name="entry_plate_000_s0"' in scene
    assert 'name="entry_plate_000_s1"' in scene
    assert 'name="j_depth_pin"' in scene
    assert 'name="depth_spring"' in scene
    assert 'name="rotor_sleeve_000"' in scene
    assert 'name="key_tip_point"' in scene
    assert scene.count('name="key_insert_grip"') == 0
    assert scene.count('name="key_insert_accent"') == 0
    assert 'name="key_grip_bar"' not in scene
    assert 'name="key_grip_accent"' not in scene
    assert "no pickup nub" in manifest["notes"]
    assert "135 mm tall" in manifest["notes"]
    assert 'file="assets/dark_wood_grain.png"' in scene
    assert 'file="assets/aged_brass.png"' in scene
    for name, dimensions in (("dark_wood_grain.png", (128, 128)),
                             ("aged_brass.png", (64, 64))):
        rel = f"assets/{name}"
        png = (tasks.asset_dir(PACK) / rel).read_bytes()
        assert png[:16] == b"\x89PNG\r\n\x1a\n\0\0\0\rIHDR"
        assert tuple(int.from_bytes(png[i:i + 4], "big") for i in (16, 20)) == dimensions
        assert rel in manifest["files"]
    assert task["success"]["of"][-1]["of"][-1] == {
        "pred": "robot_clear",
        "bodies": [f"coin{i}" for i in range(6)] + ["key", "lid"],
    }

    model = tasks.load_scene(PACK).compile()
    assert model.body("case").pos == pytest.approx(CASE_POS, abs=2e-5)
    assert model.body("case").quat == pytest.approx([-0.3420201433, 0, 0, 0.9396926208])
    assert model.body("rotor").pos[0] == pytest.approx(LOCK_X)
    assert model.geom("case_rear").size[2] * 2 == pytest.approx(.135)
    assert model.geom("key_shaft").type == mujoco.mjtGeom.mjGEOM_CYLINDER
    assert model.geom("key_shaft").size[1] * 2 == pytest.approx(.050)
    for side in (0, 1):
        assert model.geom(f"lid_endcap_{side}").contype[0] != 0


def test_key_cavity_is_open_through_insertion_and_quarter_turn(mechanism):
    model, _ = mechanism
    data = mujoco.MjData(model)
    key = model.body("key").id
    qa = model.jnt_qposadr[model.body("key").jntadr[0]]
    for angle, x in ([(0., x) for x in np.linspace(.135, .075, 121)]
                     + [(a, .075) for a in np.linspace(0, np.pi/2, 91)]):
        data.qpos[_qpos(model, "j_rotor")] = angle
        data.qpos[_qpos(model, "j_bolt_slide")] = BOLT_TRAVEL * angle / ROT_HI
        _seat_key(model, data, angle)
        data.qpos[qa:qa+3] += data.body("case").xmat.reshape(3, 3)[:, 0] * (x-.075)
        mujoco.mj_forward(model, data)
        for c in data.contact:
            if key in (model.geom_bodyid[c.geom1], model.geom_bodyid[c.geom2]):
                assert c.dist >= -1e-6, (angle, x, model.geom(c.geom1).name,
                                        model.geom(c.geom2).name, c.dist)


def test_locked_lid_physically_stays_shut_and_unlocked_lid_reaches_stop(mechanism):
    model, data = mechanism
    for i in range(12):
        shell = model.geom(f"lid_shell_{i:02d}")
        assert shell.contype and shell.conaffinity
        assert model.body(int(shell.bodyid[0])).name == "lid"

    _prepare(model, data, rotor=0.0, lid=0.0, plunger=0.0)
    _, locked_peak, locked_end = _apply_joint_torque(model, data, "j_lid", 0.6)

    _prepare(model, data, rotor=ROT_HI, lid=0.0, plunger=0.0)
    _, unlocked_peak, unlocked_end = _apply_joint_torque(model, data, "j_lid", 0.6)

    assert locked_peak < 0.0065
    assert locked_end < 0.0045
    assert unlocked_peak > 1.70
    assert unlocked_end > 1.70


def test_open_lid_physically_blocks_locking_but_closed_lid_does_not(mechanism):
    model, data = mechanism

    _prepare(model, data, rotor=ROT_HI, lid=0.0, plunger=0.0)
    closed_low, _, closed_end = _apply_joint_torque(
        model, data, "j_rotor", -0.20, steps=2400)

    _prepare(model, data, rotor=ROT_HI, lid=LID_OPEN, plunger=0.022)
    open_low, _, open_end = _apply_joint_torque(
        model, data, "j_rotor", -0.20, steps=2400)

    assert closed_low < 0.05
    assert closed_end < 0.05
    assert open_low > 0.60
    assert open_end > 0.60
    assert data.qpos[_qpos(model, "j_plunger")] > 0.020


def test_key_drives_rotor_and_bolt_slave_without_leaving_keyway(mechanism):
    model, data = mechanism
    _prepare(model, data, rotor=0.0, lid=0.0, plunger=0.0, key_out=False)
    key = model.body("key").id
    keyway = model.site("keyway").id
    axis = data.body("case").xmat.reshape(3, 3)[:, 0]

    for _ in range(round(4000 * .002 / model.opt.timestep)):
        data.xfrc_applied[key, :3] = -.12 * axis
        data.xfrc_applied[key, 3:] = 0.12 * axis
        mujoco.mj_step(model, data)
    data.xfrc_applied[:] = 0

    rotor = float(data.qpos[_qpos(model, "j_rotor")])
    bolt = float(data.qpos[_qpos(model, "j_bolt_slide")])
    assert rotor > 1.50
    assert bolt > 0.029
    assert np.all(np.abs(data.xpos[key] - data.site_xpos[keyway]) < [0.022, 0.011, 0.016])


def test_depth_pin_blocks_direct_rotor_torque_without_a_seated_key(mechanism):
    model, _ = mechanism
    data = mujoco.MjData(model)
    _, peak, final = _apply_joint_torque(model, data, "j_rotor", .12, steps=1000)
    assert peak < .005
    assert abs(final) < .005
    assert abs(data.qpos[_qpos(model, "j_depth_pin")]) < 1e-5


def _fresh_env():
    env = SimEnv(TASK, seed=0)
    env.reset(seed=0, randomize_init=False)
    return env


def _set_lock(env, rotor):
    env.data.qpos[_qpos(env.model, "j_rotor")] = rotor
    env.data.qpos[_qpos(env.model, "j_bolt_slide")] = BOLT_TRAVEL * rotor / ROT_HI
    _seat_key(env.model, env.data, rotor)


def _set_coins_inside(env, count=6):
    cavity = env.data.site("cavity").xpos.copy()
    for coin in range(count):
        qpos = env.model.jnt_qposadr[env.model.body(f"coin{coin}").jntadr[0]]
        env.data.qpos[qpos : qpos + 3] = cavity + [
            0.036 * (coin % 3 - 1),
            0.030 * (coin // 3 - 0.5),
            0.006,
        ]
        env.data.qpos[qpos + 3 : qpos + 7] = [1, 0, 0, 0]


def _forward_and_check(env, repeats=3):
    env.data.qvel[:] = 0
    mujoco.mj_forward(env.model, env.data)
    for _ in range(repeats):
        env._check(env)


def test_four_stage_sequence_rejects_three_coins_and_accepts_five():
    env = _fresh_env()
    check = env._check
    assert (check.at, check.dead) == (0, False)

    _set_lock(env, 0.0)
    _forward_and_check(env)
    assert (check.at, check.dead) == (1, False)

    _set_lock(env, 1.56)
    _forward_and_check(env)
    assert (check.at, check.dead) == (2, False)

    env.data.qpos[_qpos(env.model, "j_lid")] = 1.70
    env.data.qpos[_qpos(env.model, "j_plunger")] = 0.022
    _set_coins_inside(env, 3)
    _forward_and_check(env)
    assert (check.at, check.dead) == (2, False)

    _set_coins_inside(env, 5)
    _forward_and_check(env)
    assert (check.at, check.dead) == (3, False)

    env.data.qpos[_qpos(env.model, "j_lid")] = 0.0
    env.data.qpos[_qpos(env.model, "j_plunger")] = 0.0
    _set_lock(env, 0.0)
    _forward_and_check(env)
    assert (check.at, check.dead) == (4, False)


def test_randomized_key_starts_flat_on_table_with_independent_yaw():
    headings = []
    for seed in range(12):
        env = SimEnv(TASK, seed=seed)
        env.reset(seed=seed, randomize_init=True)
        key = env.data.body("key")
        case = env.data.body("case")
        rotation = key.xmat.reshape(3, 3)
        headings.append(float(np.arctan2(rotation[1, 0], rotation[0, 0])))
        assert key.xpos[2] == pytest.approx(CASE_POS[2] + 0.006, abs=0.001)
        assert abs(rotation[2, 1]) > 0.98
        assert np.linalg.norm(key.xpos[:2] - case.xpos[:2]) >= 0.18 - 1e-6
        assert key.xpos[1] < case.xpos[1] - 0.15
    assert np.ptp(np.unwrap(sorted(headings))) > np.deg2rad(90)


def test_partial_wrong_order_brush_is_forgiven_not_permanently_dead():
    env = _fresh_env()
    env.data.qpos[_qpos(env.model, "j_lid")] = 0.05
    _set_lock(env, 0.40)
    _forward_and_check(env)
    assert (env._check.at, env._check.dead) == (0, False)


