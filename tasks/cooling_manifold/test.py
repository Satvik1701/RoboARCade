"""Tests of the cooling_manifold task."""
from pathlib import Path
import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R
from core.grade_episode import _grasped_by
from core.sim_env import SimEnv
from helpers import (
    _probe_ik,
    _table_top_z,
)
from core.grade_episode import make_checker


GRIP_TASK = "cooling_manifold"


PIECE = "part5"


GRIP_SPAWNS_X_Y_YAW = [(0.06, -0.12, 0.0), (0.12, 0.08, 0.6), (-0.04, 0.14, -1.1)]


GRIP_CASES = [(GRIP_TASK, PIECE)]


GRIP_POSES = 8


GRIP_SETTLE = 60


JAW_SPINS_DEG = [20, 40, 60, 90]


JAW_SPIN_MAX_DEV_M = 0.030


def _piece_qadr(env, piece):
    return env.model.joint(env.model.body(piece).jntadr[0]).qposadr[0]


def _spawn_piece(env, piece, x, y, yaw):
    adr = _piece_qadr(env, piece)
    env.data.qpos[adr:adr + 3] = [x, y, _table_top_z(env) + 0.0175]
    env.data.qpos[adr + 3:adr + 7] = R.from_euler("z", yaw).as_quat()[[3, 0, 1, 2]]
    env.data.qvel[:] = 0
    mujoco.mj_forward(env.model, env.data)
    for _ in range(GRIP_SETTLE):
        mujoco.mj_step(env.model, env.data)


def _jaw_axis_yaw(env, arm):
    a, b = (env.data.body(f"{arm}_{side}_finger").xpos for side in ("left", "right"))
    return float(np.arctan2(b[1] - a[1], b[0] - a[0]))


def _grasp_piece(env, arm, piece):
    home = env.observe_proprio(arm)
    p = env.data.body(piece).xpos.copy()
    yaw = R.from_quat(env.data.body(piece).xquat[[1, 2, 3, 0]]).as_euler("xyz")[2]
    delta = (yaw + np.pi / 2) - _jaw_axis_yaw(env, arm)
    grasp_rot = R.from_euler("z", delta) * R.from_euler("xyz", home.eef_euler)
    e = grasp_rot.as_euler("xyz")
    above, at = p + [0, 0, 0.12], p.copy()
    env.move_to(above, e, 1.0)
    env.move_to(at, e, 1.0)
    env.move_to(at, e, 0.0)
    env.move_to(p + [0, 0, 0.10], e, 0.0)
    return e


def test_contact_retry_exemption_requires_closed_grasp():
    env = SimEnv(task=GRIP_TASK, seed=0)
    env.reset(seed=0, randomize_init=False)
    env.max_len = 10**9
    arm = env.arms[0]
    _spawn_piece(env, PIECE, *GRIP_SPAWNS_X_Y_YAW[0])
    _grasp_piece(env, arm, PIECE)
    assert _grasped_by(env, PIECE, arm)

    payload = {int(g) for g in env._obj_geoms[PIECE]}
    assert env.ctl._held_payload_geoms(arm) == payload

    env.curr_gripper_open[arm] = 1
    assert env.ctl._held_payload_geoms(arm) == set()


def run_grip_trial(spawn, case=None, n_poses=GRIP_POSES, seed=0, verbose=False):
    task, piece = case or (GRIP_TASK, PIECE)
    x, y, yaw = spawn
    rng = np.random.default_rng(seed)
    env = SimEnv(task=task, seed=seed)
    env.reset(seed=seed, randomize_init=False)
    env.max_len = 10**9
    arm = env.arms[0]
    _spawn_piece(env, piece, x, y, yaw)
    _grasp_piece(env, arm, piece)
    if not _grasped_by(env, piece, arm):
        return {"spawn": spawn, "grasped": False, "lost_at": None, "flown": 0,
                "rot": [], "unreachable": 0}

    table_z = _table_top_z(env)
    flown, rots, unreachable = 0, [], 0
    for i in range(n_poses):
        here = env.observe_proprio(arm)
        for _ in range(40):
            pos = np.empty(3)
            pos[:2] = here.eef_pos[:2] + rng.uniform(-0.12, 0.12, 2)
            pos[2] = rng.uniform(table_z + 0.12, table_z + 0.40)
            rot = (R.from_rotvec(rng.uniform(-1, 1, 3) * [0.6, 0.6, np.pi / 2])
                   * R.from_euler("xyz", here.eef_euler))
            if _probe_ik(env, arm, pos, rot, env.data.qpos[env.ik[arm].qadr].copy(),
                         ignore=(piece,)) is not None:
                break
        else:
            unreachable += 1
            continue
        turn = float(np.degrees(np.linalg.norm(
            (rot * R.from_euler("xyz", here.eef_euler).inv()).as_rotvec())))
        env.move_to(pos, rot.as_euler("xyz"), 0.0)
        flown += 1
        rots.append(turn)
        if verbose:
            print(f"    pose #{i}: turned {turn:5.1f} deg  "
                  f"held={_grasped_by(env, piece, arm)}")
        if not _grasped_by(env, piece, arm):
            return {"spawn": spawn, "grasped": True, "lost_at": i, "flown": flown,
                    "rot": rots, "unreachable": unreachable, "lost_after_deg": turn}
    return {"spawn": spawn, "grasped": True, "lost_at": None, "flown": flown,
            "rot": rots, "unreachable": unreachable}


def _jaw_offset(env, arm, piece):
    a, b = (env.data.body(f"{arm}_{side}_finger").xpos for side in ("left", "right"))
    return float(np.linalg.norm(env.data.body(piece).xpos - (a + b) / 2))


class _Excursion:
    def __init__(self, env, arm, at):
        self.env, self.arm, self.at, self.max_dev = env, arm, at, 0.0

    def record(self, *a, **k):
        p = self.env.observe_proprio(self.arm).eef_pos
        self.max_dev = max(self.max_dev, float(np.linalg.norm(p - self.at)))


def run_jaw_spin_trial(spawn, case=None, angles=JAW_SPINS_DEG, seed=0, verbose=False):
    task, piece = case or (GRIP_TASK, PIECE)
    x, y, yaw = spawn
    env = SimEnv(task=task, seed=seed)
    env.reset(seed=seed, randomize_init=False)
    env.max_len = 10**9
    arm = env.arms[0]
    _spawn_piece(env, piece, x, y, yaw)
    _grasp_piece(env, arm, piece)
    if not _grasped_by(env, piece, arm):
        return {"spawn": spawn, "grasped": False, "results": []}

    here = env.observe_proprio(arm)
    env.move_to(here.eef_pos + np.array([0.0, 0.0, 0.15]), here.eef_euler, 0.0)
    at = env.observe_proprio(arm).eef_pos.copy()
    base = _jaw_offset(env, arm, piece)
    out = []
    carry = env.observe_proprio(arm).eef_euler.copy()
    for deg in angles:
        if not np.allclose(env.observe_proprio(arm).eef_euler, carry, atol=1e-3):
            env.move_to(at, carry, 0.0)
        a, b = (env.data.body(f"{arm}_{side}_finger").xpos for side in ("left", "right"))
        axis = (b - a) / np.linalg.norm(b - a)
        rot = R.from_rotvec(axis * np.radians(deg)) * R.from_euler("xyz", carry)
        if _probe_ik(env, arm, at, rot, env.data.qpos[env.ik[arm].qadr].copy(),
                     ignore=(piece,), check_path=False) is None:
            out.append({"deg": deg, "held": _grasped_by(env, piece, arm), "slip_mm": 0.0,
                        "dev_mm": 0.0, "unreachable": True})
            continue
        probe = _Excursion(env, arm, at)
        observe = env.observe
        env.observe = lambda *a, **k: {}
        try:
            env.move_to(at, rot.as_euler("xyz"), 0.0, recorder=probe)
        finally:
            env.observe = observe
        held = _grasped_by(env, piece, arm)
        slip = _jaw_offset(env, arm, piece) - base
        out.append({"deg": deg, "held": held, "slip_mm": slip * 1000,
                    "dev_mm": probe.max_dev * 1000})
        if verbose:
            print(f"    spin {deg:+4d} deg -> held={held}  slip {slip * 1000:+6.1f} mm  "
                  f"eef wandered {probe.max_dev * 1000:6.1f} mm")
        if not held:
            break
    return {"spawn": spawn, "grasped": True, "results": out}


@pytest.mark.parametrize("case", GRIP_CASES, ids=lambda c: c[1])
@pytest.mark.parametrize("spawn", GRIP_SPAWNS_X_Y_YAW, ids=lambda s: f"xy{s[0]}_{s[1]}")
def test_held_piece_survives_jaw_axis_rotation(spawn, case):
    r = run_jaw_spin_trial(spawn, case)
    assert r["grasped"], f"heuristic grasp failed at spawn {spawn} — nothing to test"
    lost = [x for x in r["results"] if not x["held"]]
    assert not lost, (
        f"piece spun out of the jaws at {lost[0]['deg']} deg about the opening axis "
        f"(slip {lost[0]['slip_mm']:.0f} mm); trace: "
        + ", ".join(f"{x['deg']:+d}deg slip {x['slip_mm']:+.0f}mm" for x in r["results"]))


@pytest.mark.parametrize("case", GRIP_CASES, ids=lambda c: c[1])
@pytest.mark.parametrize("spawn", GRIP_SPAWNS_X_Y_YAW, ids=lambda s: f"xy{s[0]}_{s[1]}")
def test_in_place_rotation_stays_in_place(spawn, case):
    r = run_jaw_spin_trial(spawn, case)
    assert r["grasped"], f"heuristic grasp failed at spawn {spawn} — nothing to test"
    bad = [x for x in r["results"] if x["dev_mm"] > JAW_SPIN_MAX_DEV_M * 1000]
    assert not bad, (
        f"'rotate in place' moved the gripper {bad[0]['dev_mm']:.0f} mm at "
        f"{bad[0]['deg']} deg (cap {JAW_SPIN_MAX_DEV_M * 1000:.0f} mm); trace: "
        + ", ".join(f"{x['deg']:+d}deg {x['dev_mm']:.0f}mm" for x in r["results"]))


@pytest.mark.parametrize("case", GRIP_CASES, ids=lambda c: c[1])
@pytest.mark.parametrize("spawn", GRIP_SPAWNS_X_Y_YAW, ids=lambda s: f"xy{s[0]}_{s[1]}")
def test_held_piece_survives_random_poses(spawn, case):
    r = run_grip_trial(spawn, case)
    assert r["grasped"], f"heuristic grasp failed at spawn {spawn} — nothing to test"
    assert r["flown"] >= GRIP_POSES // 2, (
        f"only {r['flown']}/{GRIP_POSES} poses were reachable while carrying the piece "
        f"({r['unreachable']} rejected) — the trial proved nothing")
    assert r["lost_at"] is None, (
        f"piece left the jaws at pose #{r['lost_at']} of {r['flown']} "
        f"(that pose turned the wrist {r.get('lost_after_deg', 0):.0f} deg; "
        f"rotations flown: {[round(x) for x in r['rot']]})")


@pytest.fixture(scope="module")
def states():
    return np.load(Path(__file__).parent / "manifold_contact_states.npz")


def test_socket_pairs_preserve_gripper_contacts(states):
    env = SimEnv("cooling_manifold")
    m, d = env.model, env.data
    cap, socket = m.body("part5").id, m.body("part1").id
    assert m.npair > 0
    for g1, g2 in zip(m.pair_geom1, m.pair_geom2):
        assert {int(m.geom_bodyid[g1]), int(m.geom_bodyid[g2])} == {cap, socket}
    np.testing.assert_allclose(m.pair_friction[:, 2], .002)
    np.testing.assert_allclose(m.geom_friction[m.geom_bodyid == cap, 1][1:], .3)
    env.reset_to_sim_state(states["grasp"])
    mujoco.mj_step(m, d, nstep=round(.2 / m.opt.timestep))
    touched = set()
    for c in d.contact:
        b1, b2 = m.geom_bodyid[[c.geom1, c.geom2]]
        if cap not in (b1, b2):
            continue
        other = int(b2 if b1 == cap else b1)
        if other == socket or env.finger_root(other) is not None:
            touched.add(m.body(other).name)
            expected = .002 if other == socket else .3
            np.testing.assert_allclose(c.friction, [.6, .6, expected, .0001, .0001])
            np.testing.assert_allclose(c.solref, [.004, 1])
            np.testing.assert_allclose(c.solimp[:2], [.998, .998])
    assert touched == {"part1", "robot0_left_finger", "robot0_right_finger"}


@pytest.mark.parametrize("torsion,passes", [(.3, True), (.002, True)])
def test_recorded_cap_seats_after_release(states, torsion, passes):
    env = SimEnv("cooling_manifold")
    env.model.pair_friction[:, 2] = torsion
    env.reset_to_sim_state(states["released"])
    mujoco.mj_step(env.model, env.data, nstep=round(2 / env.model.opt.timestep))
    assert bool(make_checker(env.spec["success"])[0](env)) == passes
    ref, cap = env.data.body("part1"), env.data.body("part5")
    delta = ref.xmat.reshape(3, 3).T @ (cap.xpos - ref.xpos)
    height_error = delta[2] - env.spec["success"]["of"][0]["pos"][2]
    assert abs(height_error) < .001 if passes else height_error > .008
