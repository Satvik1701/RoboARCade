"""Tests of the open_middle_drawer_halfway task."""
import numpy as np
from types import SimpleNamespace
import pytest
from core.sim_env import SimEnv
import base64
import zlib
import mujoco
from scipy.spatial.transform import Rotation as R
from core.sim_env_server import InteractiveBot
from core.sim_env import Phase
from core.grade_episode import make_checker
from core.tasks import TASKS


@pytest.mark.parametrize("q0,target_pos,target_euler,expected", [
    pytest.param(
        [1.9739983787759254, -0.27041572686230403, -1.3670175668588296,
         -2.421531146521584, -2.897153939643744, 2.316703164764432,
         1.9767882756770085],
        [-0.1050, -0.0746, 1.2500],
        [-1.56 - np.pi, -0.06, 2.37],
        False,
        id="free_space_branch_flip",
    ),
    pytest.param(
        [0.6114371318788033, 0.8518804283835558, 0.36223387077778324,
         -1.9017305712842716, -1.6302688883831606, 1.3043718969531366,
         2.0114067431041134],
        [-0.1329, 0.2582, 1.0088],
        [-1.57 - np.pi, -0.08, 0.80],
        True,
        id="reachable_contact_stop",
    ),
])
def test_reaches_replays_recorded_drawer_paths(q0, target_pos, target_euler, expected):
    env = SimEnv(task="open_middle_drawer_halfway", seed=1000008)
    env.reset(seed=1000008)

    ok, _, _ = env.ctl.reaches(
        env.arms[0], np.asarray(target_pos), np.asarray(target_euler), q0=np.asarray(q0))

    assert ok is expected


CASES = (
    ("GO", 1, 5),
    ("GO", 1, 24),
    ("GO", 1, 28),
    ("GO", 1, 42),
    ("GO", 1, 43),
    ("GO", 8, 7),
    ("GS", 3, 30),
    ("GS", 4, 14),
    ("GS", 7, 47),
    ("GS", 7, 48),
    ("GS", 8, 10),
    ("GS", 8, 18),
    ("GS", 8, 36),
    ("GS", 9, 1),
    ("GS", 9, 11),
    ("GS", 9, 28),
)

RED_CASES = (
    ("GO", 3, 1, 65),
    ("GO", 3, 6, 257),
    ("GO", 8, 7, 231),
    ("GS", 0, 11, 303),
    ("GS", 0, 31, 1089),
    ("GS", 2, 54, 1686),
    ("GS", 4, 34, 1386),
    ("GS", 7, 40, 1546),
    ("GS", 7, 47, 1932),
    ("GS", 8, 4, 204),
)

_RED_STATE_B64 = """
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
"""

_STATE_B64 = """
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
wD1yzJXNLhP76v8Avgt7Bg==
"""


def _states():
    raw = zlib.decompress(base64.b64decode("".join(_STATE_B64.split())))
    return np.frombuffer(raw, dtype="<f4").reshape(len(CASES), 100)


def _red_states():
    raw = zlib.decompress(base64.b64decode("".join(_RED_STATE_B64.split())))
    return np.frombuffer(raw, dtype="<f4").reshape(len(RED_CASES), 100)


def _restore(env, row):
    m, d = env.model, env.data
    i = 0
    d.time = float(row[i])
    i += 1
    d.qpos[:] = row[i:i + m.nq]
    i += m.nq
    d.qvel[:] = row[i:i + m.nv]
    i += m.nv
    d.ctrl[:] = row[i:i + m.nu]
    d.qacc_warmstart[:] = 0.0
    mujoco.mj_forward(m, d)
    arm = env.arms[0]
    env.ik[arm].q_des = d.qpos[env.ik[arm].qadr].copy()
    env.curr_gripper_open[arm] = 1
    env.num_step = 0
    env.max_len = 10**9
    env.phase = Phase.RUNNING


def _under(model, body, root):
    while body > 0:
        if body == root:
            return True
        body = int(model.body_parentid[body])
    return False


def _arm_touches_cabinet(env, arm):
    root = env.model.body("wooden_cabinet_1_main").id
    return any(
        _under(env.model, int(env.model.geom_bodyid[g]), root)
        for g in env.ctl._arm_contacts(arm)
    )


def _cabinet_motion(env, action):
    root = env.model.body("wooden_cabinet_1_main").id
    start = env.data.xpos[root].copy()
    maximum = 0.0
    original = env._step_physics

    def step():
        nonlocal maximum
        original()
        maximum = max(maximum, float(np.linalg.norm(env.data.xpos[root] - start)))

    env._step_physics = step
    try:
        action()
    finally:
        env._step_physics = original
    return maximum * 1000.0


def test_home_pose_config_is_exact_fk_and_does_not_mutate_live_state():
    env = SimEnv("open_middle_drawer_halfway", seed=0)
    env.reset(seed=0)
    arm = env.arms[0]
    qpos = env.data.qpos.copy()
    pos, euler = env.ctl.home_pose(arm)

    bot = InteractiveBot.__new__(InteractiveBot)
    bot.env = env
    bot._ui_z_offset = float(env.pc["z_offset"])
    pose = bot.collect_home_poses()[arm]

    np.testing.assert_array_equal(env.data.qpos, qpos)
    np.testing.assert_allclose(
        bot.transform_uiframe_to_robotframe(
            np.asarray(pose["fingertip_pos_ui"]).reshape(1, 3)).squeeze(),
        pos,
        atol=1e-12,
    )
    np.testing.assert_allclose(pose["ee_euler_ui"], euler + [np.pi, 0.0, 0.0])
    assert not hasattr(env.ctl, "go_home")
    assert hasattr(env.ctl, "_reconfigure_home")


def test_all_red_archived_homes_use_joint_fallback(monkeypatch):
    env = SimEnv("open_middle_drawer_halfway", seed=0)
    env.reset(seed=0)
    env._check = lambda _env: False
    arm = env.arms[0]
    target_pos, target_euler = env.ctl.home_pose(arm)
    ik = env.ik[arm]

    def reject_straight(*_args, **_kwargs):
        pytest.fail("red home called the straight move")

    monkeypatch.setattr(env.ctl, "move_to", reject_straight)
    seen = []
    for ident, row in zip(RED_CASES, _red_states(), strict=True):
        _restore(env, row)
        assert not env.ctl.reaches(arm, target_pos, target_euler)[0], ident
        gripper_ctrl = env.data.ctrl[env._grip_act[arm]].copy()

        env.ctl.home(arm)

        seen.append(ident)
        assert 100 <= env.num_step <= 200, ident
        np.testing.assert_allclose(ik.q_des, ik.q_home, atol=1e-12)
        if env.num_step < 200:
            assert np.max(np.abs(env.data.qpos[ik.qadr] - ik.q_home)) < env.ctl.HOME_TOL, ident
        np.testing.assert_allclose(env.data.ctrl[ik.act], ik.q_home, atol=1e-12)
        np.testing.assert_array_equal(env.data.ctrl[env._grip_act[arm]], gripper_ctrl)
        assert env.curr_gripper_open[arm] == 1
    assert seen == list(RED_CASES)


def test_only_the_home_flag_uses_controller_home(monkeypatch, tmp_path):
    env = SimEnv("open_middle_drawer_halfway", seed=0)
    arm = env.arms[0]
    homes, moves = [], []
    monkeypatch.setattr(env.ctl, "home", lambda *a, **k: homes.append((a, k)))
    monkeypatch.setattr(env, "move_to", lambda *a, **k: moves.append((a, k)))
    bot = InteractiveBot.__new__(InteractiveBot)
    bot.env = env
    bot.demo_folder = str(tmp_path)
    bot._ui_z_offset = float(env.pc["z_offset"])
    bot._rec_paused = True
    bot._rec_stopped = False
    bot.recorder = SimpleNamespace(waypoint_idx=0)
    ui = bot.collect_home_poses()[arm]
    x, y, z = ui["fingertip_pos_ui"]
    e0, e1, e2 = ui["ee_euler_ui"]
    cmd = {"arm": arm, "gripper_open": 1.0, "position": {"x": x, "y": y, "z": z},
           "orientation": {"x": e0, "y": e2, "z": -e1}}

    bot.apply_waypoint_mode({**cmd, "home": True})
    bot.apply_waypoint_mode(cmd)

    assert len(homes) == 1 and len(moves) == 1
    (called_arm, jaw), kw = homes[0]
    assert called_arm == arm and jaw == 1.0 and kw["recorder"] is None and callable(kw["stream_fn"])


def test_archived_disturbed_green_homes_use_straight_waypoints(monkeypatch):
    env = SimEnv("open_middle_drawer_halfway", seed=0)
    env.reset(seed=0)
    env._check = lambda _env: False
    arm = env.arms[0]
    target_pos, target_euler = env.ctl.home_pose(arm)
    results = {}

    def reject_fallback(*_args, **_kwargs):
        pytest.fail("green home called the joint fallback")

    monkeypatch.setattr(env.ctl, "_reconfigure_home", reject_fallback)

    for ident, row in zip(CASES, _states(), strict=True):
        _restore(env, row)
        cabinet_contact = _arm_touches_cabinet(env, arm)
        green = env.ctl.reaches(arm, target_pos, target_euler)[0]
        if not green:
            results[ident] = (False, cabinet_contact, None, None)
            continue
        straight_mm = _cabinet_motion(
            env, lambda: env.ctl.home(arm))
        straight_steps = env.num_step

        _restore(env, row)
        start = env.observe_proprio(arm)
        start_rot = R.from_euler("xyz", start.eef_euler)
        idle_mm = _cabinet_motion(
            env,
            lambda: [env.ctl._hold_pose(start.eef_pos, start_rot, 1, arm)
                     for _ in range(straight_steps)],
        )
        results[ident] = (green, cabinet_contact, straight_mm, idle_mm)

    red = {ident for ident, (green, *_rest) in results.items() if not green}
    assert red == {("GO", 8, 7), ("GS", 7, 47)}

    hooked = {
        ident
        for ident, (green, contact, straight_mm, _idle_mm) in results.items()
        if green and contact and straight_mm >= 10.0
    }
    assert hooked == {
        ("GO", 1, 24),
        ("GO", 1, 28),
        ("GS", 8, 10),
        ("GS", 8, 36),
        ("GS", 9, 28),
    }

    falling = results[("GS", 7, 48)]
    assert falling[0] and not falling[1]
    assert falling[2] >= 10.0 and falling[3] >= 10.0
    assert abs(falling[2] - falling[3]) < 1.0

    eligible = {
        ident
        for ident, (green, contact, _straight_mm, idle_mm) in results.items()
        if green and not contact and idle_mm < 10.0
    }
    assert eligible == {
        ("GO", 1, 5),
        ("GO", 1, 42),
        ("GS", 3, 30),
        ("GS", 4, 14),
        ("GS", 8, 18),
        ("GS", 9, 1),
    }
    for ident in eligible:
        assert results[ident][2] < 10.0, f"{ident}: straight home moved the cabinet"


def test_libero_open_drawer_region_specific():
    if not (TASKS / "open_middle_drawer_halfway" / "task.json").exists():
        pytest.skip("libero/open_middle_drawer_halfway pack not materialized")
    env = SimEnv("open_middle_drawer_halfway", seed=0)
    env.reset(seed=0)
    spec = env.spec["success"]
    assert spec["pred"] == "joint_in"
    check, _ = make_checker(spec)
    assert check(env) is False

    open_q = (spec["lo"] + spec["hi"]) / 2
    top = env.model.joint("wooden_cabinet_1_top_level").qposadr[0]
    mid = env.model.joint(spec["joint"]).qposadr[0]
    env.data.qpos[top] = open_q
    mujoco.mj_forward(env.model, env.data)
    assert check(env) is False
    env.data.qpos[top] = 0.0
    env.data.qpos[mid] = open_q
    mujoco.mj_forward(env.model, env.data)
    assert check(env) is True
