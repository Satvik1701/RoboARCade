"""Tests of the dual_stack task."""
import numpy as np
from scipy.spatial.transform import Rotation as R
from core.sim_env import SimEnv
from helpers import (
    Episode,
    project,
)
import pytest
import mujoco
from core.grade_episode import hold_steps, make_checker
from helpers import (  # noqa: F401 (ds_env and goal_state are fixtures)
    ds_env,
    goal_state,
    set_free_body,
    zero_free_qvel,
)
from core.grade_episode import _robot_touching


pytestmark = pytest.mark.e2e


def _fold(deg, period=90.0):
    return (deg + period / 2) % period - period / 2


def _yaw_deg(env, body):
    q = env.data.body(body).xquat
    return R.from_quat([q[1], q[2], q[3], q[0]]).as_euler("zyx", degrees=True)[0]


def pick_and_place(ep, green, place_xy, place_z, diamond_yaw, udir, pre_yaw=0):
    env, ui, k = ep.env, ep.ui, ep.k
    gp = k.object_poses()[green]
    u, v = project(env, "agentview", gp + [0, 0, 0.025])
    ui.gripper_teleport_via_click("agentview", u, v)
    ep.do(f"teleport above {green}")
    gyaw = _fold(_yaw_deg(env, green))
    grasp_rot = _fold(gyaw + pre_yaw, 180.0)
    if abs(grasp_rot) > 0.5:
        ui.gripper_rotate("z", grasp_rot)
        ep.do(f"pre-rotate {grasp_rot:.0f}")
    ui.gripper_advance_or_retreat(0.09)
    ep.do("descend (coarse)")
    ui.gripper_advance_or_retreat(0.035)
    ep.do("descend")
    ui.gripper_toggle()
    ep.do("close")
    assert env.gripper_width() > 0.03, f"grasp missed {green}"
    ui.gripper_translate(0, 0, 0.15)
    ep.do("lift block")
    place_rot = _fold(diamond_yaw - grasp_rot, 180.0)
    ui.gripper_rotate("z", place_rot)
    ep.do("rotate to diamond")
    off = k.object_poses()[green][:2] - k.proprio().eef_pos[:2]
    aim = np.asarray(place_xy) - off
    ui.target_pos = np.array([*aim, ui.target_pos[2]])
    ep.do("move over red")
    ui.target_pos = np.array([*aim, place_z + 0.03])
    ep.do("lower (coarse)")
    ui.target_pos = np.array([*aim, place_z])
    ep.do("lower onto red")
    for _ in range(2):
        err = k.object_poses()[green][:2] - np.asarray(place_xy)
        if np.linalg.norm(err) < 0.0015:
            break
        ui.gripper_translate(-err[0], -err[1], 0)
        ep.do(f"nudge {np.round(err * 1000, 1)}mm")
    ui.gripper_toggle()
    ep.do("open")
    ui.gripper_translate(0, 0, 0.025)
    ep.do("pop up")
    vdir = np.array([udir[1], -udir[0]])
    ui.gripper_translate(*(0.0396 * vdir), 0.095)
    ep.do("retreat")
    ui.target_pos = np.array([-0.10, 0.0, 1.0])
    ep.do("park at home")
    ui.gripper_rotate("z", -(grasp_rot + place_rot))
    ep.do("rotate back")


def test_dual_stack_predicate():
    print("=== dual_stack success predicate (state probe) ===")
    import mujoco
    env = SimEnv("dual_stack", seed=0)
    env.reset()
    spec = env.spec["success"]
    rq = env.model.joint(env.model.body(spec["base"]).jntadr[0]).qposadr[0]
    env.data.qpos[rq:rq + 3] = [0, 0, 0.825]
    env.data.qpos[rq + 3:rq + 7] = [1, 0, 0, 0]
    for g, s in zip(spec["greens"], (1, -1)):
        q = env.model.joint(env.model.body(g).jntadr[0]).qposadr[0]
        env.data.qpos[q:q + 3] = [s * 0.019, s * 0.019, 0.8765]
        quat = R.from_euler("z", 45, degrees=True).as_quat()
        env.data.qpos[q + 3:q + 7] = [quat[3], quat[0], quat[1], quat[2]]
    mujoco.mj_forward(env.model, env.data)
    env._success_streak = 0
    fired = False
    for _ in range(400):
        mujoco.mj_step(env.model, env.data)
        env._success_streak = env._success_streak + 1 if env._check(env) else 0
        fired = fired or env._success_streak >= hold_steps(env)
    assert fired and env._success_streak >= hold_steps(env), "predicate did not fire on the goal state"
    env.reset()
    env._success_streak = 0
    assert not env._check(env), "predicate fired on the initial state"
    print("  PASS: fires+holds on the diamond goal state; False on the start state\n")


def test_dual_stack(seed=0):
    print(f"=== dual_stack (default interface, seed {seed}) ===")
    ep = Episode("dual_stack", seed=seed)
    env, k = ep.env, ep.k

    poses = k.object_poses()
    red = poses["block_blue_main"]
    print(f"  red {np.round(red, 3)}  greens "
          f"{np.round(poses['block_green1_main'], 3)} {np.round(poses['block_green2_main'], 3)}")

    PLACE_Z = red[2] + 0.055 + 0.006
    D = 0.019 * np.sqrt(2)
    psi = _fold(_yaw_deg(env, "block_blue_main"))
    ang = np.radians(psi + 45.0)
    u = np.array([np.cos(ang), np.sin(ang)])
    pick_and_place(ep, "block_green1_main", red[:2] + D * u, PLACE_Z,
                   diamond_yaw=psi + 45.0, udir=u)
    poses = k.object_poses()
    red2, g1 = poses["block_blue_main"], poses["block_green1_main"]
    psi = _fold(_yaw_deg(env, "block_blue_main"))
    ang = np.radians(psi + 45.0)
    u = np.array([np.cos(ang), np.sin(ang)])
    a2 = min(-D, (g1[:2] - red2[:2]) @ u - 0.05 - 0.010)
    pick_and_place(ep, "block_green2_main", red2[:2] + a2 * u, PLACE_Z + 0.012,
                   diamond_yaw=psi + 45.0, udir=u, pre_yaw=90)

    ep.wait_success()
    poses = k.object_poses()
    print(f"  settled: red {np.round(poses['block_blue_main'], 3)}  greens "
          f"{np.round(poses['block_green1_main'], 3)} {np.round(poses['block_green2_main'], 3)}")
    assert make_checker(env.spec["success"], env.task)[0](env), "dual_stack did not succeed"
    ep.save_video(f"dual_stack_seed{seed}")
    print(f"  PASS: steps={env.num_step}\n")


def set_free_qvel_lin(env, name, v):
    jadr = env.model.body(name).jntadr[0]
    vadr = env.model.joint(jadr).dofadr[0]
    env.data.qvel[vadr : vadr + 3] = v


def test_dual_stack_goal_fires(goal_state):
    env, spec = goal_state
    check, _ = make_checker(spec, "dual_stack")
    assert check(env) is True


def test_dual_stack_start_state_false(ds_env):
    env = ds_env
    env.reset(seed=0)
    check, _ = make_checker(env.spec["success"], env.task)
    assert check(env) is False


def test_dual_stack_clause_height(goal_state):
    env, spec = goal_state
    check, _ = make_checker(spec, "dual_stack")
    g = spec["greens"][0]
    set_free_body(env, g, [0.019, 0.019, 0.8765 - spec["dz_tol"] - 0.005], yaw_deg=45)
    mujoco.mj_forward(env.model, env.data)
    assert check(env) is False


def test_dual_stack_clause_xy(goal_state):
    env, spec = goal_state
    check, _ = make_checker(spec, "dual_stack")
    set_free_body(env, spec["greens"][0],
                  [0.019 + spec["xy_tol"] + 0.01, 0.019, 0.8765], yaw_deg=45)
    mujoco.mj_forward(env.model, env.data)
    assert check(env) is False


def test_dual_stack_clause_same_level(goal_state):
    env, spec = goal_state
    assert 2 * 0.008 > spec["same_level_tol"] and 0.008 < spec["dz_tol"]
    check, _ = make_checker(spec, "dual_stack")
    set_free_body(env, spec["greens"][0], [0.019, 0.019, 0.8765 + 0.008], yaw_deg=45)
    set_free_body(env, spec["greens"][1], [-0.019, -0.019, 0.8765 - 0.008], yaw_deg=45)
    mujoco.mj_forward(env.model, env.data)
    assert check(env) is False


def test_dual_stack_clause_gripper_open(goal_state):
    env, spec = goal_state
    check, _ = make_checker(spec, "dual_stack")
    fingers = [env.model.joint(f"{env.robot_prefix[0]}finger_joint{i}").qposadr[0] for i in (1, 2)]
    saved = [env.data.qpos[a] for a in fingers]
    for a in fingers:
        env.data.qpos[a] = 0.01
    mujoco.mj_forward(env.model, env.data)
    assert check(env) is False
    for a, v in zip(fingers, saved):
        env.data.qpos[a] = v


def test_dual_stack_clause_at_rest(goal_state):
    env, spec = goal_state
    check, _ = make_checker(spec, "dual_stack")
    set_free_qvel_lin(env, spec["greens"][0], [spec["lin_vel_max"] * 3, 0, 0])
    assert check(env) is False
    zero_free_qvel(env, spec["greens"][0])
    assert check(env) is True


def test_robot_touching_helper(ds_env):
    env = ds_env
    env.reset(seed=0)
    block = env.spec["success"]["greens"][0]
    assert _robot_touching(env, [block]) is False
    hand = env.data.body(f"{env.robot_prefix[0]}hand").xpos.copy()
    set_free_body(env, block, hand)
    mujoco.mj_forward(env.model, env.data)
    assert _robot_touching(env, [block]) is True


def test_verdict_pure_function_of_state(goal_state):
    env, spec = goal_state
    check, _ = make_checker(spec, "dual_stack")
    state = env.sim_state()
    first = check(env)
    env.reset(seed=3)
    env.reset_to_sim_state(state)
    assert check(env) is first is True
