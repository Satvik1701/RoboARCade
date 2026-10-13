"""Helpers shared by tasks/*/test.py: shared scenes and predicate probes, controller probes, and scripted
solves through the default interface."""
from pathlib import Path

import cv2
import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

from core.control import make_arm_ik
from core.sim_env import Phase, SimEnv
from core.tasks import TASKS
from sim_kernel import Kernel

REPO = Path(__file__).resolve().parents[1]


BOWL = "akita_black_bowl_1_main"


PLATE = "plate_1_main"


@pytest.fixture(scope="session")
def lift_env():
    env = SimEnv("bowl_on_plate", seed=0)
    env.reset(seed=0)
    return env


@pytest.fixture(scope="session")
def ds_env():
    env = SimEnv("dual_stack", seed=0)
    env.reset(seed=0)
    return env


def settle_steps(env, seconds):
    return int(round(seconds / env.model.opt.timestep))


def set_free_body(env, name, pos, yaw_deg=0.0):
    qadr = env.model.joint(env.model.body(name).jntadr[0]).qposadr[0]
    env.data.qpos[qadr : qadr + 3] = pos
    quat = R.from_euler("z", yaw_deg, degrees=True).as_quat()
    env.data.qpos[qadr + 3 : qadr + 7] = [quat[3], quat[0], quat[1], quat[2]]


def zero_free_qvel(env, name):
    jadr = env.model.body(name).jntadr[0]
    vadr = env.model.joint(jadr).dofadr[0]
    env.data.qvel[vadr : vadr + 6] = 0


@pytest.fixture()
def goal_state(ds_env):
    env = ds_env
    env.reset(seed=0)
    spec = env.spec["success"]
    set_free_body(env, spec["base"], [0, 0, 0.825])
    for g, s in zip(spec["greens"], (1, -1)):
        set_free_body(env, g, [s * 0.019, s * 0.019, 0.8765], yaw_deg=45)
    for b in (spec["base"], *spec["greens"]):
        zero_free_qvel(env, b)
    mujoco.mj_forward(env.model, env.data)
    return env, spec


def _pack_env(task):
    if not (TASKS / task / "task.json").exists():
        pytest.skip(f"{task} pack not materialized")
    env = SimEnv(task, seed=0)
    env.reset(seed=0)
    return env


def _press(env, check, joint, times=1):
    j = env.model.joint(joint)
    adr, lo = j.qposadr[0], env.model.jnt_range[j.id][0]
    for _ in range(times):
        for q in (0.0, lo, lo, 0.0):
            env.data.qpos[adr] = q
            mujoco.mj_forward(env.model, env.data)
            check(env)


ROYGBIV = ["cube_red", "cube_orange", "cube_yellow", "cube_green",
           "cube_blue", "cube_indigo", "cube_violet"]


def nodes(spec):
    yield spec
    of = spec.get("of")
    for s in (of if isinstance(of, list) else [of] if of else []):
        yield from nodes(s)
    if spec.get("invariant"):
        yield from nodes(spec["invariant"])


POS_TOL = 0.005


PROBE_ITERS = 300


def _robot_in_contact(env, d, arm, ignore=()):
    body_of = env.model.geom_bodyid
    for c in d.contact[: d.ncon]:
        names = [mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_BODY, body_of[g])
                 for g in (c.geom1, c.geom2)]
        if any(n in ignore for n in names if n):
            continue
        if any(n and n.startswith(arm + "_") for n in names):
            return True
    return False


def _probe_ik(env, arm, goal_pos, goal_rot, q_start, ignore=(), check_path=True):
    p = arm + "_"
    solver = make_arm_ik(env.model, p, env.spec["robots"][env.arms.index(arm)]["home_qpos"],
                         env.kin[arm])
    d = mujoco.MjData(env.model)
    d.qpos[:] = env.data.qpos
    d.qpos[solver.qadr] = q_start
    mujoco.mj_forward(env.model, d)
    q_goal = np.asarray(q_start, dtype=float).copy()
    if check_path:
        ok, _, _ = env.ctl.reaches(
            arm, goal_pos, goal_rot.as_euler("xyz"), q0=q_goal)
        if not ok:
            return None
        site = env._site[arm]
        start_pos = d.site_xpos[site].copy()
        start_rot = R.from_matrix(d.site_xmat[site].reshape(3, 3))
        turn = (goal_rot * start_rot.inv()).as_rotvec()
        for a in np.linspace(0.0, 1.0, env.ctl.REACH_PATH_SAMPLES + 1)[1:]:
            ref_pos = start_pos + (goal_pos - start_pos) * a
            ref_rot = R.from_rotvec(turn * a) * start_rot
            q_goal = solver.solve_pose(q_goal, ref_pos, ref_rot)
            d.qpos[solver.qadr] = q_goal
            mujoco.mj_forward(env.model, d)
            mujoco.mj_collision(env.model, d)
            if _robot_in_contact(env, d, arm, ignore):
                return None
    else:
        solver.q_des = q_goal
        for _ in range(PROBE_ITERS):
            solver.substep(d, goal_pos, goal_rot)
            d.qpos[solver.qadr] = solver.q_des
            mujoco.mj_forward(env.model, d)
        q_goal = np.asarray(solver.q_des, dtype=float).copy()
    site = env._site[arm]
    e_pos = np.linalg.norm(d.site_xpos[site] - goal_pos)
    rot = R.from_matrix(d.site_xmat[site].reshape(3, 3))
    e_rot = np.degrees(np.linalg.norm((goal_rot * rot.inv()).as_rotvec()))
    mujoco.mj_collision(env.model, d)
    if e_pos >= 0.002 or e_rot >= 1.0 or _robot_in_contact(env, d, arm, ignore):
        return None
    return q_goal


def _table_top_z(env):
    m = env.model
    tops = []
    for i in range(m.ngeom):
        body = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[i]) or ""
        geom = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) or ""
        if "table" in (body + geom).lower():
            half = m.geom_size[i][2] if m.geom_type[i] == mujoco.mjtGeom.mjGEOM_BOX else 0.0
            tops.append(env.data.geom_xpos[i][2] + half)
    return max(tops) if tops else 0.8


OUT = REPO / "data" / "test_videos"


class ViaInterface:
    def __init__(self, kernel: Kernel):
        self.k = kernel
        self.sync_target()

    def sync_target(self):
        p = self.k.proprio()
        self.target_pos = p.eef_pos.copy()
        self.target_euler = p.eef_euler.copy()
        self.target_gripper = self.k._target_gripper[self.k.env.arms[0]]

    def _approach_axis(self):
        return R.from_euler("xyz", self.target_euler).apply([0.0, 0.0, 1.0])

    def gripper_teleport_via_click(self, cam: str, u: int, v: int, retreat: float = 0.10):
        point = self.k.raycast(cam, u, v)
        self.target_pos = point - self._approach_axis() * retreat
        return point

    def gripper_translate(self, dx: float, dy: float, dz: float):
        self.target_pos = self.target_pos + [dx, dy, dz]

    def gripper_rotate(self, axis: str, degrees: float):
        rot = R.from_euler(axis, degrees, degrees=True)
        self.target_euler = (rot * R.from_euler("xyz", self.target_euler)).as_euler("xyz")

    def gripper_advance_or_retreat(self, dist: float):
        self.target_pos = self.target_pos + self._approach_axis() * dist

    def gripper_toggle(self):
        self.target_gripper = 1 - self.target_gripper

    def execute_waypoint(self, stream_fn=None):
        self.k.set_ee_target(self.target_pos, self.target_euler)
        self.k.set_gripper(self.target_gripper)
        p = self.k.execute(stream_fn=stream_fn)
        env = self.k.env
        return {
            "pos": p.eef_pos, "euler": p.eef_euler, "gripper_open": p.gripper_open,
            "pos_err": float(np.linalg.norm(p.eef_pos - self.target_pos)),
            "num_step": env.num_step, "phase": env.phase,
        }


def project(env, cam, point):
    K = env.get_camera_intrinsics(cam)
    T = env.get_camera_extrinsics(cam)
    p = np.linalg.inv(T) @ [*point, 1.0]
    return int(round(K[0, 0] * p[0] / p[2] + K[0, 2])), int(round(K[1, 1] * p[1] / p[2] + K[1, 2]))


class Episode:
    def __init__(self, task, seed):
        self.env = SimEnv(task, seed=seed)
        self.k = Kernel(self.env, mode="dev")
        self.k.reset()
        self.env.phase = Phase.RUNNING
        self.ui = ViaInterface(self.k)
        self.frames = []
        self.stream = lambda obs: self.frames.append(obs["agentview_image"])

    def do(self, label):
        if self.env.episode_over:
            return None
        r = self.ui.execute_waypoint(stream_fn=self.stream)
        objs = " ".join(f"{n.split('_')[-1]}=({p[0]:+.3f},{p[1]:+.3f},{p[2]:.3f})"
                        for n, p in self.k.object_poses().items())
        print(f"  {label:<28} eef {np.round(r['pos'], 3)}  err {r['pos_err'] * 1000:.1f}mm  "
              f"step {r['num_step']}  {objs}")
        return r

    def wait_success(self, max_steps=100):
        for _ in range(max_steps):
            if self.env.episode_over or self.env._check(self.env):
                break
            obs = self.env.observe()
            self.env.apply_action(np.zeros(3), np.zeros(3),
                                  self.env.curr_gripper_open[self.env.arms[0]])
            self.stream(obs)

    def save_video(self, name):
        OUT.mkdir(parents=True, exist_ok=True)
        h, w = self.frames[0].shape[:2]
        vw = cv2.VideoWriter(str(OUT / f"{name}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 20, (w, h))
        for f in self.frames:
            vw.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        vw.release()
        cv2.imwrite(str(OUT / f"{name}_final.png"),
                    cv2.cvtColor(self.env.observe_camera()["agentview_image"], cv2.COLOR_RGB2BGR))
        print(f"  video: {OUT / f'{name}.mp4'} ({len(self.frames)} frames)")
