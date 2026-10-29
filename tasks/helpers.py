"""Helpers shared by tasks/*/test.py: shared scenes and predicate probes, controller probes, and scripted
solves through the default interface."""


from pathlib import Path

from scipy.spatial.transform import Rotation as R
import cv2
import mujoco
import numpy as np
import pytest

from core.control import make_arm_ik
from core.grade_episode import make_checker
from core.sim_env import Phase, SimEnv
from core.tasks import TASKS, load_scene
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


def spawn_mesh_piece(env, piece, x, y, yaw):
    adr = env.model.joint(env.model.body(piece).jntadr[0]).qposadr[0]
    model = env.model
    body = model.body(piece).id
    bottom = np.inf
    for g in range(model.ngeom):
        if model.geom_bodyid[g] != body or not model.geom_contype[g]:
            continue
        mesh = model.geom_dataid[g]
        start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
        vertices = model.mesh_vert[start:start + count]
        rotation = R.from_quat(model.geom_quat[g][[1, 2, 3, 0]])
        bottom = min(bottom, (rotation.apply(vertices) + model.geom_pos[g])[:, 2].min())
    # Collider bounds keep long insertion stems from starting inside the table.
    env.data.qpos[adr:adr + 3] = [x, y, _table_top_z(env) - bottom + .0005]
    env.data.qpos[adr + 3:adr + 7] = R.from_euler("z", yaw).as_quat()[[3, 0, 1, 2]]
    env.data.qvel[:] = 0
    mujoco.mj_forward(env.model, env.data)
    mujoco.mj_step(env.model, env.data,
                  nstep=round(.3 / env.model.opt.timestep))


def grasp_mesh_piece(env, arm, piece):
    home = env.observe_proprio(arm)
    p = env.data.body(piece).xpos.copy()
    yaw = R.from_quat(env.data.body(piece).xquat[[1, 2, 3, 0]]).as_euler("xyz")[2]
    a, b = (env.data.body(f"{arm}_{side}_finger").xpos for side in ("left", "right"))
    jaw_yaw = float(np.arctan2(b[1] - a[1], b[0] - a[0]))
    delta = (yaw + np.pi / 2) - jaw_yaw
    grasp_rot = R.from_euler("z", delta) * R.from_euler("xyz", home.eef_euler)
    e = grasp_rot.as_euler("xyz")
    above, at = p + [0, 0, 0.12], p.copy()
    env.move_to(above, e, 1.0)
    env.move_to(at, e, 1.0)
    env.move_to(at, e, 0.0)
    env.move_to(p + [0, 0, 0.10], e, 0.0)
    return e


def check_mating_surface(task, cfg):
    model = load_scene(TASKS / task).compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    for name in (cfg["target"], cfg["receiver"]):
        body = model.body(name).id
        model.geom_group[:] = 5
        for gid in range(model.ngeom):
            if model.geom_bodyid[gid] == body:
                model.geom_group[gid] = (
                    3 if model.geom_contype[gid] or model.geom_conaffinity[gid] else 1
                )
                model.geom_rgba[gid, 3] = 1
        target = name == cfg["target"]
        xy = np.zeros(2) if target else np.array(cfg["xy"])
        z0, z1 = (
            (cfg["peg_bottom"], cfg["peg_top"])
            if target
            else (cfg["hole_bottom"], cfg["hole_top"])
        )
        for z in np.linspace(z0 + 0.001, z1 - 0.001, 7):
            # Avoid casting exactly along polygon seams, where a ray/triangle
            # intersection can be numerically ambiguous.
            for angle in np.linspace(0, 2 * np.pi, 128, endpoint=False) + 0.012345:
                ray = np.array([np.cos(angle), np.sin(angle), 0])
                origin = np.array([*xy, z])
                if target:
                    origin += 0.02 * ray
                    ray = -ray
                rotation = data.xmat[body].reshape(3, 3)
                hits = []
                for group in (1, 3):
                    mask = np.zeros(6, dtype=np.uint8)
                    mask[group] = 1
                    hits.append(
                        mujoco.mj_ray(
                            model,
                            data,
                            data.xpos[body] + rotation @ origin,
                            rotation @ ray,
                            mask,
                            True,
                            -1,
                            np.array([-1], dtype=np.int32),
                        )
                    )
                assert min(hits) >= 0, "Missing visual or collision surface"
                # 30 um allows polygon approximation and source mesh rounding.
                assert abs(hits[1] - hits[0]) < cfg["erosion"] + 0.00003


def check_mating_insertion(task, cfg, seed):
    env = SimEnv(task, seed=seed)
    env.reset(seed=seed)
    model, data = env.model, env.data
    target, base = model.body(cfg["target"]).id, model.body(cfg["receiver"]).id
    qadr = int(model.jnt_qposadr[model.body_jntadr[target]])
    vadr = int(model.jnt_dofadr[model.body_jntadr[target]])
    goal = np.array(env.spec["success"]["of"][0]["pos"])

    def place(relative):
        data.qpos[qadr : qadr + 3] = (
            data.xpos[base] + data.xmat[base].reshape(3, 3) @ relative
        )
        data.qpos[qadr + 3 : qadr + 7] = data.xquat[base]
        data.qvel[vadr : vadr + 6] = 0
        mujoco.mj_forward(model, data)

    def depths():
        return [
            c.dist
            for c in data.contact
            if {int(model.geom_bodyid[c.geom1]), int(model.geom_bodyid[c.geom2])}
            == {target, base}
        ]

    for dz in np.linspace(0.03, 0, 61):
        place(goal + [0, 0, dz])
        assert (
            min(depths(), default=0) > -0.00005
        ), "A collision hull blocks the aligned insertion"
    place(goal + [0.006, 0, 0])
    assert min(depths(), default=0) < -0.001, "Socket must block a misaligned insertion"
    place(goal + [0, 0, 0.003])
    mujoco.mj_step(model, data, nstep=round(2 / model.opt.timestep))
    assert np.all(np.isfinite(data.qpos))
    assert make_checker(env.spec["success"])[0](env)
    relative = data.xmat[base].reshape(3, 3).T @ (data.xpos[target] - data.xpos[base])
    assert np.linalg.norm(relative - goal) < 0.0015


def check_mating_pickup(task, piece):
    from core.grade_episode import _grasped_by

    env = SimEnv(task, seed=0)
    env.reset(seed=0, randomize_init=False)
    env.max_len = 10**9
    arm = env.arms[0]
    spawn_mesh_piece(env, piece, 0.06, -0.12, 0.0)
    height = env.data.body(piece).xpos[2]
    grasp_mesh_piece(env, arm, piece)
    assert _grasped_by(env, piece, arm)
    assert env.data.body(piece).xpos[2] > height + 0.05
