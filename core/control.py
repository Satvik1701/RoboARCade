"""Proprio, Mink IK, and the waypoint controller.

SimEnv owns the world; this module owns how the arm moves through it."""
import threading

import mink
import mujoco
import numpy as np
from ruckig import InputParameter, OutputParameter, Result, Ruckig, RuckigError
from scipy.spatial.transform import Rotation as R

from core.episode_recorder import ActMode
from core.grade_episode import hold_steps


class Proprio:
    def __init__(self, eef_pos, eef_quat, gripper_open: float):
        self.eef_pos = np.asarray(eef_pos)
        self.eef_quat = np.asarray(eef_quat)
        self.gripper_open = gripper_open
        self.gripper_open_np = np.array([gripper_open])
        self.eef_euler = R.from_quat(self.eef_quat).as_euler("xyz")


class ArmIKMink:
    GAIN = 0.3
    POSTURE_COST = 1e-5
    DAMPING = 1e-6
    DQ_MAX = 0.05
    LEASH = 0.05

    def __init__(self, model, joint_names, actuator_names, site_name, q_home):
        self.model = model
        self.dof = np.array([model.joint(n).dofadr[0] for n in joint_names])
        self.qadr = np.array([model.joint(n).qposadr[0] for n in joint_names])
        self.act = np.array([model.actuator(n).id for n in actuator_names])
        self.q_home = np.asarray(q_home, dtype=float)
        jr = np.array([model.joint(n).range for n in joint_names])
        self.q_lo, self.q_hi = jr[:, 0], jr[:, 1]
        self.site_id = model.site(site_name).id
        self.q_des = self.q_home.copy()
        self.last_pos_err = 0.0
        self.DT = float(model.opt.timestep)
        self.cfg = mink.Configuration(model)
        self.frame = mink.FrameTask(site_name, "site", position_cost=10.0,
                                    orientation_cost=5.0, gain=self.GAIN,
                                    lm_damping=1e-6)
        cost = np.zeros(model.nv)
        cost[self.dof] = self.POSTURE_COST
        self.posture = mink.PostureTask(model, cost=cost)
        q_target = model.qpos0.copy()
        q_target[self.qadr] = self.q_home
        self.posture.set_target(q_target)
        self.limits = [mink.ConfigurationLimit(model)]

    def reset(self, data):
        self.q_des = self.q_home.copy()
        data.ctrl[self.act] = self.q_des

    def substep(self, data, goal_pos, goal_rot: R):
        self.cfg.update(data.qpos)
        quat = goal_rot.as_quat()
        self.frame.set_target(mink.SE3.from_rotation_and_translation(
            mink.SO3(np.array([quat[3], quat[0], quat[1], quat[2]])), goal_pos))
        v = mink.solve_ik(self.cfg, [self.frame, self.posture], dt=self.DT,
                          solver="daqp", damping=self.DAMPING, limits=self.limits)
        dq = v[self.dof] * self.DT
        n = np.linalg.norm(dq)
        if n > self.DQ_MAX:
            dq *= self.DQ_MAX / n
        q_act = data.qpos[self.qadr]
        self.q_des = np.clip(self.q_des + dq, q_act - self.LEASH, q_act + self.LEASH)
        self.q_des = np.clip(self.q_des, self.q_lo, self.q_hi)
        data.ctrl[self.act] = self.q_des

    def solve_pose(self, q_init, goal_pos, goal_rot: R, iters=200):
        cfg = self.cfg
        q = self.model.qpos0.copy()
        q[self.qadr] = q_init
        cfg.update(q)
        quat = goal_rot.as_quat()
        self.frame.set_target(mink.SE3.from_rotation_and_translation(
            mink.SO3(np.array([quat[3], quat[0], quat[1], quat[2]])), goal_pos))
        home_target = self.posture.target_q.copy()
        q_reg = self.model.qpos0.copy(); q_reg[self.qadr] = q_init
        self.posture.set_target(q_reg)
        try:
            for _ in range(iters):
                v = mink.solve_ik(cfg, [self.frame, self.posture], self.DT, solver="daqp",
                                  damping=1e-3, limits=self.limits)
                step_rad = np.linalg.norm(v[self.dof]) * self.DT
                if step_rad > self.DQ_MAX:
                    v = v * (self.DQ_MAX / step_rad)
                cfg.integrate_inplace(v, self.DT)
                if step_rad < 1e-4:
                    break
        finally:
            self.posture.set_target(home_target)
        self.last_pos_err = float(np.linalg.norm(cfg.data.site_xpos[self.site_id] - goal_pos))
        return cfg.q[self.qadr].copy()


def make_arm_ik(model, arm_prefix, q_home, kin):
    names = dict(
        joint_names=[arm_prefix + n for n in kin["joints"]],
        actuator_names=[arm_prefix + n for n in kin["actuators"]],
        site_name=arm_prefix + kin["eef_site"],
        q_home=q_home,
    )
    return ArmIKMink(model, **names)


def rot_error_deg(a: R, b: R) -> float:
    return float(np.degrees(np.linalg.norm((a * b.inv()).as_rotvec())))


class WaypointController:
    def __init__(self, env):
        self.env = env
        self._payload_jacp = np.zeros((3, env.model.nv))
        self._payload_jacr = np.zeros((3, env.model.nv))
        names = [env.model.body(b).name for b in range(env.model.nbody)]
        self._arm_body = {arm: np.array([n.startswith(arm + "_") for n in names])
                          for arm in env.arms}
        self._probes = {}
        self._probe_lock = threading.Lock()

    def _record_step(self, recorder, obs):
        if recorder is not None:
            recorder.record(ActMode.Interpolate, obs)

    def _payload(self, arm):
        geoms = self._held_payload_geoms(arm)
        if not geoms:
            return None
        m = self.env.model
        bodies = np.unique(m.geom_bodyid[np.fromiter(geoms, dtype=int)])
        masses = m.body_mass[bodies]
        loaded = masses > 0
        if not np.any(loaded):
            return None
        return bodies[loaded], masses[loaded]

    def _feedforward(self, arm, ik, kp, payload):
        m, d = self.env.model, self.env.data
        torque = d.qfrc_bias[ik.dof] - d.qfrc_gravcomp[ik.dof]
        if payload is None:
            return torque / kp
        bodies, masses = payload
        mass = float(masses.sum())
        com = np.average(d.xipos[bodies], axis=0, weights=masses)
        support = -mass * m.opt.gravity
        site = self.env._site[arm]
        moment = np.cross(com - d.site_xpos[site], support)
        mujoco.mj_jacSite(m, d, self._payload_jacp, self._payload_jacr, site)
        generalized = self._payload_jacp.T @ support + self._payload_jacr.T @ moment
        return (torque + generalized[ik.dof]) / kp

    def _hold_pose(self, goal_pos, goal_rot, gripper_open: float, arm):
        self.env.data.ctrl[self.env._grip_act[arm]] = self.env._grip_open_ctrl[arm] * gripper_open
        ik = self.env.ik[arm]
        ik.q_des = ik.solve_pose(ik.q_des, goal_pos, goal_rot)
        kp = self.env.model.actuator_gainprm[ik.act, 0]
        payload = self._payload(arm)
        for _ in range(self.env._substeps):
            self.env.data.ctrl[ik.act] = ik.q_des + self._feedforward(arm, ik, kp, payload)
            self.env._step_physics()
        self.env._advance_episode_state()

    def apply_action(self, ee_pos, ee_euler, gripper_open: float, arm=None):
        arm = arm or self.env.arms[0]
        site = self.env._site[arm]
        d_pos = np.asarray(ee_pos) * self.env.output_max[:3]
        d_rot = np.asarray(ee_euler) * self.env.output_max[3:]
        site_rot = R.from_matrix(self.env.data.site_xmat[site].reshape(3, 3))
        self._hold_pose(self.env.data.site_xpos[site] + d_pos,
                        R.from_euler("xyz", d_rot) * site_rot, gripper_open, arm)

    def move_to(self, target_pos, target_euler, gripper_open, recorder=None, stream_fn=None,
                arm=None):
        arm = arm or self.env.arms[0]
        self.update_pose(target_pos, target_euler, recorder, stream_fn, arm)
        if self.env.episode_over:
            return
        assert gripper_open in (0, 1)
        if self.env.curr_gripper_open[arm] != gripper_open:
            self.update_gripper(gripper_open, recorder, stream_fn, arm)
            self.env.curr_gripper_open[arm] = gripper_open

    _JOINT_LIMS = dict(v=2.0, a=6.0)
    _LIN_LIMS = dict(v=1.0, a=5.0, j=50.0)
    _ROT_LIMS = dict(v=0.7, a=2.0, j=20.0)
    MIN_WAYPOINT_CYCLES = 3

    def _pose_steps(self, target_pos, target_euler, arm):
        ik = self.env.ik[arm]
        p0 = self.env.data.site_xpos[self.env._site[arm]].copy()
        r0 = R.from_matrix(self.env.data.site_xmat[self.env._site[arm]].reshape(3, 3))
        d = np.asarray(target_pos, float) - p0
        turn = (R.from_euler("xyz", target_euler) * r0.inv()).as_rotvec()
        dist, ang = float(np.linalg.norm(d)), float(np.linalg.norm(turn))
        if dist < 1e-6 and ang < 1e-6:
            return
        lim = {k: min(self._LIN_LIMS[k] / dist if dist > 1e-9 else np.inf,
                      self._ROT_LIMS[k] / ang if ang > 1e-9 else np.inf)
               for k in ("v", "a", "j")}
        cycle_s = 1.0 / self.env.control_freq
        vmax = 1.0 / (self.MIN_WAYPOINT_CYCLES * cycle_s)
        lim = {"v": min(lim["v"], vmax), "a": min(lim["a"], vmax / cycle_s),
               "j": min(lim["j"], vmax / cycle_s ** 2)}
        otg, inp, out = Ruckig(1, 1.0 / self.env.control_freq), InputParameter(1), OutputParameter(1)
        inp.current_position = [0.0]; inp.current_velocity = [0.0]; inp.current_acceleration = [0.0]
        inp.target_position = [1.0]; inp.target_velocity = [0.0]; inp.target_acceleration = [0.0]
        inp.max_velocity = [lim["v"]]; inp.max_acceleration = [lim["a"]]; inp.max_jerk = [lim["j"]]
        q_ref = self.env.data.qpos[ik.qadr].copy()
        kp = self.env.model.actuator_gainprm[ik.act, 0]
        dq_max_rad = self._JOINT_LIMS["v"] / self.env.control_freq
        ddq_max_rad = self._JOINT_LIMS["a"] / self.env.control_freq ** 2
        step, s, res, waiting = np.zeros(len(ik.qadr)), 0.0, None, False
        for _ in range(self.env.waypoint_max_step):
            if not waiting:
                try:
                    res = otg.update(inp, out)
                except RuckigError:
                    res, s = Result.Finished, 1.0
                else:
                    inp.current_position = out.new_position
                    inp.current_velocity = out.new_velocity
                    inp.current_acceleration = out.new_acceleration
                    s = float(np.clip(out.new_position[0], 0.0, 1.0))
            ref_pos = p0 + d * s
            q_want = ik.solve_pose(q_ref, ref_pos, R.from_rotvec(turn * s) * r0)
            if not np.all(np.isfinite(q_want)):
                return
            dq = q_want - q_ref
            peak = float(np.abs(dq).max())
            cap = min(dq_max_rad, float(np.abs(step).max()) + ddq_max_rad)
            waiting = peak > cap
            step = dq * (cap / peak) if waiting else dq
            q_ref = q_ref + step
            ik.q_des = q_ref
            self.env.data.ctrl[self.env._grip_act[arm]] = \
                self.env._grip_open_ctrl[arm] * self.env.curr_gripper_open[arm]
            payload = self._payload(arm)
            for _ in range(self.env._substeps):
                self.env.data.ctrl[ik.act] = q_ref + self._feedforward(arm, ik, kp, payload)
                self.env._step_physics()
            if not np.all(np.isfinite(self.env.data.qpos)):
                return
            if ik.last_pos_err > self.TRACK_TOL:
                return
            yield bool(res == Result.Finished) and not waiting

    def _drive_to(self, target_pos, target_euler, recorder, stream_fn, arm) -> bool:
        tgt, tgt_rot = np.asarray(target_pos), R.from_euler("xyz", target_euler)
        settle_steps = 0
        for finished in self._pose_steps(target_pos, target_euler, arm):
            obs = self.env.observe(depth=False) if (stream_fn or recorder) else None
            self._record_step(recorder, obs)
            if stream_fn:
                stream_fn(obs)
            self.env._advance_episode_state()
            if self.env.episode_over:
                return True
            if finished:
                pos_err_m = np.linalg.norm(self.env.observe_proprio(arm).eef_pos - tgt)
                now = R.from_matrix(self.env.data.site_xmat[self.env._site[arm]].reshape(3, 3))
                rot_err_rad = (tgt_rot * now.inv()).magnitude()
                if settle_steps >= 25 or (pos_err_m < 1e-4 and rot_err_rad < self.ARRIVE_ROT_TOL):
                    return pos_err_m < self.ARRIVE_TOL and rot_err_rad < self.ARRIVE_ROT_TOL
            settle_steps += finished
        return False

    def home_pose(self, arm=None):
        arm = arm or self.env.arms[0]
        ik = self.env.ik[arm]
        d = mujoco.MjData(self.env.model)
        d.qpos[:] = self.env.data.qpos
        d.qpos[ik.qadr] = ik.q_home
        mujoco.mj_kinematics(self.env.model, d)
        site = self.env._site[arm]
        pos = d.site_xpos[site].copy()
        euler = R.from_matrix(d.site_xmat[site].reshape(3, 3)).as_euler("xyz")
        return pos, euler

    HOME_TOL = 0.01

    def _reconfigure_home(self, arm, recorder=None, stream_fn=None):
        ik = self.env.ik[arm]
        q0 = self.env.data.qpos[ik.qadr].copy()
        ramp_steps = 100
        for i in range(2 * ramp_steps):
            a = min(1.0, (i + 1) / ramp_steps)
            ik.q_des = (1 - a) * q0 + a * ik.q_home
            self.env.data.ctrl[ik.act] = ik.q_des
            for _ in range(self.env._substeps):
                self.env._step_physics()
            obs = self.env.observe(depth=False) if (stream_fn or recorder) else None
            self.env._advance_episode_state()
            self._record_step(recorder, obs)
            if stream_fn:
                stream_fn(obs)
            if self.env.episode_over:
                return
            if a >= 1.0 and np.max(np.abs(self.env.data.qpos[ik.qadr] - ik.q_home)) < self.HOME_TOL:
                return

    def home(self, arm=None, gripper_open=None, recorder=None, stream_fn=None):
        arm = arm or self.env.arms[0]
        target_pos, target_euler = self.home_pose(arm)
        if self.reaches(arm, target_pos, target_euler)[0]:
            if gripper_open is None:
                gripper_open = self.env.curr_gripper_open[arm]
            self.move_to(target_pos, target_euler, gripper_open,
                         recorder=recorder, stream_fn=stream_fn, arm=arm)
        else:
            self._reconfigure_home(arm, recorder, stream_fn)

    def reaches(self, arm, target_pos, target_euler, q0=None):
        q0 = self.env.data.qpos[self.env.ik[arm].qadr] if q0 is None else np.asarray(q0, float)
        target_pos = np.asarray(target_pos, float)
        target_rot = R.from_euler("xyz", target_euler)
        with self._probe_lock:
            if arm not in self._probes:
                robot = self.env.spec["robots"][self.env.arms.index(arm)]
                self._probes[arm] = (
                    make_arm_ik(self.env.model, arm + "_", robot["home_qpos"], self.env.kin[arm]),
                    mujoco.MjData(self.env.model))
            ik, d = self._probes[arm]
            q_t = ik.solve_pose(q0, target_pos, target_rot)
            if not np.all(np.isfinite(q_t)):
                return False, float("inf"), float("inf")
            d.qpos[:] = self.env.data.qpos
            d.qpos[ik.qadr] = q0
            mujoco.mj_kinematics(self.env.model, d)
            start_pos = d.site_xpos[ik.site_id].copy()
            start_rot = R.from_matrix(d.site_xmat[ik.site_id].reshape(3, 3))
            d.qpos[ik.qadr] = q_t
            mujoco.mj_kinematics(self.env.model, d)
            pos_err = float(np.linalg.norm(d.site_xpos[ik.site_id] - target_pos))
            actual_rot = R.from_matrix(d.site_xmat[ik.site_id].reshape(3, 3))
            rot_err = float((target_rot * actual_rot.inv()).magnitude())
            ok = pos_err < self.ARRIVE_TOL and rot_err < self.ARRIVE_ROT_TOL
            if ok:
                q_path = np.asarray(q0, float).copy()
                delta = target_pos - start_pos
                turn = (target_rot * start_rot.inv()).as_rotvec()
                for s in np.linspace(0.0, 1.0, self.REACH_PATH_SAMPLES + 1)[1:]:
                    ref_pos = start_pos + delta * s
                    ref_rot = R.from_rotvec(turn * s) * start_rot
                    q_path = ik.solve_pose(q_path, ref_pos, ref_rot)
                    if not np.all(np.isfinite(q_path)):
                        return False, float("inf"), float("inf")
                    d.qpos[ik.qadr] = q_path
                    mujoco.mj_kinematics(self.env.model, d)
                    path_pos = float(np.linalg.norm(d.site_xpos[ik.site_id] - ref_pos))
                    path_rot = R.from_matrix(d.site_xmat[ik.site_id].reshape(3, 3))
                    path_rerr = float((ref_rot * path_rot.inv()).magnitude())
                    if path_pos > self.TRACK_TOL:
                        return False, path_pos, path_rerr
            return bool(ok), pos_err, rot_err

    ARRIVE_TOL = 0.002
    ARRIVE_ROT_TOL = 0.03
    REFINE_TOL_M = 0.0005
    REFINE_ROT_TOL_RAD = 0.010
    RESOLVE_TRIES = 3
    TRACK_TOL = 0.01
    REACH_PATH_SAMPLES = 24

    SETTLE_VERDICT_STEPS = 40

    def settle_for_verdict(self, recorder=None, stream_fn=None):
        env, hold, rested = self.env, hold_steps(self.env), 0
        for _ in range(self.SETTLE_VERDICT_STEPS + hold):
            rested = rested + 1 if rested or env._objects_at_rest() else 0
            if rested > hold:
                return
            for _ in range(env._substeps):
                env._step_physics()
            obs = env.observe(depth=False) if (stream_fn or recorder) else None
            env._advance_episode_state()
            self._record_step(recorder, obs)
            if stream_fn:
                stream_fn(obs)

    def update_pose(self, target_pos, target_euler, recorder=None, stream_fn=None, arm=None):
        arm = arm or self.env.arms[0]
        now = self.env.observe_proprio(arm).eef_pos
        rot = R.from_matrix(self.env.data.site_xmat[self.env._site[arm]].reshape(3, 3))
        if (np.linalg.norm(now - np.asarray(target_pos, float)) < self.REFINE_TOL_M
                and (R.from_euler("xyz", target_euler) * rot.inv()).magnitude()
                < self.REFINE_ROT_TOL_RAD):
            return
        contact_baseline = self._payload_contact_baseline(arm)
        if not self.reaches(arm, target_pos, target_euler)[0]:
            self._drive_to(target_pos, target_euler, recorder, stream_fn, arm)
            return
        prev_pos_err_m = None
        for attempt in range(self.RESOLVE_TRIES + 1):
            if self._drive_to(target_pos, target_euler, recorder, stream_fn, arm):
                return
            if self.env.episode_over:
                return
            if self.new_contact(arm, contact_baseline):
                return
            pos_err_m = float(np.linalg.norm(
                self.env.observe_proprio(arm).eef_pos - np.asarray(target_pos)))
            if prev_pos_err_m is not None and prev_pos_err_m - pos_err_m < 1e-3:
                break
            prev_pos_err_m = pos_err_m

    def _arm_contacts(self, arm) -> set:
        body_of, mine = self.env.model.geom_bodyid, self._arm_body[arm]
        out = set()
        for c in self.env.data.contact[: self.env.data.ncon]:
            a, b = mine[body_of[c.geom1]], mine[body_of[c.geom2]]
            if a != b:
                out.add(int(c.geom2 if a else c.geom1))
        return out

    def _contact_bodies(self, geoms) -> set:
        body_of = self.env.model.geom_bodyid
        return {int(body_of[g]) for g in geoms}

    def new_contact(self, arm, baseline) -> bool:
        return bool(self._contact_bodies(self._arm_contacts(arm)) - set(baseline))

    def _payload_contact_baseline(self, arm) -> set:
        return self._contact_bodies(self._held_payload_geoms(arm))

    def _finger_bodies(self, arm) -> set:
        return {b for b in range(self.env.model.nbody)
                if self.env.finger_root(b, arm) is not None}

    def _held_payload_geoms(self, arm) -> set:
        if self.env.curr_gripper_open[arm]:
            return set()
        m, env = self.env.model, self.env
        contacts = self._arm_contacts(arm)
        out = set()
        for geoms in env._obj_geoms.values():
            payload = {int(g) for g in geoms}
            if payload & contacts:
                out.update(payload)
        finger_roots, pads_by_root = env.finger_roots(arm), {}
        for contact in env.data.contact[: env.data.ncon]:
            for finger, other in ((contact.geom1, contact.geom2), (contact.geom2, contact.geom1)):
                pad = env.finger_root(int(m.geom_bodyid[finger]), arm)
                if pad is None:
                    continue
                root = int(m.geom_bodyid[other])
                while root:
                    if m.body_dofnum[root] and not m.body(root).name.startswith(env.robot_prefix):
                        pads_by_root.setdefault(root, set()).add(pad)
                    root = int(m.body_parentid[root])
        for root, pads in pads_by_root.items():
            if pads == finger_roots:
                out.update(int(g) for g in env._subtree_geoms(m.body(root).name))
        return out

    CENTRING_DAMPING_NS_PER_M = 2500.0
    CENTRING_ENGAGE_FORCE_N = 10.0
    CENTRING_ENGAGE_TIME_S = 0.2
    CENTRING_MAX_VEL_M_S = 0.040
    CENTRING_MAX_DISP_M = 0.015
    CENTRING_SETTLE_VEL_M_S = 5e-4
    MIN_JAW_STABLE_STEPS = 5

    def _jaw_axis(self, arm):
        m, d = self.env.model, self.env.data
        jid = m.joint(arm + "_" + self.env.kin[arm]["gripper"]["joints"][0]).id
        return d.xmat[m.jnt_bodyid[jid]].reshape(3, 3) @ m.jnt_axis[jid]

    def _grasp_alignment_force(self, arm, axis, fingers):
        tot, f = 0.0, np.zeros(6)
        for i in range(self.env.data.ncon):
            c = self.env.data.contact[i]
            b1, b2 = self.env.model.geom_bodyid[c.geom1], self.env.model.geom_bodyid[c.geom2]
            if b1 not in fingers and b2 not in fingers:
                continue
            mujoco.mj_contactForce(self.env.model, self.env.data, i, f)
            force_world = c.frame.reshape(3, 3).T @ f[:3]
            tot += float(force_world @ axis) * (1.0 if b2 in fingers else -1.0)
        return tot

    def update_gripper(self, gripper_open, recorder=None, stream_fn=None, arm=None):
        arm = arm or self.env.arms[0]
        g = float(gripper_open)
        prev = self.env.observe_proprio(arm).gripper_open
        stable_steps = 0
        axis = self._jaw_axis(arm) if gripper_open == 0 else None
        fingers = self._finger_bodies(arm) if axis is not None else None
        site = self.env._site[arm]
        close_pos = self.env.data.site_xpos[site].copy()
        close_rot = R.from_matrix(self.env.data.site_xmat[site].reshape(3, 3))
        shift_m, dt = 0.0, 1.0 / self.env.control_freq
        engaged, persist_steps = False, 0
        for _ in range(self.env.waypoint_max_step):
            shift_step_m = 0.0
            if axis is not None:
                force_n = self._grasp_alignment_force(arm, axis, fingers)
                if not engaged:
                    persist_steps = persist_steps + 1 if abs(force_n) > self.CENTRING_ENGAGE_FORCE_N else 0
                    engaged = persist_steps >= round(self.CENTRING_ENGAGE_TIME_S / dt)
                if engaged:
                    qdot = float(np.clip(force_n / self.CENTRING_DAMPING_NS_PER_M,
                                         -self.CENTRING_MAX_VEL_M_S, self.CENTRING_MAX_VEL_M_S))
                    shift_before_m = shift_m
                    shift_m = float(np.clip(shift_m + qdot * dt,
                                            -self.CENTRING_MAX_DISP_M, self.CENTRING_MAX_DISP_M))
                    shift_step_m = shift_m - shift_before_m
            self._hold_pose(close_pos if axis is None else close_pos + shift_m * axis,
                            close_rot, g, arm)
            obs = self.env.observe(depth=False) if (stream_fn or recorder) else None
            self._record_step(recorder, obs)
            if stream_fn:
                stream_fn(obs)
            curr = self.env.observe_proprio(arm).gripper_open
            stable_steps = stable_steps + 1 if (abs(curr - prev) < 0.002
                                                and abs(shift_step_m) < self.CENTRING_SETTLE_VEL_M_S * dt) else 0
            prev = curr
            if stable_steps >= self.MIN_JAW_STABLE_STEPS or self.env.episode_over:
                return
