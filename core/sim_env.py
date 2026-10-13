"""One task's simulation: reset, actions and observations."""
import atexit
import copy
import json
import logging
import os
import time
import weakref
from pathlib import Path
from enum import Enum

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

from core import grade_episode, tasks
from core.control import Proprio, WaypointController, make_arm_ik
from core.initial_layout import DEFAULT_INIT_REGION as _DEFAULT_INIT_REGION, Layout
from core.utils import camera_intrinsics, deproject

logger = logging.getLogger(__name__)

PHYSICS_DT = 0.00025


def _close_before_egl_exit(owner, resource):
    live = [resource]

    def close():
        if live:
            try:
                live.pop().close()
            except Exception:
                pass

    atexit.register(close)
    weakref.finalize(owner, close)



class Phase(Enum):
    SETUP = 0
    RUNNING = 1
    DONE = 2


class SimEnv(Layout):
    def __init__(self, task: str, seed: int | None = None, pack: Path | None = None):
        self.task = task
        self.pack = Path(pack) if pack else tasks.resolve_task(task)
        self.spec = json.loads((self.pack / "task.json").read_text())
        physics = self.spec.get("physics") or {}
        self._timestep = float(physics.get("timestep", PHYSICS_DT))
        self.kin = {rb["name"]: tasks.robot_kinematics(rb["model"])
                    for rb in self.spec["robots"]}
        self.language_instruction = self.spec["language"]
        self._rng = np.random.default_rng(seed)
        self._build_model()
        self._wire_arms()
        self._wire_spec()
        self._wire_objects()
        self._init_episode_state()
        if (self.pack / "dynamics.py").is_file():
            self.dynamics = tasks.task_module(self.pack.name, "dynamics").Dynamics(self)

    def _build_model(self):
        mspec = tasks.load_scene(self.pack)
        mspec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        mspec.option.timestep = self._timestep
        mspec.memory = max(mspec.memory, 64 << 20)
        for rb in self.spec["robots"]:
            arm = mujoco.MjSpec.from_file(
                str(tasks.ASSETS / "robots" / rb["model"] / self.kin[rb["name"]]["mjcf"]))
            frame = mspec.worldbody.add_frame(pos=rb["base_pos"], quat=rb["base_quat"])
            mspec.attach(arm, frame=frame, prefix=rb["name"] + "_")
            bx, by, bz = rb["base_pos"]
            if bz > 1e-3:
                pedestal = mspec.worldbody.add_geom()
                pedestal.name = rb["name"] + "_pedestal"
                pedestal.type = mujoco.mjtGeom.mjGEOM_CYLINDER
                pedestal.pos = [bx, by, bz / 2]
                pedestal.size = [0.12, bz / 2, 0.0]
                pedestal.contype, pedestal.conaffinity = 0, 0
                pedestal.group = 1
                pedestal.rgba = [0.30, 0.31, 0.34, 1.0]
        self._ring_names = tasks.add_cameras(mspec, self.spec, self.task)
        self.model = mspec.compile()
        self.data = mujoco.MjData(self.model)
        self.step_stats = [0, 0, 0.0, 0]

    def _wire_arms(self):
        self.arms = [rb["name"] for rb in self.spec["robots"]]
        self.robot_prefix = tuple(n + "_" for n in self.arms)
        self.ik, self._finger_qadr, self._finger_roots = {}, {}, {}
        self._grip_act, self._site = {}, {}
        self._width_coef, self._grip_open_ctrl, self._open_qpos = {}, {}, {}
        for rb in self.spec["robots"]:
            n = rb["name"]
            p = n + "_"
            kin = self.kin[n]
            grip = kin["gripper"]
            self.ik[n] = make_arm_ik(self.model, p, rb["home_qpos"], kin)
            finger_joints = [self.model.joint(p + j).id for j in grip["joints"]]
            self._finger_qadr[n] = [self.model.jnt_qposadr[j] for j in finger_joints]
            self._finger_roots[n] = frozenset(
                int(self.model.jnt_bodyid[j]) for j in finger_joints)
            if len(self._finger_roots[n]) != 2:
                raise ValueError(
                    f"{n}: gripper joints must declare 2 distinct finger roots, "
                    f"got {len(self._finger_roots[n])}")
            self._grip_act[n] = self.model.actuator(p + grip["actuator"]).id
            self._site[n] = self.model.site(p + kin["eef_site"]).id
            self._width_coef[n] = np.asarray(grip["width_coef"], float)
            self._grip_open_ctrl[n] = float(grip["ctrl_open"])
            self._open_qpos[n] = [float(v) for v in grip["open_qpos"]]

    def _wire_spec(self):
        ctl = self.spec["control"]
        self.control_freq = ctl["control_freq"]
        self._substeps = int(round(1.0 / (self.control_freq * self._timestep)))
        self.output_max = np.array(ctl["output_max"])
        self.image_size = ctl["image_size"]
        declared = [
            (tasks.camera_name(c), c)
            for c in self.spec["cameras"]
            if not (self._ring_names and c["role"] == "pc")
        ]
        generated = self._ring_names + ["birdseye", "zoomed_birdseye", "eyelevel", "farside"]
        self.cameras = [name for name, _ in declared] + generated
        self._camera_facts = {
            **{name: c for name, c in declared},
            **{name: {"name": name, "role": "pc"} for name in generated},
        }
        self._check = grade_episode.make_checker(self.spec["success"], self.task)[0]
        self.max_len = self.spec["termination"]["max_len"]

        rnd = self.spec["randomization"]
        self.full_rand = bool(rnd.get("full_rand", True))
        self.init_region = rnd.get("region") or _DEFAULT_INIT_REGION
        self._face_robot = set(rnd.get("face_robot") or [])
        self._min_sep_xy = [(a, b, float(d)) for a, b, d in (rnd.get("min_sep") or [])]
        self._min_base_distance = dict(rnd.get("min_base_distance") or {})
        self._referents = [dict(r) for r in (rnd.get("referents") or [])]
        self._grasp_clearance = float(rnd.get("grasp_clearance") or 0.0)
        self._articulated_reach = set(rnd.get("articulated_reach") or ())
        self._reach_sites = list(rnd.get("reach_sites") or ())
        self._lift_reach = {name: float(height)
                            for name, height in (rnd.get("lift_reach") or {}).items()}
        self._reach_site_bodies = {
            self.model.body(b).name for b in range(self.model.nbody) for site in self._reach_sites
            if self._ancestor_in(self.model.site_bodyid[self.model.site(site).id], {b})}
        self._placement_regions = dict(rnd.get("placement_regions") or {})
        self.ctl = WaypointController(self)
        self.dynamics = None
        self.episode_t0 = 0.0
        self._dealt = False
        self.pc = self.spec["pc_crop"]
        self._renderer = None
        self._blender = None
        self._subtree_cache, self._legal_cache, self._obj_color_cache, self._reach_grid = {}, {}, {}, {}
        self._base_xy = np.asarray(self.spec["robots"][0]["base_pos"][:2], float)
        self.waypoint_max_step = ctl["waypoint_max_step"]
        self.idle_step = ctl["idle_step"]

    def camera_role(self, cam: str) -> str:
        return self._camera_facts[cam]["role"]

    def camera_robot(self, cam: str) -> str | None:
        return self._camera_facts[cam].get("robot")

    def cameras_by_role(self, role: str) -> list[str]:
        return [cam for cam in self.cameras if self.camera_role(cam) == role]

    def _wire_objects(self):
        rnd = self.spec["randomization"]
        self._free_objects = list(self.spec.get("objects", []))
        self._obj_qadr = {n: self.model.joint(self.model.body(n).jntadr[0]).qposadr[0]
                          for n in self._free_objects}
        self._obj_dofadr = {n: self.model.joint(self.model.body(n).jntadr[0]).dofadr[0]
                            for n in self._free_objects}
        self._free_body_ids = {self.model.body(n).id for n in self._free_objects}
        self._static_objects = sorted(
            {n for g in (rnd.get("groups") or []) for n in g} - set(self._free_objects))
        self._static_base = {
            n: (self.model.body(n).pos.copy(), self.model.body(n).quat.copy())
            for n in self._static_objects}
        support_roots = {self.model.body(n).id for n in self.spec["support"]}
        free_ids = {n: self.model.body(n).id for n in self._free_objects}
        obj_geoms, obst_geoms = {}, {}
        for g in range(self.model.ngeom):
            if self.model.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE:
                continue
            b = int(self.model.geom_bodyid[g])
            o = next((n for n, r in free_ids.items() if self._ancestor_in(b, {r})), None)
            if o is not None:
                obj_geoms.setdefault(o, []).append(g)
            else:
                name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, b) or ""
                if (b != 0 and not self._ancestor_in(b, support_roots)
                        and not name.startswith(self.robot_prefix)):
                    obst_geoms.setdefault(b, []).append(g)
        self._obj_geoms = {n: np.array(v) for n, v in obj_geoms.items()}
        self._obstacle_geoms = {b: np.array(v) for b, v in obst_geoms.items()}
        base_geoms = []
        for b in range(self.model.nbody):
            bname = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, b) or ""
            if not bname.startswith(self.robot_prefix):
                continue
            anc, jointed = b, False
            while anc > 0:
                if self.model.body_jntnum[anc]:
                    jointed = True
                    break
                anc = int(self.model.body_parentid[anc])
            if not jointed:
                base_geoms.extend(range(int(self.model.body_geomadr[b]),
                                        int(self.model.body_geomadr[b]) + int(self.model.body_geomnum[b])))
        self._robot_base_geoms = np.array(base_geoms, int)
        bb = {}
        for g in base_geoms:
            bb.setdefault(int(self.model.geom_bodyid[g]), []).append(g)
        self._obstacle_geoms.update({b: np.array(v) for b, v in bb.items()})
        self._static_geoms = {n: self._subtree_geoms(n) for n in self._static_objects}
        self._support_rect = None
        self._support_top_z = None
        if support_roots:
            mujoco.mj_kinematics(self.model, self.data)
            gids = [g for g in range(self.model.ngeom)
                    if self._ancestor_in(int(self.model.geom_bodyid[g]), support_roots)]
            if gids:
                c = self.data.geom_xpos[gids]
                h = self.model.geom_size[gids][:, :2]
                self._support_rect = (float((c[:, 0] - h[:, 0]).min()),
                                      float((c[:, 0] + h[:, 0]).max()),
                                      float((c[:, 1] - h[:, 1]).min()),
                                      float((c[:, 1] + h[:, 1]).max()))
                aabb = self.model.geom_aabb[gids]
                self._support_top_z = float((c[:, 2] + aabb[:, 2] + aabb[:, 5]).max())

    def _init_episode_state(self):
        self.curr_gripper_open = {n: 1 for n in self.arms}
        self.gripper_max_width = float(
            np.dot(self._width_coef[self.arms[0]], self._open_qpos[self.arms[0]]))
        self.rest_z = {}
        self._rim_at_base = {}
        self._centre_at_base = {}
        self._path_at_base = {}
        self._heading_at_base = {}
        self._settled_bases = None
        self.phase = Phase.SETUP
        self.num_step = 0
        self._home_euler = {
            rb["name"]: np.asarray(rb.get("home_euler")
                                   or self.kin[rb["name"]]["home_euler"], float)
            for rb in self.spec["robots"]}

    SETTLE_EXTRA_S = 0.8

    def _unsupported(self) -> bool:
        free = self._free_body_ids
        held = set()
        for c in self.data.contact[: self.data.ncon]:
            b1, b2 = int(self.model.geom_bodyid[c.geom1]), int(self.model.geom_bodyid[c.geom2])
            r1, r2 = self.model.body_rootid[b1], self.model.body_rootid[b2]
            if r1 in free and r2 not in free:
                held.add(r1)
            elif r2 in free and r1 not in free:
                held.add(r2)
        return any(b not in held for b in free)

    def _objects_at_rest(self):
        if any(float(np.abs(self.data.qvel[a:a + 6]).max()) > 3e-3
               for a in self._obj_dofadr.values()):
            return False
        m, d, free = self.model, self.data, self._free_body_ids
        for c in d.contact[: d.ncon]:
            b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
            if b1 != b2 and b1 in free and b2 in free and c.dist < -1e-3:
                return False
        return True

    def settle_until_rest(self):
        extra = int(round(self.SETTLE_EXTRA_S / self.model.opt.timestep))
        for i in range(self.idle_step + extra):
            self._step_physics()
            if i + 1 >= self.idle_step and self._objects_at_rest():
                return

    def _subtree_geoms(self, name):
        cached = self._obj_geoms.get(name)
        if cached is not None and len(cached):
            return cached
        if name not in self._subtree_cache:
            root = {self.model.body(name).id}
            self._subtree_cache[name] = np.array(
                [g for g in range(self.model.ngeom)
                 if self.model.geom_type[g] != mujoco.mjtGeom.mjGEOM_PLANE
                 and self._ancestor_in(int(self.model.geom_bodyid[g]), root)], dtype=int)
        return self._subtree_cache[name]

    def _ancestor_in(self, body, roots) -> bool:
        while body > 0:
            if body in roots:
                return True
            body = self.model.body_parentid[body]
        return False

    def object_color(self, name):
        cache = self._obj_color_cache
        if name in cache:
            return cache[name]
        m = self.model
        geoms = self._subtree_geoms(name)
        vis = [g for g in geoms if m.geom_group[g] == 1] or \
              [g for g in geoms if m.geom_rgba[g][3] > 0.1]
        col, wsum = np.zeros(3), 0.0
        for g in vis:
            rgb = np.array(m.geom_rgba[g][:3], float)
            mat = int(m.geom_matid[g])
            if mat >= 0:
                t = int(np.atleast_1d(m.mat_texid[mat]).max())
                if t >= 0:
                    n = m.tex_height[t] * m.tex_width[t] * m.tex_nchannel[t]
                    px = m.tex_data[m.tex_adr[t]: m.tex_adr[t] + n]
                    rgb = px.reshape(-1, m.tex_nchannel[t])[:, :3].mean(axis=0) / 255.0
            w = float(m.geom_rbound[g]) ** 3
            col, wsum = col + w * rgb, wsum + w
        out = (col / wsum).round(3).tolist() if wsum > 0 else [0.13, 0.77, 0.37]
        cache[name] = out
        return out

    def reset(self, seed: int | None = None, randomize_init: bool = True):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._dealt = False
        self._clean_state()
        self._dynamics("restore")
        mujoco.mj_kinematics(self.model, self.data)
        if self._settled_bases is None and self.spec["randomization"].get("type") == "init_states":
            self._place_objects(False)
            self._lift_out_of_penetration()
            stored = {n: (self.data.qpos[a:a + 3].copy(), self.data.qpos[a + 3:a + 7].copy())
                      for n, a in self._obj_qadr.items()}
            if self._unsupported():
                self.settle_until_rest()
            got = {n: (self.data.qpos[a:a + 3].copy(), self.data.qpos[a + 3:a + 7].copy())
                   for n, a in self._obj_qadr.items()}
            only_lowered = all(got[n][0][2] <= stored[n][0][2] + 5e-3
                               and np.linalg.norm(got[n][0][:2] - stored[n][0][:2]) < 0.02
                               for n in stored)
            if not only_lowered:
                logger.warning(f"[init] {self.task}: settling the stored pose moved objects "
                               f"up/sideways (overlapping init state); keeping stored poses")
            self._settled_bases = got if only_lowered else stored
            self._clean_state()
        self._place_objects(randomize_init)
        mujoco.mj_forward(self.model, self.data)

        self.curr_gripper_open = {n: 1 for n in self.arms}
        self.phase = Phase.SETUP
        self.settle_until_rest()
        self.rest_z = {n: float(self.data.body(n).xpos[2]) for n in self.init_bases}

        for n in self.arms:
            proprio = self.observe_proprio(n)
            self.move_to(proprio.eef_pos, self._home_euler[n], 1, arm=n)

        self.num_step = 0
        self.redeal_episode()

    STATE_SPEC = mujoco.mjtState.mjSTATE_INTEGRATION.value

    def _lift_out_of_penetration(self):
        owner = {int(g): n for n in self._free_objects for g in self._obj_geoms[n]}
        for _ in range(12):
            mujoco.mj_forward(self.model, self.data)
            lift = {}
            for c in self.data.contact[: self.data.ncon]:
                if c.dist >= -1e-4:
                    continue
                for g in (c.geom1, c.geom2):
                    n = owner.get(int(g))
                    if n is not None:
                        lift[n] = max(lift.get(n, 0.0), -float(c.dist))
            if not lift:
                return
            for n, d in lift.items():
                self.data.qpos[self._obj_qadr[n] + 2] += d + 1e-4

    def _clean_state(self):
        mujoco.mj_resetData(self.model, self.data)
        for n in self.arms:
            ik = self.ik[n]
            for a, q in zip(ik.qadr, ik.q_home):
                self.data.qpos[a] = q
            for a, q in zip(self._finger_qadr[n], self._open_qpos[n]):
                self.data.qpos[a] = q
            ik.reset(self.data)
            self.data.ctrl[self._grip_act[n]] = self._grip_open_ctrl[n]

    def sim_state(self):
        n = mujoco.mj_stateSize(self.model, self.STATE_SPEC)
        state = np.empty(n)
        mujoco.mj_getState(self.model, self.data, state, self.STATE_SPEC)
        return state

    def redeal_episode(self):
        self._check = grade_episode.make_checker(self.spec["success"], self.task)[0]
        self.episode_t0 = float(self.data.time)
        self._dynamics("deal")
        self._dealt = True

    def _dynamics(self, hook):
        fn = getattr(self.dynamics, hook, None)
        return fn(self) if fn else None

    def reset_to_sim_state(self, state):
        mujoco.mj_setState(self.model, self.data, np.asarray(state, dtype=np.float64),
                           self.STATE_SPEC)
        self._dynamics("restore")
        mujoco.mj_forward(self.model, self.data)

    def forward(self):
        mujoco.mj_forward(self.model, self.data)

    def checkpoint(self):
        return {
            "sim_state": self.sim_state(),
            "grading_state": copy.deepcopy(self._check),
            "q_des": {n: self.ik[n].q_des.copy() for n in self.arms},
            "curr_gripper_open": dict(self.curr_gripper_open),
            "num_step": self.num_step,
        }

    def restore(self, checkpoint):
        self.reset_to_sim_state(checkpoint["sim_state"])
        self._check = copy.deepcopy(checkpoint["grading_state"])
        for n, q_des in checkpoint["q_des"].items():
            self.ik[n].q_des = q_des.copy()
        self.curr_gripper_open = dict(checkpoint["curr_gripper_open"])
        self.num_step = checkpoint["num_step"]

    def _step_physics(self):
        if self._dealt:
            self._dynamics("physics_step")
        t = time.perf_counter()
        mujoco.mj_step(self.model, self.data)
        self.step_stats[0] = max(self.step_stats[0], int(self.data.ncon))
        self.step_stats[1] = max(self.step_stats[1], int(self.data.nefc))
        self.step_stats[2] += time.perf_counter() - t
        self.step_stats[3] += 1

    def contact_body_pairs(self):
        m, d = self.model, self.data
        return [(int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2]))
                for c in d.contact[:d.ncon]]

    def _advance_episode_state(self):
        self.num_step += 1
        self._dynamics("control_step")
        if self.phase is Phase.RUNNING and (self.num_step >= self.max_len or self.scene_over):
            self.phase = Phase.DONE

    @property
    def scene_over(self) -> bool:
        return bool(self._dynamics("over"))

    @property
    def episode_over(self) -> bool:
        return self.phase is Phase.DONE

    def move_to(self, *a, **kw):
        return self.ctl.move_to(*a, **kw)

    def apply_action(self, *a, **kw):
        return self.ctl.apply_action(*a, **kw)

    def robot_clear(self, body_names) -> bool:
        m = self.model
        targets = set()
        for name in body_names:
            bid = m.body(name).id
            while bid and not m.body_jntnum[bid]:
                bid = m.body_parentid[bid]
            targets.add(bid)

        def targeted(bid):
            while bid:
                if bid in targets:
                    return True
                bid = m.body_parentid[bid]
            return False

        for b1, b2 in self.contact_body_pairs():
            r1 = m.body(b1).name.startswith(self.robot_prefix)
            if r1 == m.body(b2).name.startswith(self.robot_prefix):
                continue
            other = b2 if r1 else b1
            if targeted(other):
                return False
        return True

    def gripper_width(self, arm=None) -> float:
        arm = arm or self.arms[0]
        return float(np.dot(self._width_coef[arm], self.data.qpos[self._finger_qadr[arm]]))

    def finger_roots(self, arm=None) -> frozenset[int]:
        return self._finger_roots[arm or self.arms[0]]

    def held_bodies(self, arm, data=None) -> set[int]:
        m, d = self.model, self.data if data is None else data
        pads = {}
        for c in d.contact[:d.ncon]:
            for finger, other in ((c.geom1, c.geom2), (c.geom2, c.geom1)):
                root = self.finger_root(int(m.geom_bodyid[finger]), arm)
                if root is not None:
                    pads.setdefault(int(m.geom_bodyid[other]), set()).add(root)
        roots = self.finger_roots(arm)
        return {b for b, p in pads.items()
                if p == roots and not m.body(b).name.startswith(self.robot_prefix)}

    def finger_root(self, body: int, arm=None) -> int | None:
        roots = self.finger_roots(arm)
        while body > 0:
            if body in roots:
                return int(body)
            body = int(self.model.body_parentid[body])
        return None

    def observe_proprio(self, arm=None) -> Proprio:
        arm = arm or self.arms[0]
        site = self._site[arm]
        quat = R.from_matrix(self.data.site_xmat[site].reshape(3, 3)).as_quat()
        return Proprio(
            eef_pos=self.data.site_xpos[site].copy(),
            eef_quat=quat,
            gripper_open=self.gripper_width(arm) / self.gripper_max_width,
        )

    def _renderer_for(self):
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, self.image_size, self.image_size)
            _close_before_egl_exit(self, self._renderer)
        return self._renderer

    def observe_camera(self, depth=True) -> dict[str, np.ndarray]:
        obs = {}
        r = self._renderer_for()
        for cam in self.cameras:
            r.update_scene(self.data, cam)
            rgb = r.render()
            obs[f"{cam}_image"] = rgb
            if depth:
                r.enable_depth_rendering()
                d = r.render()
                r.disable_depth_rendering()
                obs[f"{cam}_depth"] = d[..., None]
        return obs

    def rerender(self, obs: dict, force: bool = False) -> dict:
        from core import render

        if not (force or render.enabled()):
            return obs
        try:
            if self._blender is None:
                self._blender = render.start(self)
            if not self._blender:
                return obs
            t0 = time.perf_counter()
            done = self._blender.render()
        except Exception as e:
            if not force:
                logger.error(f"[render] ROBOARCADE_RENDER=blender and blender failed "
                             f"({type(e).__name__}: {e}); ending the seed as render_failed")
                raise SystemExit(render.RENDER_FAILED_EXIT)
            logger.warning(f"[render] preview stays mujoco: {type(e).__name__}: {e}")
            self._blender = False
            return obs
        logger.info(f"[render] {len(done)} cameras in "
                    f"{(time.perf_counter() - t0) * 1000:.0f} ms")
        for cam, rgb in done.items():
            key = f"{cam}_image"
            if key not in obs or rgb.shape[:2] != obs[key].shape[:2]:
                continue
            obs[key] = rgb[..., :3]
        return obs

    def observe(self, depth=True) -> dict[str, np.ndarray]:
        obs = self.observe_camera(depth)
        proprio = self.observe_proprio()
        obs["ee_pos"] = proprio.eef_pos
        obs["ee_euler"] = proprio.eef_euler
        obs["gripper_open"] = proprio.gripper_open_np
        return obs

    def get_camera_intrinsics(self, cam: str) -> np.ndarray:
        fovy = float(self.model.cam(cam).fovy[0])
        size = self.image_size
        return camera_intrinsics(fovy, size, size)

    def get_camera_extrinsics(self, cam: str) -> np.ndarray:
        cid = self.model.cam(cam).id
        T = np.eye(4)
        T[:3, :3] = self.data.cam_xmat[cid].reshape(3, 3) @ np.diag([1.0, -1.0, -1.0])
        T[:3, 3] = self.data.cam_xpos[cid]
        return T

    def get_point_cloud(self, obs, crop_table: bool):
        pts_all, col_all = [], []
        z_min = self.pc["z_min_crop_table"] if crop_table else self.pc["z"][0]
        for cam in self.cameras:
            if self.camera_role(cam) == "wrist":
                continue
            depth = obs[f"{cam}_depth"].squeeze()
            pts = deproject(depth, self.get_camera_intrinsics(cam),
                            self.get_camera_extrinsics(cam))
            ok = (
                (pts[:, 0] >= self.pc["x"][0]) & (pts[:, 0] <= self.pc["x"][1])
                & (pts[:, 1] >= self.pc["y"][0]) & (pts[:, 1] <= self.pc["y"][1])
                & (pts[:, 2] >= z_min) & (pts[:, 2] <= self.pc["z"][1])
            )
            pts_all.append(pts[ok])
            col_all.append(obs[f"{cam}_image"].reshape(pts.shape)[ok] / 255.0)
        return np.vstack(pts_all), np.vstack(col_all)
