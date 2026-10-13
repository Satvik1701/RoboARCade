"""Draws where every object starts for a seed.

Also answers the same reach, region and footprint queries on demand for the teleop init editor.
"""
import logging

import mujoco
import numpy as np
from scipy.ndimage import binary_erosion
from scipy.spatial.transform import Rotation as R

from core import grade_episode
from core.tasks import ASSETS

logger = logging.getLogger(__name__)

DEFAULT_INIT_REGION = {"x": [-0.10, 0.10], "y": [-0.10, 0.10], "yaw": [-np.pi, np.pi]}

REACH_MARGIN_FRACTION = 0.20

CORNER_SIGNS = np.array(np.meshgrid(*([-1.0, 1.0],) * 3)).reshape(3, -1).T


def _obb_intersect(A, B) -> bool:
    cA, RA, hA = A
    cB, RB, hB = B
    R = RA.T @ RB
    absR = np.abs(R) + 1e-9
    t = RA.T @ (cB - cA)
    for i in range(3):
        if abs(t[i]) > hA[i] + hB @ absR[i, :]:
            return False
    for j in range(3):
        if abs(t @ R[:, j]) > hA @ absR[:, j] + hB[j]:
            return False
    for i in range(3):
        for j in range(3):
            i1, i2, j1, j2 = (i + 1) % 3, (i + 2) % 3, (j + 1) % 3, (j + 2) % 3
            ra = hA[i1] * absR[i2, j] + hA[i2] * absR[i1, j]
            rb = hB[j1] * absR[i, j2] + hB[j2] * absR[i, j1]
            if abs(t[i2] * R[i1, j] - t[i1] * R[i2, j]) > ra + rb:
                return False
    return True


def _yaw_bands(reg):
    w = reg.get("yaw_offset_deg", [0.0, 0.0])
    return [list(b) for b in w] if w and isinstance(w[0], (list, tuple)) else [list(w)]


class Layout:
    def _rest_offset(self, name):
        m = self.model
        zs = []
        for g in np.flatnonzero(m.geom_bodyid == m.body(name).id):
            c, h = m.geom_aabb[g][:3], m.geom_aabb[g][3:]
            rot = R.from_quat(m.geom_quat[g][[1, 2, 3, 0]]).as_matrix()
            zs.append(float(m.geom_pos[g][2] + rot[2] @ c - np.abs(rot[2]) @ h))
        return -min(zs)

    def _base_placements(self):
        rnd = self.spec["randomization"]
        t = rnd.get("type")
        bases = {}
        if t == "init_states":
            state = rnd["states"][0]
            for jname, q in state.get("joints", {}).items():
                self.data.qpos[self.model.joint(jname).qposadr[0]] = q
            for name, pose in state["objects"].items():
                bases[name] = (np.array(pose[:3], float), np.array(pose[3:7], float))
        elif t == "uniform":
            ref = np.array(rnd["reference"], float)
            for name, r in rnd["objects"].items():
                z = ref[2] + r.get("z_offset", rnd["z_offset"]) + self._rest_offset(name)
                bases[name] = (np.array([ref[0], ref[1], z]), np.array([1.0, 0, 0, 0]))
        for name in self._free_objects:
            if name not in bases:
                a = self._obj_qadr[name]
                bases[name] = (self.model.qpos0[a:a + 3].copy(), self.model.qpos0[a + 3:a + 7].copy())
        for name, (p, q) in self._static_base.items():
            bases[name] = (p.copy(), q.copy())
        if self._settled_bases:
            bases.update({n: (p.copy(), q.copy()) for n, (p, q) in self._settled_bases.items()})
        return bases

    REJECT_TRIES = 300
    STROKE_SAMPLES = 5
    LAYOUT_TRIES = 4

    def _place_objects(self, randomize=False, pinned=None):
        self._legal_cache.clear()
        rnd = self.spec["randomization"]
        bases = self._base_placements()
        self.init_bases = bases
        self.rest_z = {n: float(p[2]) for n, (p, _q) in bases.items()}
        self._reach_site_lift = max(
            (float(self._aabb(self._obj_geoms[n])[1][2] - self.data.body(n).xpos[2])
             for n in self._spec_names("obj") if n in self._obj_geoms), default=0.0)

        self._write_objects(bases)
        if not randomize:
            return
        mujoco.mj_kinematics(self.model, self.data)
        self._rim_at_base = {n: self._rim_reachable(n, xyz) for n in bases
                             if (xyz := self._live_xyz(n)) is not None}
        self._centre_at_base = {n: self._any_arm_reachable(xyz) for n in bases
                                if (xyz := self._live_xyz(n)) is not None}
        self._path_at_base = {}
        self._path_at_base = {n: self._handle_path_reachable([n]) for n in bases}
        self._heading_at_base = {n: self._front_heading(n) for n in self._face_robot}
        authored_overlaps = self._all_overlaps()
        grouped = [list(g) for g in (rnd.get("groups") or [])]
        drawable = set(self._free_objects) | set(self._static_objects)
        rides = {r["obj"]: r["of"] for r in self._referents if r["kind"] == "on"}
        riding_fixed = {o for o, sup in rides.items() if sup not in drawable}
        for o, sup in rides.items():
            if sup not in drawable:
                continue
            g = next((g for g in grouped if sup in g), None)
            if g is None:
                g = [sup]
                grouped.append(g)
            if o not in g:
                g.append(o)
            grouped = [x if x is g else [n for n in x if n != o] for x in grouped]
        grouped = [[n for n in g if n not in riding_fixed] for g in grouped]
        grouped = [g for g in grouped if g]
        slots = rnd.get("slots")
        if slots is not None:
            parent, members, poses = (slots["parent"], slots["members"],
                                      np.asarray(slots["poses"], float))
            names = [parent, *members]
            if (not members or poses.shape != (len(poses), 4) or len(poses) < len(members)
                    or len(names) != len(set(names)) or not set(names) <= drawable
                    or not np.isfinite(poses).all()):
                raise ValueError("randomization.slots requires distinct bodies and finite poses")
            slots = {"parent": parent, "members": members, "poses": poses}
            member_set = set(members)
            grouped = [[n for n in g if n not in member_set] for g in grouped]
            grouped = [g for g in grouped if g]
            parent_group = next((g for g in grouped if parent in g), None)
            if parent_group is None:
                parent_group = [parent]
                grouped.append(parent_group)
            parent_group.extend(members)
        loose = [n for n in self._free_objects
                 if n not in riding_fixed and not any(n in g for g in grouped)]
        groups = grouped + [[n] for n in loose]
        if pinned:
            self._write_objects(pinned)
            mujoco.mj_kinematics(self.model, self.data)
            groups = [g for g in groups if not (set(g) & set(pinned))]
        groups.sort(key=lambda g: (not any(n in self._placement_regions for n in g),
                                   -max(self.footprint_radius(n) for n in g)))
        for attempt in range(self.LAYOUT_TRIES):
            if attempt:
                logger.warning(f"[init] {self.task}: relaying the whole layout "
                               f"(attempt {attempt + 1}/{self.LAYOUT_TRIES}): "
                               f"a group had no legal placement")
            try:
                self._place_groups(groups, bases, authored_overlaps, set(pinned or ()) | riding_fixed,
                                   slots)
                break
            except RuntimeError:
                if attempt == self.LAYOUT_TRIES - 1:
                    raise

    def _write_slots(self, slots, parent_pose):
        parent_pos, parent_quat = parent_pose
        parent_rot = R.from_quat(np.asarray(parent_quat)[[1, 2, 3, 0]])
        chosen = self._rng.choice(len(slots["poses"]), len(slots["members"]), replace=False)
        poses = {}
        for member, index in zip(slots["members"], chosen):
            x, y, z, yaw = slots["poses"][index]
            rotation = parent_rot * R.from_rotvec([0, 0, yaw])
            poses[member] = (np.asarray(parent_pos) + parent_rot.apply([x, y, z]),
                             rotation.as_quat()[[3, 0, 1, 2]])
        self._write_objects(poses)

    def _place_groups(self, groups, bases, authored_overlaps, placed, slots=None):
        authored_only = slots is None and not self.full_rand and all(
            all(abs(float(value)) < 1e-12 for value in self.init_region[axis])
            for axis in ("x", "y", "yaw"))
        if authored_only:
            for group in groups:
                self._write_objects({name: bases[name] for name in group})
                placed.update(group)
            return

        drawn_off_base = set()
        articulated_down = []
        for g in groups:
            group_slots = slots if slots and slots["parent"] in g else None
            draw_names = ([n for n in g if n not in group_slots["members"]]
                          if group_slots else g)
            grp_names = set(g)
            exempt_overlaps = {p for p in authored_overlaps
                               if p <= grp_names or any(not isinstance(k, str) for k in p)}
            ctx = self._others_ctx(g, placed)
            for _ in range(self.REJECT_TRIES):
                draw = self._sample_group(bases, draw_names)
                self._write_objects(draw)
                if group_slots:
                    self._write_slots(group_slots, draw[group_slots["parent"]])
                if self._group_accepted(g, exempt_overlaps, ctx, articulated_down, placed):
                    placed.update(g)
                    drawn_off_base.update(g)
                    if any(self._joints_of(n) for n in g):
                        articulated_down.append(list(g))
                    break
            else:
                self._write_objects({n: bases[n] for n in draw_names})
                if group_slots:
                    self._write_slots(group_slots, bases[group_slots["parent"]])
                if self._disallowed_overlap(g, exempt_overlaps, self._others_ctx(g, drawn_off_base)) \
                        or not self._min_sep_ok(g) \
                        or not self.can_open(g) \
                        or not all(self.can_open(a) for a in articulated_down):
                    raise RuntimeError(
                        f"[init] {self.task}: task cannot be initialized: no collision-"
                        f"free placement for {sorted(g)} in {self.REJECT_TRIES} draws and "
                        f"the authored fallback pose is blocked; the randomization ranges "
                        f"are too tight for these objects")
                placed.update(g)
                if any(self._joints_of(n) for n in g):
                    articulated_down.append(list(g))
                logger.warning(f"[init] {self.task}: no clear placement for {g} "
                               f"in {self.REJECT_TRIES} draws; left at its authored pose")

    def _group_accepted(self, group, exempt_overlaps, ctx, articulated_down, placed):
        return not self._disallowed_overlap(group, exempt_overlaps, ctx) \
            and not self._off_support(group) \
            and self._within_reach(group) \
            and self._in_pc_crop(group) \
            and self._min_sep_ok(group) \
            and self._referents_ok(group) \
            and self._graspable(group, placed) \
            and self._base_distance_ok(group) \
            and self._handle_path_reachable(group) \
            and self.can_open(group) \
            and all(self.can_open(a) for a in articulated_down)

    def _xy_points(self, name):
        pts = [(c + (CORNER_SIGNS * h) @ rot.T)[:, :2]
               for c, rot, h in (self._obb(int(g)) for g in self._subtree_geoms(name))]
        return np.concatenate(pts) if pts else np.empty((0, 2))

    def footprint_radius(self, name) -> float:
        pts = self._xy_points(name)
        if not len(pts):
            return 0.0
        return float(np.linalg.norm(pts - self.data.body(name).xpos[:2], axis=1).max())

    def static_poses(self):
        names = list(self._static_objects)
        poses = np.array([[*self.model.body(n).pos, *self.model.body(n).quat] for n in names])
        return names, poses.reshape(-1, 7)

    def set_static_poses(self, names, poses):
        self._write_objects({n: (p[:3], p[3:]) for n, p in zip(names, poses)})
        mujoco.mj_forward(self.model, self.data)

    def _live_xyz(self, n):
        if n in self._obj_qadr:
            return np.array(self.data.qpos[self._obj_qadr[n] : self._obj_qadr[n] + 3])
        if n in self._static_base:
            return np.array(self.model.body(n).pos[:3])
        return None

    def _base_distance_ok(self, names) -> bool:
        for n in names:
            need = self._min_base_distance.get(n)
            p = self._live_xyz(n) if need else None
            if p is not None and float(np.linalg.norm(p[:2] - self._base_xy)) < need:
                return False
        return True

    def _in_pc_crop(self, names) -> bool:
        x0, x1, y0, y1 = self.placement_region()
        for n in names:
            pts = self._xy_points(n)
            if not len(pts):
                continue
            lo, hi = pts.min(0), pts.max(0)
            if not (x0 <= lo[0] and hi[0] <= x1 and y0 <= lo[1] and hi[1] <= y1):
                return False
        return True

    def _any_arm_reachable(self, xyz) -> bool:
        return any(self.reachable(xyz, arm) for arm in self.arms)

    def _rim_reachable(self, n, xyz) -> bool:
        fr = min(self.footprint_radius(n), self.gripper_max_width)
        return all(self._any_arm_reachable([xyz[0] + dx, xyz[1] + dy, xyz[2]])
                   for dx, dy in ((fr, 0), (-fr, 0), (0, fr), (0, -fr)))

    def _body_reach_ok(self, n, xyz) -> bool:
        xy = np.asarray(xyz[:2], float)
        if n in self._articulated_reach:
            v = xy - self._base_xy
            nv = float(np.linalg.norm(v))
            if nv > 1e-9:
                near_face = xy - v / nv * self.footprint_radius(n)
                return self._any_arm_reachable([near_face[0], near_face[1], float(xyz[2])])
        if not self._any_arm_reachable(xyz) and self._centre_at_base.get(n, True):
            return False
        return (n in self._articulated_reach or not self._rim_at_base.get(n, True)
                or self._rim_reachable(n, xyz))

    def _reach_ok(self, bodies, sites) -> bool:
        return (all(self._body_reach_ok(n, xyz)
                    and (not (dz := self._lift_reach.get(n))
                         or self._any_arm_reachable(np.asarray(xyz, float) + [0, 0, dz]))
                    for n, xyz in bodies)
                and all(self._any_arm_reachable(s + [0, 0, dz])
                        for s in sites for dz in (0.0, self._reach_site_lift)))

    def _within_reach(self, names) -> bool:
        bodies = [(n, xyz) for n in names if n not in self._reach_site_bodies
                  and (xyz := self._live_xyz(n)) is not None]
        ids = {self.model.body(n).id for n in names}
        sites = [self.data.site_xpos[sid] for sid in (self.model.site(s).id for s in self._reach_sites)
                 if self._ancestor_in(self.model.site_bodyid[sid], ids)]
        return self._reach_ok(bodies, sites)

    def clipped_station(self, body, reg):
        x0, x1, y0, y1 = self.group_center_box([body])
        dy = list(reg["y"])
        cy = [max(dy[0], y0), min(dy[1], y1)]
        dx = list(reg["x"]) if reg.get("x") else None
        cx = [max(dx[0], x0), min(dx[1], x1)] if dx else None
        clipped = (cy != dy) or (dx is not None and cx != dx)
        if cy[0] > cy[1]:
            cy = [max(y0, min(y1, dy[0]))] * 2
        if cx and cx[0] > cx[1]:
            cx = [max(x0, min(x1, dx[0]))] * 2
        return cx, cy, clipped

    def _write_objects(self, poses):
        for name, (p, q) in poses.items():
            if name in self._obj_qadr:
                a = self._obj_qadr[name]
                self.data.qpos[a:a + 3] = p
                self.data.qpos[a + 3:a + 7] = q
            else:
                body = self.model.body(name)
                body.pos[:] = p
                body.quat[:] = q
        flags = self.model.opt.enableflags
        if flags & int(mujoco.mjtEnableBit.mjENBL_SLEEP) and any(
                name not in self._obj_qadr for name in poses):
            self.model.opt.enableflags = flags & ~int(mujoco.mjtEnableBit.mjENBL_SLEEP)
            mujoco.mj_kinematics(self.model, self.data)
            self.model.opt.enableflags = flags

    def _joints_of(self, name):
        m = self.model
        out = []
        for i in [k for k in range(m.nbody) if self._ancestor_in(k, {m.body(name).id})]:
            for j in range(m.body_jntadr[i], m.body_jntadr[i] + m.body_jntnum[i]):
                if m.jnt_type[j] not in (mujoco.mjtJoint.mjJNT_HINGE,
                                         mujoco.mjtJoint.mjJNT_SLIDE):
                    continue
                jb = int(m.jnt_bodyid[j])
                out.append((j, np.array(
                    [g for k in range(m.nbody) if self._ancestor_in(k, {jb})
                     for g in range(m.body_geomadr[k],
                                    m.body_geomadr[k] + m.body_geomnum[k])], int)))
        return out

    def _spec_names(self, key):
        return {n[key] for n in grade_episode.nodes(self.spec["success"]) if isinstance(n.get(key), str)}

    def can_open(self, names) -> bool:
        m, d = self.model, self.data
        own_bodies = {b for n in names for b in range(m.nbody)
                      if self._ancestor_in(b, {m.body(n).id})}
        want = self._spec_names("joint")
        qsave = d.qpos.copy()
        try:
            for n in names:
                js = self._joints_of(n)
                scoped = [(j, mv) for j, mv in js if m.joint(j).name in want]
                for j, moving in (scoped or js):
                    if not len(moving):
                        continue
                    d.qpos[:] = qsave
                    lo, hi = m.jnt_range[j]
                    d.qpos[m.jnt_qposadr[j]] = lo if abs(lo) > abs(hi) else hi
                    mujoco.mj_kinematics(m, d)
                    if len(self._robot_base_geoms) \
                            and self._geoms_overlap(moving, self._robot_base_geoms):
                        return False
                    for b, gs in self._obstacle_geoms.items():
                        if b not in own_bodies and len(gs) and self._geoms_overlap(moving, gs):
                            return False
        finally:
            d.qpos[:] = qsave
            mujoco.mj_kinematics(m, d)
        return True

    def _front_heading(self, name):
        m, d = self.model, self.data
        bid = m.body(name).id
        for i in [k for k in range(m.nbody) if self._ancestor_in(k, {bid})]:
            for j in range(m.body_jntadr[i], m.body_jntadr[i] + m.body_jntnum[i]):
                t = m.jnt_type[j]
                if t == mujoco.mjtJoint.mjJNT_SLIDE:
                    lo, hi = m.jnt_range[j]
                    v = (-1.0 if abs(lo) > abs(hi) else 1.0) * (
                        d.xmat[i].reshape(3, 3) @ m.jnt_axis[j])
                elif t == mujoco.mjtJoint.mjJNT_HINGE:
                    v = d.xpos[i][:2] - d.xpos[bid][:2]
                else:
                    continue
                if np.hypot(v[0], v[1]) > 1e-6:
                    return float(np.arctan2(v[1], v[0]))
        return None

    EDGE_MARGIN_M = 0.02

    def placement_region(self):
        c, sr = self.pc, self._support_rect
        m = self.EDGE_MARGIN_M
        return (max(c["x"][0], sr[0] if sr else -9) + m,
                min(c["x"][1], sr[1] if sr else 9) - m,
                max(c["y"][0], sr[2] if sr else -9) + m,
                min(c["y"][1], sr[3] if sr else 9) - m)

    def group_center_box(self, names):
        rx0, rx1, ry0, ry1 = self.placement_region()
        piv = np.mean([self.init_bases[n][0][:2] for n in names], axis=0)
        x0, y0, x1, y1 = -9.0, -9.0, 9.0, 9.0
        for n in names:
            mx0, mx1, my0, my1 = self.footprint_extent(n)
            ox, oy = self.init_bases[n][0][:2] - piv
            x0, x1 = max(x0, rx0 + mx0 - ox), min(x1, rx1 - mx1 - ox)
            y0, y1 = max(y0, ry0 + my0 - oy), min(y1, ry1 - my1 - oy)
        return x0, x1, y0, y1

    YAW_SAMPLES = 33

    def footprint_extent(self, name):
        r = self.footprint_radius(name)
        if name in self._face_robot or name in self._placement_regions:
            return r, r, r, r
        lo, hi = self.init_region["yaw"]
        if hi - lo >= 2 * np.pi - 1e-6:
            return r, r, r, r
        pts = self._xy_points(name) - self.data.body(name).xpos[:2]
        if not len(pts):
            return r, r, r, r
        out = np.zeros(4)
        for a in np.linspace(lo, hi, self.YAW_SAMPLES):
            ca, sa = np.cos(a), np.sin(a)
            q = pts @ np.array([[ca, sa], [-sa, ca]])
            out = np.maximum(out, [-q[:, 0].min(), q[:, 0].max(),
                                   -q[:, 1].min(), q[:, 1].max()])
        return tuple(float(v) for v in np.minimum(out, r))

    def legal_cells(self, names):
        key = tuple(sorted(names))
        cache = self._legal_cache
        if key not in cache:
            step = 0.02
            x0, x1, y0, y1 = box = self.group_center_box(names)
            xs = np.arange(np.ceil(x0 / step) * step, x1 + 1e-9, step)
            ys = np.arange(np.ceil(y0 / step) * step, y1 + 1e-9, step)
            pts = [(x, y) for x in xs for y in ys if self.center_fits(names, x, y, box)]
            qs, bases = self.data.qpos.copy(), self.init_bases
            pivot = np.mean([bases[n][0][:2] for n in names], axis=0)
            authored_overlaps = self._all_overlaps()
            exempt_overlaps = {q for q in authored_overlaps
                               if q <= set(names) or any(not isinstance(k, str) for k in q)}
            fixed_ctx = self._others_ctx(names, set())
            keep = []
            for (x, y) in pts:
                dx, dy = x - pivot[0], y - pivot[1]
                yaw = self._face_yaw(names, pivot, dx, dy) or 0.0
                self._write_objects(self._group_pose(bases, names, dx, dy, yaw))
                mujoco.mj_kinematics(self.model, self.data)
                if (not self._disallowed_overlap(names, exempt_overlaps, fixed_ctx)
                        and self.can_open(names)
                        and self._handle_path_reachable(names)):
                    keep.append((x, y))
            self.data.qpos[:] = qs
            mujoco.mj_kinematics(self.model, self.data)
            need = max((self._min_base_distance.get(n, 0.0) for n in names), default=0.0)
            if need > 0:
                keep = [c for c in keep if np.linalg.norm(np.asarray(c, float) - self._base_xy) >= need]
            cache[key] = np.array(keep, float)
        return cache[key]

    def center_fits(self, names, px, py, box=None) -> bool:
        x0, x1, y0, y1 = box if box is not None else self.group_center_box(names)
        if not (x0 <= px <= x1 and y0 <= py <= y1):
            return False
        piv = np.mean([self.init_bases[n][0][:2] for n in names], axis=0)
        bodies = [(n, [px + b[0] - piv[0], py + b[1] - piv[1], float(self.rest_z.get(n, b[2]))])
                  for n in names if n not in self._reach_site_bodies
                  for b in [self.init_bases[n][0]]]
        sites = []
        for site in self._reach_sites:
            sid = self.model.site(site).id
            owner = self.model.site_bodyid[sid]
            n = next((n for n in names
                      if self._ancestor_in(owner, {self.model.body(n).id})), None)
            if n is None:
                continue
            off = self.data.site_xpos[sid] - self.data.body(n).xpos
            b = self.init_bases[n][0]
            sites.append(np.array([px + b[0] - piv[0] + off[0], py + b[1] - piv[1] + off[1],
                                   float(self.rest_z.get(n, b[2])) + off[2]]))
        return self._reach_ok(bodies, sites)

    def _sample_group(self, bases, names):
        r = self.init_region
        pivot = np.mean([bases[n][0][:2] for n in names], axis=0)
        if self.full_rand:
            regions = next((self._placement_regions[n] for n in names
                            if n in self._placement_regions), None)
            if regions:
                reg = regions[int(self._rng.integers(len(regions)))]
                body0 = next(n for n in names if n in self._placement_regions)
                cx, cy, _ = self.clipped_station(body0, reg)
                dy = float(self._rng.uniform(*cy)) - pivot[1]
                dx = (float(self._rng.uniform(*cx)) - pivot[0]) if cx else 0.0
                bands = _yaw_bands(reg)
                lo, hi = bands[int(self._rng.integers(len(bands)))]
                yaw_off = np.radians(float(self._rng.uniform(lo, hi)))
                yaw = self._face_yaw(names, pivot, dx, dy)
                return self._group_pose(bases, names, dx, dy, (yaw or 0.0) + yaw_off)
            cells = self.legal_cells(names)
            if len(cells):
                cx, cy = cells[int(self._rng.integers(len(cells)))]
                dx = cx + self._rng.uniform(-0.01, 0.01) - pivot[0]
                dy = cy + self._rng.uniform(-0.01, 0.01) - pivot[1]
            else:
                dx = dy = 0.0
        else:
            dx = self._rng.uniform(*r["x"])
            dy = self._rng.uniform(*r["y"])
        yaw = self._face_yaw(names, pivot, dx, dy)
        if yaw is None:
            yaw = self._rng.uniform(*r["yaw"])
        return self._group_pose(bases, names, dx, dy, yaw)

    def _group_pose(self, bases, names, dx, dy, yaw):
        rz = R.from_rotvec([0, 0, yaw])
        pivot = np.mean([bases[n][0][:2] for n in names], axis=0)
        out = {}
        for n in names:
            pos, quat = bases[n]
            xy = pivot + rz.as_matrix()[:2, :2] @ (pos[:2] - pivot) + [dx, dy]
            out[n] = (np.array([xy[0], xy[1], pos[2]]),
                      (rz * R.from_quat(quat[[1, 2, 3, 0]])).as_quat()[[3, 0, 1, 2]])
        return out

    def _face_yaw(self, names, pivot, dx, dy):
        if not (self._face_robot & set(names)):
            return None
        b = self._base_xy
        new = np.asarray(pivot, float) + [dx, dy]
        f0 = next((self._heading_at_base[n] for n in names if self._heading_at_base.get(n) is not None),
                  np.arctan2(b[1] - pivot[1], b[0] - pivot[0]))
        return float(np.arctan2(b[1] - new[1], b[0] - new[0]) - f0)

    def _others_ctx(self, names, placed=None):
        mujoco.mj_kinematics(self.model, self.data)
        ns = set(names)
        others = [(n, self._obj_geoms[n]) for n in self._free_objects
                  if n not in ns and n in self._obj_geoms
                  and (placed is None or n in placed)]
        own_static_bodies = {int(self.model.geom_bodyid[g])
                             for n in ns if n in self._static_geoms for g in self._static_geoms[n]}
        others += [(b, g) for b, g in self._obstacle_geoms.items() if b not in own_static_bodies]
        keys = [k for k, gs in others for _ in gs]
        if not keys:
            return [], np.zeros(0, int), np.zeros((0, 3)), np.zeros((0, 3))
        gids = np.concatenate([gs for _, gs in others])
        c, r = self.data.geom_xpos[gids], self.model.geom_rbound[gids][:, None]
        lo = c - r
        if self._support_top_z is not None:
            free = np.array([k in self._obj_geoms for k in keys])
            lo[free, 2] = np.minimum(lo[free, 2], self._support_top_z)
        return keys, gids, lo, c + r

    def _overlapping(self, names, ctx):
        mujoco.mj_kinematics(self.model, self.data)
        keys, gids, lo, hi = ctx
        if not keys:
            return
        grp = [(n, self._obj_geoms[n]) for n in names if n in self._obj_geoms]
        grp += [(n, self._static_geoms[n]) for n in names if n in self._static_geoms]
        for gn, ggs in grp:
            glo, ghi = self._aabb(ggs)
            column = self._landing_column(ggs, self.LANDING_MARGIN_M) if gn in self._obj_geoms else None
            gpad = self.LANDING_MARGIN_M if column is not None else 0.0
            if column is not None:
                glo = np.minimum(glo, column[0] - column[2])
            near = np.nonzero(~(np.any(ghi + gpad < lo, axis=1) | np.any(hi < glo - gpad, axis=1)))[0]
            for i in near:
                if self._geoms_overlap(ggs, gids[i:i + 1], gpad):
                    yield gn, keys[i]
                    continue
                ocol = self._landing_column(gids[i:i + 1]) if keys[i] in self._obj_geoms else None
                if column is not None:
                    if _obb_intersect(column, ocol or self._obb(int(gids[i]))):
                        yield gn, keys[i]
                elif ocol is not None and any(_obb_intersect(ocol, self._obb(int(g), gpad))
                                              for g in ggs):
                    yield gn, keys[i]

    LANDING_MARGIN_M = 0.02

    def _landing_column(self, gs, pad=0.0):
        if self._support_top_z is None:
            return None
        pts = np.vstack([c + (CORNER_SIGNS * h) @ rot.T for c, rot, h in (self._obb(int(g)) for g in gs)])
        bottom, top = float(pts[:, 2].min()), float(pts[:, 2].max())
        if bottom < self._support_top_z + 0.005:
            return None
        lo, hi = pts[:, :2].min(0) - pad, pts[:, :2].max(0) + pad
        centre = np.array([*(lo + hi) / 2, (self._support_top_z + top) / 2])
        half = np.array([*(hi - lo) / 2, (top - self._support_top_z) / 2])
        return centre, np.eye(3), half

    def _disallowed_overlap(self, names, authored_overlaps, ctx):
        return any(frozenset((n, k)) not in authored_overlaps for n, k in self._overlapping(names, ctx))

    def _aabb(self, gs):
        c = self.data.geom_xpos[gs]
        r = self.model.geom_rbound[gs][:, None]
        return (c - r).min(0), (c + r).max(0)

    def _min_sep_ok(self, names) -> bool:
        ns = set(names)
        for a, b, d in self._min_sep_xy:
            if a not in ns and b not in ns:
                continue
            pa, pb = self._live_xyz(a), self._live_xyz(b)
            if pa is None or pb is None:
                continue
            if float(np.hypot(pa[0] - pb[0], pa[1] - pb[1])) < d:
                return False
        return True

    def _referents_ok(self, names) -> bool:
        ns = set(names)
        for r in self._referents:
            kind = r["kind"]
            if kind == "on":
                continue
            a, b, rival = r["obj"], r["of"], r.get("rival")
            if not (ns & {a, b, rival}):
                continue
            pa, pb = self._live_xyz(a), self._live_xyz(b)
            if pa is None or pb is None:
                continue
            m = float(r.get("margin", 0.0))
            if kind == "left_of":
                if float(pb[1] - pa[1]) < m:
                    return False
            elif kind == "nearest":
                pr = self._live_xyz(rival)
                if pr is None:
                    continue
                near = float(np.hypot(pa[0] - pb[0], pa[1] - pb[1]))
                far = float(np.hypot(pr[0] - pb[0], pr[1] - pb[1]))
                if near + m > far:
                    return False
        return True

    def _graspable(self, names, considered) -> bool:
        if not self._grasp_clearance:
            return True
        others = [self._obj_geoms[o] for o in considered if o not in names and o in self._obj_geoms]
        return not any(self._geoms_overlap(self._obj_geoms[n], gs, self._grasp_clearance)
                       for n in names if n in self._obj_geoms for gs in others)

    def _handle_path_reachable(self, names) -> bool:
        if not self._articulated_reach & set(names):
            return True
        if not all(self._path_at_base.get(n, True) for n in names
                   if n in self._articulated_reach):
            return True
        want = self._spec_names("joint")
        qsave = self.data.qpos.copy()
        try:
            for n in names:
                if n not in self._articulated_reach:
                    continue
                for j, moving in self._joints_of(n):
                    if self.model.joint(j).name not in want or not len(moving):
                        continue
                    lo, hi = self.model.jnt_range[j]
                    for q in np.linspace(lo, hi, self.STROKE_SAMPLES):
                        self.data.qpos[:] = qsave
                        self.data.qpos[self.model.jnt_qposadr[j]] = q
                        mujoco.mj_kinematics(self.model, self.data)
                        pts = np.array([self.data.geom_xpos[g] for g in moving])
                        if not len(pts):
                            continue
                        h = pts[np.argmin(np.linalg.norm(pts[:, :2] - self._base_xy, axis=1))]
                        if not self.reachable(h):
                            return False
        finally:
            self.data.qpos[:] = qsave
            mujoco.mj_kinematics(self.model, self.data)
        return True

    def _off_support(self, names) -> bool:
        if self._support_rect is None:
            return False
        x0, x1, y0, y1 = self._support_rect
        for n in names:
            for g in self._obj_geoms.get(n, ()):
                c, rot, h = self._obb(int(g))
                pts = c + (CORNER_SIGNS * h) @ rot.T
                if (pts[:, 0].min() < x0 or pts[:, 0].max() > x1
                        or pts[:, 1].min() < y0 or pts[:, 1].max() > y1):
                    return True
        return False

    def _geoms_overlap(self, ga, gb, pad: float = 0.0) -> bool:
        m, d = self.model, self.data
        ca, cb = d.geom_xpos[ga], d.geom_xpos[gb]
        ra, rb = m.geom_rbound[ga][:, None] + pad, m.geom_rbound[gb][:, None]
        a_lo, a_hi = (ca - ra).min(0), (ca + ra).max(0)
        b_lo, b_hi = (cb - rb).min(0), (cb + rb).max(0)
        if np.any(a_hi < b_lo) or np.any(b_hi < a_lo):
            return False
        reach = ra + rb.T
        i, j = np.nonzero(np.linalg.norm(ca[:, None, :] - cb[None, :, :], axis=2) < reach)
        for k in range(len(i)):
            if _obb_intersect(self._obb(int(ga[i[k]]), pad), self._obb(int(gb[j[k]]))):
                return True
        return False

    def _obb(self, g, pad: float = 0.0):
        rot = self.data.geom_xmat[g].reshape(3, 3)
        return (self.data.geom_xpos[g] + rot @ self.model.geom_aabb[g][:3],
                rot, self.model.geom_aabb[g][3:] + pad)

    def _all_overlaps(self):
        return {frozenset((n, k)) for n in self._free_objects if n in self._obj_geoms
                for _, k in self._overlapping([n], self._others_ctx([n]))}

    def reach_grid(self, arm=None):
        arm = arm or self.arms[0]
        cache = self._reach_grid
        if arm not in cache:
            model = self.spec["robots"][self.arms.index(arm)]["model"]
            f = ASSETS / "robots" / model / "reach.npz"
            if not f.is_file():
                cache[arm] = None
            else:
                z = np.load(f)
                shape = tuple(int(v) for v in z["shape"])
                grid = np.unpackbits(z["grid"])[: int(np.prod(shape))].reshape(shape)
                grid = grid.astype(bool)
                step = float(z["step"])
                if REACH_MARGIN_FRACTION > 0 and grid.any():
                    ii, jj = np.nonzero(grid.any(axis=2))
                    ctr = np.array(grid.shape[:2]) / 2.0
                    r_max = np.hypot(ii - ctr[0], jj - ctr[1]).max() * step
                    k = int(round(REACH_MARGIN_FRACTION * r_max / step))
                    if k:
                        d = np.arange(-k, k + 1)
                        disc = (d[:, None] ** 2 + d[None, :] ** 2) <= k * k
                        for iz in range(grid.shape[2]):
                            grid[:, :, iz] = binary_erosion(grid[:, :, iz], disc)
                cache[arm] = (grid, z["origin"].astype(float), step)
        return cache[arm]

    def reachable(self, xyz, arm=None) -> bool:
        arm = arm or self.arms[0]
        g = self.reach_grid(arm)
        if g is None:
            return True
        grid, origin, step = g
        rb = self.spec["robots"][self.arms.index(arm)]
        base = np.asarray(rb["base_pos"], float)
        q = rb["base_quat"]
        local = R.from_quat([q[1], q[2], q[3], q[0]]).inv().apply(np.asarray(xyz, float) - base)
        idx = np.round((local - origin) / step).astype(int)
        if np.any(idx < 0) or np.any(idx >= np.array(grid.shape)):
            return False
        return bool(grid[tuple(idx)])


def systematic_extremes(env, bodies=None):
    want = set(bodies) if bodies else None
    out = []
    for body, regions in env._placement_regions.items():
        if want and body not in want:
            continue
        for ri, reg in enumerate(regions):
            rots = []
            for wlo, whi in _yaw_bands(reg):
                rots += [wlo] if abs(whi - wlo) < 1e-9 else [wlo, whi]
            cx, cy, _ = env.clipped_station(body, reg)
            xs = cx if cx else [None]
            for x in xs:
                for y in cy:
                    for w in rots:
                        out.append({"body": body, "station": ri,
                                    "x": None if x is None else float(x),
                                    "y": float(y), "yaw_offset_deg": float(w)})
    movable = set(env._free_objects) | set(env._static_objects)
    for body in (want or ()):
        if body in env._placement_regions or body not in movable:
            continue
        cells = env.legal_cells([body])
        if not len(cells):
            continue
        c = np.asarray(cells, float)
        corners = [(c[:, 0].min(), c[:, 1].min()), (c[:, 0].min(), c[:, 1].max()),
                   (c[:, 0].max(), c[:, 1].min()), (c[:, 0].max(), c[:, 1].max())]
        yr = float(env.init_region["yaw"][1])
        rots = [0.0] if body in env._face_robot else [-np.degrees(yr), np.degrees(yr)]
        for (x, y) in corners:
            for w in rots:
                out.append({"body": body, "station": None, "x": float(x),
                            "y": float(y), "yaw_offset_deg": float(w)})
    return out


def place_at_extreme(env, k: int, bodies=None):
    ex = systematic_extremes(env, bodies)
    if not ex:
        return None
    choice = ex[k % len(ex)]
    bases = dict(env.init_bases)
    names = [choice["body"]]
    pivot = np.mean([bases[n][0][:2] for n in names], axis=0)
    dx = float(choice["x"]) - pivot[0] if choice.get("x") is not None else 0.0
    dy = float(choice["y"]) - pivot[1]
    yaw = (env._face_yaw(names, pivot, dx, dy) or 0.0) \
        + np.radians(float(choice["yaw_offset_deg"]))
    pinned = {n: env._group_pose(bases, names, dx, dy, yaw)[n] for n in names}
    for attempt in range(env.LAYOUT_TRIES):
        try:
            env._place_objects(True, pinned=pinned)
            break
        except RuntimeError:
            if attempt == env.LAYOUT_TRIES - 1:
                logger.warning(f"[systematic] extreme {k % len(ex)} of {choice['body']} "
                               f"leaves no room for the rest of the scene")
                env._place_objects(False)
                env._write_objects(pinned)
                return {**choice, "admissible": False}
    env.forward()
    if not (env.can_open(names) and env._handle_path_reachable(names)
            and env._within_reach(names)):
        logger.warning(f"[systematic] extreme {k % len(ex)} of {choice['body']} "
                       f"(y={choice['y']:+.3f}, yaw_off={choice['yaw_offset_deg']:+.0f}) "
                       f"is NOT admissible: the declared band reaches past what the "
                       f"scene allows")
        return {**choice, "admissible": False}
    return {**choice, "admissible": True}


def init_regions(env):
    rnd = env.spec["randomization"]
    r = env.init_region
    rng = {"x": r["x"][1], "y": r["y"][1], "yaw": r["yaw"][1]}
    out, gi = {}, 0
    for g in (rnd.get("groups") or []):
        for n in g:
            out[n] = {**rng, "group": gi}
        gi += 1
    for n in env._free_objects:
        if n not in out:
            out[n] = {**rng, "group": gi}
            gi += 1
    names = env._free_objects + [n for n in env._static_objects if n in out]
    for i, n in enumerate(names, 1):
        if n in out:
            out[n]["num"] = i
    return {n: out[n] for n in names if n in out}
