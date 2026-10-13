"""The conveyor belts of pack tools into toolboxes.

The pack tools conveyor: belts that carry tools and toolboxes in, release a dealt queue, and either wrap riders
back to the start or (`exit: remove`) carry a toolbox off stage with its contents. task.json's `conveyor` declares
the belt body, axis, speed, start, end, spawn queue and exit mode."""
import mujoco
import numpy as np

from core.grade_episode import _bbox_world, _footprint_hull, _hull_contains_xy

OFF_STAGE_SHIFT_M = 10.0


class Dynamics:
    def __init__(self, env):
        self.cfg = env.spec["conveyor"]
        self.spawn_schedule, self.exit_poses, self.off_stage = (), {}, {}
        self.contype, self.conaffinity = env.model.geom_contype.copy(), env.model.geom_conaffinity.copy()

    def restore(self, env):
        axis = np.asarray(self.cfg["axis"], float)
        axis /= np.linalg.norm(axis)
        beyond = float(self.cfg["end"]) + OFF_STAGE_SHIFT_M / 2
        self.off_stage = {}
        for name in env._free_objects:
            a = env._obj_qadr[name]
            if float(env.data.qpos[a:a + 3] @ axis) > beyond:
                self.off_stage[name] = env.data.qpos[a:a + 7].copy()
        self.sync_contacts(env)

    def sync_contacts(self, env):
        m = env.model
        for name in env._free_objects:
            g = env._subtree_geoms(name)
            on = name not in self.off_stage
            m.geom_contype[g] = self.contype[g] * on
            m.geom_conaffinity[g] = self.conaffinity[g] * on

    def deal(self, env):
        cfg = self.cfg.get("spawn") if self.cfg else None
        self.off_stage = {}
        if cfg is None:
            self.spawn_schedule = ()
            self.exit_poses = {}
            return
        axis = np.asarray(self.cfg["axis"], float)
        axis /= np.linalg.norm(axis) or 1.0
        initial = cfg.get("initial")
        queue = cfg["queue"]
        jitter, case_yaws_deg = cfg.get("jitter"), cfg.get("box_yaw_deg")
        lateral = np.cross([0.0, 0.0, 1.0], axis)
        tools = set(initial["shuffle"] if initial is not None else ()) | set(queue["poses"])
        cases = set(initial["phase"]["items"] if initial is not None else ()) | set(cfg["items"])

        def dealt(pose, body):
            pose = np.array(pose, float)
            if jitter and body in tools:
                pose[:3] += lateral * env._rng.uniform(-jitter["lateral"], jitter["lateral"])
                yaw = np.radians(env._rng.uniform(-jitter["yaw_deg"], jitter["yaw_deg"]))
            elif case_yaws_deg and body in cases:
                yaw = np.radians(float(env._rng.choice(case_yaws_deg)))
            else:
                return pose
            turn = np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])
            mujoco.mju_mulQuat(pose[3:7], turn, pose[3:7].copy())
            return pose
        if self.cfg.get("exit") == "remove":
            removable = list(initial["shuffle"] if initial is not None else ())
            removable += list(queue["poses"])
            self.exit_poses = {
                body: env.data.qpos[
                    env._obj_qadr[body]:env._obj_qadr[body] + 7].copy()
                for body in removable
            }
        else:
            self.exit_poses = {}
        initial_schedule = []
        if initial is not None:
            bodies = list(initial["shuffle"])
            if self.cfg.get("exit") == "remove":
                for body in bodies:
                    a = env._obj_qadr[body]
                    env.data.qpos[a:a + 7] = env.model.qpos0[a:a + 7]
            stations = [float(
                env.data.qpos[env._obj_qadr[b]:env._obj_qadr[b] + 3] @ axis)
                for b in bodies]
            order = [bodies[int(i)] for i in env._rng.permutation(len(bodies))]
            phase = float(env._rng.uniform(*initial["phase"]["range"]))
            pitched = "spacing" in initial and bool(self.exit_poses)
            if pitched:
                leader = order[0]
                pos = env.data.qpos[
                    env._obj_qadr[leader]:env._obj_qadr[leader] + 3]
                pos += axis * (max(stations) - float(pos @ axis))
                for body in order[1:]:
                    a = env._obj_qadr[body]
                    env.data.qpos[a:a + 7] = self.exit_poses[body]
                phase_items = [
                    body for body in initial["phase"]["items"] if body not in bodies]
                phase_items.append(leader)
            else:
                for body, station in zip(order, stations):
                    a = env._obj_qadr[body]
                    pos = env.data.qpos[a:a + 3]
                    pos += axis * (station - float(pos @ axis))
                    env.data.qpos[a:a + 7] = dealt(env.data.qpos[a:a + 7], body)
                phase_items = initial["phase"]["items"]
            for body in phase_items:
                a, dof = env._obj_qadr[body], env._obj_dofadr[body]
                env.data.qpos[a:a + 3] += axis * phase
                env.data.qpos[a:a + 7] = dealt(env.data.qpos[a:a + 7], body)
                env.data.qvel[dof:dof + 6] = 0.0
            env.forward()
            if pitched:
                spacing = float(initial["spacing"])
                lead = float(env.data.qpos[
                    env._obj_qadr[leader]:env._obj_qadr[leader] + 3] @ axis)
                first = spacing - (lead - float(self.cfg["start"]))
                if first < 0:
                    raise ValueError("initial conveyor leader is more than one pitch past start")
                initial_schedule = [
                    (first + (i - 1) * spacing, body,
                     dealt(env.model.qpos0[env._obj_qadr[body]:
                                            env._obj_qadr[body] + 7], body))
                    for i, body in enumerate(order[1:], 1)
                ]
        after = float(cfg["after"])
        schedule = initial_schedule
        schedule += [(after, body, dealt(pose, body)) for body, pose in cfg["items"].items()]
        order = list(queue["poses"])
        if queue["shuffle"]:
            order = [order[int(i)] for i in env._rng.permutation(len(order))]
        spacing = float(queue["spacing"])
        schedule += [(float(queue["after"]) + i * spacing, body, dealt(queue["poses"][body], body))
                     for i, body in enumerate(order)]
        self.spawn_schedule = tuple(
            (distance, body, np.asarray(pose, float),
             env.data.qpos[env._obj_qadr[body]:env._obj_qadr[body] + 3].copy())
            for distance, body, pose in sorted(schedule))

    def spawn(self, env, c, axis):
        if not self.spawn_schedule:
            return
        travelled = max(0.0, (float(env.data.time) - env.episode_t0) * float(c["speed"]))
        for distance, body, authored, parked in self.spawn_schedule:
            a, dof = env._obj_qadr[body], env._obj_dofadr[body]
            if body in self.exit_poses \
                    and travelled >= distance + float(c["end"]) - float(c["start"]):
                continue
            if distance > travelled or np.linalg.norm(
                    env.data.qpos[a:a + 3] - parked) > 0.02:
                continue
            pose = authored.copy()
            pose[:3] += axis * (float(c["start"]) - float(np.dot(pose[:3], axis)))
            candidate = _bbox_world(env, body) + pose[:3] - env.data.qpos[a:a + 3]
            if self.entry_occupied(env, candidate, {body}):
                break
            env.data.qpos[a:a + 7] = pose
            env.data.qvel[dof:dof + 3] = axis * float(c["speed"])
            env.data.qvel[dof + 3:dof + 6] = 0.0

    def physics_step(self, env):
        c = self.cfg
        axis = np.asarray(c["axis"], float)
        axis = axis / (np.linalg.norm(axis) or 1.0)
        self.spawn(env, c, axis)
        belt = {int(g) for g in env._subtree_geoms(c["body"])}
        m, d, root = env.model, env.data, env.model.body_rootid
        for name, pose in self.off_stage.items():
            a, dof = env._obj_qadr[name], env._obj_dofadr[name]
            d.qpos[a:a + 7] = pose
            d.qvel[dof:dof + 6] = 0.0
        free = {m.body(name).id for name in env._free_objects}
        riding, slab = {}, {}
        held = set()
        for k in range(d.ncon):
            con = d.contact[k]
            g1, g2 = int(con.geom1), int(con.geom2)
            b1, b2 = int(root[m.geom_bodyid[g1]]), int(root[m.geom_bodyid[g2]])
            on1, on2 = g1 in belt, g2 in belt
            if on1 != on2:
                body, belt_geom = (b2, g1) if on1 else (b1, g2)
                if body in free:
                    riding[body] = max(riding.get(body, 0.0),
                                       float(m.geom_friction[belt_geom, 0]))
                    slab[body] = belt_geom
            elif not on1:
                r1 = m.body(b1).name.startswith(env.robot_prefix)
                if r1 != m.body(b2).name.startswith(env.robot_prefix):
                    body = b2 if r1 else b1
                    if body in free:
                        held.add(body)
        for body in held:
            riding.pop(body, None)

        carrier_groups = []
        if self.exit_poses:
            for name in env._free_objects:
                if name in self.exit_poses:
                    continue
                geoms = np.fromiter(env._subtree_geoms(name), dtype=int)
                sphere_edge = np.max(
                    d.geom_xpos[geoms] @ axis + m.geom_rbound[geoms])
                if sphere_edge > c["end"] \
                        and float(np.max(_bbox_world(env, name) @ axis)) > c["end"]:
                    carrier_groups.append(self.carry_group(env, name, held))
        contained = {name for group in carrier_groups for name in group
                     if name in self.exit_poses}
        for name in contained:
            riding.pop(m.body(name).id, None)

        gravity = float(np.linalg.norm(m.opt.gravity))
        for name in env._free_objects:
            body = m.body(name).id
            applied = d.xfrc_applied[body, :3]
            applied -= axis * float(np.dot(applied, axis))
            if body not in riding:
                continue
            dof = env._obj_dofadr[name]
            v = d.qvel[dof:dof + 3]
            along = float(np.dot(v, axis))
            mass = float(m.body_subtreemass[body])
            gain = mass / m.opt.timestep
            limit = riding[body] * mass * gravity
            target = c.get("slab_speeds", {}).get(m.geom(slab[body]).name, c["speed"])
            force = float(np.clip(gain * (float(target) - along), -limit, limit))
            applied += axis * force
        wrapped = set()
        for name in env._free_objects:
            a = env._obj_qadr[name]
            if m.body(name).id not in riding or name in wrapped or name in self.off_stage \
                    or float(np.dot(d.qpos[a:a + 3], axis)) <= c["end"]:
                continue
            if name in self.exit_poses:
                if any(name in group for group in carrier_groups):
                    continue
                dof = env._obj_dofadr[name]
                d.qpos[a:a + 7] = self.exit_poses[name]
                d.qvel[dof:dof + 6] = 0.0
                d.xfrc_applied[m.body(name).id] = 0.0
                wrapped.add(name)
                continue
            group = self.carry_group(env, name, held) - wrapped
            if c.get("exit") == "remove":
                for n in group:
                    qa, dof = env._obj_qadr[n], env._obj_dofadr[n]
                    d.qpos[qa:qa + 3] += axis * OFF_STAGE_SHIFT_M
                    d.qvel[dof:dof + 6] = 0.0
                    d.xfrc_applied[m.body(n).id] = 0.0
                    self.off_stage[n] = d.qpos[qa:qa + 7].copy()
                self.sync_contacts(env)
                wrapped |= group
                continue
            group_box = np.vstack([_bbox_world(env, n) for n in group])
            start_shift = axis * (float(c["start"])
                                  - float(np.dot(d.qpos[a:a + 3], axis)))
            target = c["end"] if self.entry_occupied(env, 
                group_box + start_shift, group) else c["start"]
            shift = axis * (float(target) - float(np.dot(d.qpos[a:a + 3], axis)))
            for n in group:
                qa, dof = env._obj_qadr[n], env._obj_dofadr[n]
                d.qpos[qa:qa + 3] += shift
                d.qvel[dof:dof + 6] = 0.0
            wrapped |= group

    def carry_group(self, env, name, held):
        box = _bbox_world(env, name)
        hull = _footprint_hull(box)
        group = {name}
        m, d, root = env.model, env.data, env.model.body_rootid
        carrier = m.body(name).id
        carrier_geoms = {int(g) for g in env._subtree_geoms(name)}
        belt_geoms = ({int(g) for g in env._subtree_geoms(self.cfg["body"])}
                      if self.cfg else set())
        free = {m.body(other).id for other in env._free_objects}
        supported_elsewhere = set()
        for contact in d.contact[:d.ncon]:
            g1, g2 = int(contact.geom1), int(contact.geom2)
            b1 = int(root[m.geom_bodyid[g1]])
            b2 = int(root[m.geom_bodyid[g2]])
            for candidate, external in ((b1, g2), (b2, g1)):
                if candidate == carrier or candidate not in free:
                    continue
                external_root = int(root[m.geom_bodyid[external]])
                if external in carrier_geoms or external in belt_geoms \
                        or external_root in free \
                        or m.body(external_root).name.startswith(env.robot_prefix):
                    continue
                supported_elsewhere.add(candidate)
        for other in env._free_objects:
            if other == name or other not in self.exit_poses \
                    or m.body(other).id in held:
                continue
            p = d.xpos[m.body(other).id]
            if m.body(other).id not in supported_elsewhere \
                    and _hull_contains_xy(hull, p[:2]) \
                    and box[:, 2].min() < p[2] < box[:, 2].max():
                group.add(other)
        return group

    def entry_occupied(self, env, candidate, excluded):
        gmin, gmax = np.min(candidate, axis=0), np.max(candidate, axis=0)
        for other in env._free_objects:
            if other in excluded:
                continue
            box = _bbox_world(env, other)
            omin, omax = np.min(box, axis=0), np.max(box, axis=0)
            if np.all(gmax >= omin) and np.all(omax >= gmin):
                return True
        return False

    def over(self, env) -> bool:
        c = self.cfg
        spawn = c.get("spawn") if c else None
        if not spawn or c.get("exit") != "remove":
            return False
        initial = spawn.get("initial") or {}
        cases = set((initial.get("phase") or {}).get("items", ())) | set(spawn.get("items", {}))
        return bool(cases) and cases <= set(self.off_stage)
