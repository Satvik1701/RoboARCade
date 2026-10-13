"""Shell game: deals the colours and moves the shells.

Shell game motion: the reset RNG deals which colour starts in which slot and which slot pair trades on each
shuffle press; the shells and cubes then move along scripted legs, each opened by a press of a button.
task.json's `scene_script_gen` declares the game (cubes, shells, slot row, detour rows, buttons, press count)."""
import numpy as np

from core.grade_episode import PressEdge


class Dynamics:
    def __init__(self, env):
        self.cfg = env.spec["scene_script_gen"]
        self.build(env, [])

    def deal(self, env):
        cfg, rng, m = self.cfg, env._rng, env.model
        cubes, shells = list(cfg["cubes"]), list(cfg["shells"])
        slot_y = [float(y) for y in cfg["slot_y"]]
        n = len(slot_y)
        assert len(cubes) == len(shells) == n
        cover_x, start_x = float(cfg["cover_x"]), float(cfg["start_x"])
        front_x, back_x = float(cfg["front_x"]), float(cfg["back_x"])
        cube_z, shell_z, carry_z = (float(cfg[k]) for k in ("cube_z", "shell_z", "carry_z"))
        cover_btn, shuffle_btn = cfg["cover_btn"], cfg["shuffle_btn"]

        perm = rng.permutation(n)
        cube_at = {}
        for i, name in enumerate(cubes):
            slot = int(perm[i])
            cube_at[slot] = name
            a = m.joint(m.body(name).jntadr[0]).qposadr[0]
            env.data.qpos[a:a + 3] = [cover_x, slot_y[slot], cube_z]
            env.data.qpos[a + 3:a + 7] = [1.0, 0.0, 0.0, 0.0]
        shell_at = {s: shells[s] for s in range(n)}

        segs = []
        def move(body, press, t0, t1, x, y, z, btn):
            segs.append({"body": body, "after": [btn], "press": press,
                         "from": round(t0, 2), "to": round(t1, 2),
                         "pos": [round(x, 4), round(y, 4), z]})

        for slot in range(n):
            sh = shell_at[slot]
            move(sh, 1, 0.3, 0.7, start_x, slot_y[slot], carry_z, cover_btn)
            move(sh, 1, 0.7, 1.1, cover_x, slot_y[slot], carry_z, cover_btn)
            move(sh, 1, 1.1, 1.5, cover_x, slot_y[slot], shell_z, cover_btn)

        for k in range(1, int(cfg["presses"]) + 1):
            a, b = (int(v) for v in rng.choice(n, size=2, replace=False))
            for src, dst, row in ((a, b, front_x), (b, a, back_x)):
                for body, z in ((shell_at[src], shell_z), (cube_at[src], cube_z)):
                    move(body, k, 0.2, 0.6, row, slot_y[src], z, shuffle_btn)
                    move(body, k, 0.6, 1.2, row, slot_y[dst], z, shuffle_btn)
                    move(body, k, 1.2, 1.6, cover_x, slot_y[dst], z, shuffle_btn)
            cube_at[a], cube_at[b] = cube_at[b], cube_at[a]
            shell_at[a], shell_at[b] = shell_at[b], shell_at[a]
        self.build(env, segs)
        env.forward()

    def build(self, env, segs):
        self.script = {}
        for seg in sorted(segs, key=lambda s: (len(s.get("after") or ()), s.get("press", 1), s["from"])):
            seg = dict(seg)
            after = seg.get("after")
            seg["after"] = [after] if isinstance(after, str) else list(after or ())
            self.script.setdefault(seg["body"], []).append(seg)
        m = env.model
        self.adr = {b: (m.joint(m.body(b).jntadr[0]).qposadr[0], m.joint(m.body(b).jntadr[0]).dofadr[0])
                    for b in self.script}
        self.edges = [PressEdge(j) for j in dict.fromkeys(
            t for segs_ in self.script.values() for seg in segs_ for t in seg["after"])]
        self.pressed = {}

    def physics_step(self, env):
        now = float(env.data.time)
        for e in self.edges:
            if e.update(env):
                self.pressed.setdefault(e.joint, []).append(now)
            ts = self.pressed.get(e.joint, [])
            kept = [t for t in ts if t <= now]
            if len(kept) != len(ts):
                self.pressed[e.joint], e.down = kept, False

        for body, segs in self.script.items():
            active = t0 = None
            best = (-1, -1.0)
            for seg in segs:
                epoch = self.epoch(env, seg)
                if epoch is None:
                    continue
                t_open = epoch + seg["from"]
                rank = (len(seg["after"]), t_open)
                if now >= t_open and rank >= best:
                    active, t0, best = seg, epoch, rank
            if active is None:
                continue
            if active is segs[-1] and now > t0 + active["to"]:
                continue
            a, dof = self.adr[body]
            start = active.get("_start")
            if start is None:
                start = active["_start"] = env.data.qpos[a:a + 3].copy()
            goal = start if active["pos"] is None else np.asarray(active["pos"], float)
            f = float(np.clip((now - t0 - active["from"]) / max(active["to"] - active["from"], 1e-9), 0.0, 1.0))
            env.data.qpos[a:a + 3] = start + (goal - start) * f
            env.data.qvel[dof:dof + 6] = 0.0

    def epoch(self, env, seg):
        if not seg.get("after"):
            return env.episode_t0
        n = int(seg.get("press", 1))
        times = []
        for j in seg["after"]:
            ts = self.pressed.get(j, ())
            if len(ts) < n:
                return None
            times.append(ts[n - 1])
        return max(times)
