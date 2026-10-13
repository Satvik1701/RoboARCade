"""Grades an episode with its task's grade.py, and the check language grade.py uses.

Grading runs the task's subtask checks over the episode's saved simulator states in order. The episode is solved when
every subtask is done.

tasks/<task>/grade.py defines subtasks(), a fresh list of Subtask per episode, and may define INVARIANT. Each check is a
function of the Scene at one control step. The last subtask is the task's own success (s.success(): task.json's success
tree held for half a second). The rule, the same for every task:

  - a subtask is done if its check holds at the last step; an `ever` subtask (one the task undoes on purpose) is done
    once its check held after its `needs` were done;
  - `end_needs`: a subtask also needs these subtasks done at the end;
  - INVARIANT: the first step it fails ends grading there: sequential subtasks are judged at the last step before
    it, `ever` ones by what they earned before it, and `end_needs` still apply;
  - an episode that ran out of its step budget does not get its last subtask.

The score is the share of subtasks done. An episode is graded once, right after it runs;
`python -m core.grade_episode <episode>` regrades its saved states against the current task and prints (or --write
writes) the grade.

The check language: Scene has one call per predicate leaf (s.in_site_box(obj=..., site=...),
s.robot_clear(bodies=[...]), s.joint_in(joint=..., lo=..., hi=...)), compiled once per episode and evaluated on the
current simulator state. task.json's success tree uses the same leaves. A leaf only one task needs is a function
in that task's grade.py named as the leaf, taking the leaf's fields and returning check(env); its docstring, with
the fields filled in, is the sentence the Tasks page shows for it.
"""
import json
import types
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial import ConvexHull, QhullError

from core.tasks import ASSETS, task_module

PRESS_DEPTH_FRACTION = 0.25


def _body_pos(env, name):
    return env.data.body(name).xpos


def _free_joint_qvel(env, body_name):
    jadr = env.model.body(body_name).jntadr[0]
    vadr = env.model.joint(jadr).dofadr[0]
    return env.data.qvel[vadr : vadr + 6]


def _contact_body_pairs(env):
    return env.contact_body_pairs()


def _robot_touching(env, body_names):
    bids = {env.model.body(n).id for n in body_names}
    for b1, b2 in _contact_body_pairs(env):
        for bid, other in ((b1, b2), (b2, b1)):
            if other in bids and env.model.body(bid).name.startswith(env.robot_prefix):
                return True
    return False


def _touching(env, a, b):
    root = env.model.body_rootid
    ids = {root[env.model.body(a).id], root[env.model.body(b).id]}
    return any({root[b1], root[b2]} == ids for b1, b2 in _contact_body_pairs(env))


def _grasped(env, obj, prefix=None):
    prefix = prefix or env.robot_prefix
    oid = env.model.body(obj).id
    return any(oid in env.held_bodies(arm) for arm in env.arms if (arm + "_").startswith(prefix))


def _grasped_by(env, obj, arm):
    return _grasped(env, obj, arm + "_")


def _released_at_rest(env, obj):
    if not env.robot_clear([obj]):
        return False
    qv = _free_joint_qvel(env, obj)
    return bool(np.linalg.norm(qv[:3]) <= 0.01 and np.linalg.norm(qv[3:]) <= 0.1)


_CORNER_SIGNS = np.array(np.meshgrid(*([-1.0, 1.0],) * 3)).reshape(3, -1).T


def _bbox_world(env, name):
    m, d = env.model, env.data
    geoms = env._subtree_geoms(name)
    if not len(geoms):
        return None
    c, h = m.geom_aabb[geoms, :3], m.geom_aabb[geoms, 3:]
    local = c[:, None, :] + _CORNER_SIGNS[None, :, :] * h[:, None, :]
    rot = d.geom_xmat[geoms].reshape(-1, 3, 3)
    return (np.einsum("nij,nkj->nki", rot, local) + d.geom_xpos[geoms][:, None, :]).reshape(-1, 3)


def _footprint_hull(pts):
    xy = np.asarray(pts, float)[:, :2]
    if len(xy) < 3:
        return None
    try:
        return xy[ConvexHull(xy).vertices]
    except QhullError:
        return None


def _hull_contains_xy(hull, q) -> bool:
    if hull is None:
        return False
    a, b = hull, np.roll(hull, -1, axis=0)
    return bool(np.all((b[:, 0] - a[:, 0]) * (q[1] - a[:, 1])
                       - (b[:, 1] - a[:, 1]) * (q[0] - a[:, 0]) >= -1e-12))


def _subtrees_touching(env, a, b):
    m, d = env.model, env.data
    for ga in env._subtree_geoms(a):
        for gb in env._subtree_geoms(b):
            can_collide = ((m.geom_contype[ga] & m.geom_conaffinity[gb])
                           or (m.geom_contype[gb] & m.geom_conaffinity[ga]))
            if can_collide and mujoco.mj_geomDistance(m, d, ga, gb, 1e-5, None) <= 0:
                return True
    return False


def _in_site_geom(env, obj, site, size) -> bool:
    s = env.data.site(site)
    R = s.xmat.reshape(3, 3)
    p = R.T @ (_body_pos(env, obj) - s.xpos)
    lo, hi = -np.asarray(size, float), np.asarray(size, float).copy()
    down_axis = int(np.argmax(np.abs(R[2])))
    if R[2, down_axis] > 0:
        lo[down_axis] -= 0.01
    else:
        hi[down_axis] += 0.01
    return bool(np.all(p > lo) and np.all(p < hi))


def _build(spec, task=None):
    pred = spec["pred"]

    if pred in ("all", "any", "count"):
        checks = [_build(s, task) for s in spec["of"]]
        return _Combine(checks, len(checks) if pred == "all" else
                        1 if pred == "any" else int(spec["min"]))
    if pred == "not":
        return _Combine([_build(spec["of"], task)], 1, negate=True)
    if pred == "sequence":
        inv = spec.get("invariant")
        holds = [s.get("hold", 1) for s in spec["of"]]
        if any(not isinstance(n, int) or isinstance(n, bool) or n < 1 for n in holds):
            raise ValueError("sequence stage hold must be a positive integer")
        return _Sequence([_build(s, task) for s in spec["of"]], holds,
                         _build(inv, task) if inv else None)
    if pred == "lift_above":
        obj, z_above = spec["obj"], spec["z_above"]
        return lambda env: bool(_body_pos(env, obj)[2] > z_above)

    if pred == "z_below":
        obj, z_below = spec["obj"], spec["z_below"]
        return lambda env: bool(_body_pos(env, obj)[2] < z_below)

    if pred == "above":
        obj, base = spec["obj"], spec["base"]
        return lambda env: bool(_body_pos(env, obj)[2] >= _body_pos(env, base)[2])

    if pred == "joint_in":
        joint, lo, hi = spec["joint"], spec["lo"], spec["hi"]
        def joint_in(env):
            j = env.model.joint(joint)
            q = env.data.qpos[j.qposadr[0]]
            if env.model.jnt_limited[j.id]:
                q = np.clip(q, *env.model.jnt_range[j.id])
            return bool(lo <= q <= hi)
        return joint_in

    if pred == "in_site_box":
        obj, site, size = spec["obj"], spec["site"], np.asarray(spec["size"])
        def in_site_box(env):
            return _in_site_geom(env, obj, site, size) and _released_at_rest(env, obj)
        return in_site_box

    if pred == "over_site":
        obj, site = spec["obj"], spec["site"]
        lo, hi = np.asarray(spec["lo"]), np.asarray(spec["hi"])
        def over_site(env):
            s = env.data.site(site)
            p = s.xmat.reshape(3, 3).T @ (_body_pos(env, obj) - s.xpos)
            return bool(np.all(p > lo) and np.all(p < hi))
        return over_site

    if pred == "on_site":
        obj, site, size = spec["obj"], spec["site"], np.asarray(spec["size"])
        parent = spec.get("parent")
        def on_site(env):
            s = env.data.site(site)
            delta = s.xmat.reshape(3, 3).T @ (_body_pos(env, obj) - s.xpos)
            geom_ok = (size[2] - 0.005 < delta[2] < size[2] + 0.10
                       and np.all(np.abs(delta[:2]) < size[:2]))
            if not geom_ok or (parent and not _touching(env, parent, obj)):
                return False
            return _released_at_rest(env, obj)
        return on_site

    if pred == "within_dist":
        a, b, tol = spec["a"], spec["b"], spec["tol"]
        dims = 2 if spec.get("axes", "xyz") == "xy" else 3
        norm = (lambda d: np.abs(d).max()) if spec.get("per_axis") else np.linalg.norm
        return lambda env: bool(norm(_body_pos(env, a)[:dims] - _body_pos(env, b)[:dims]) < tol)

    if pred == "touching":
        a, b = spec["a"], spec["b"]
        return lambda env: _touching(env, a, b)

    if pred == "touching_part":
        a, b = spec["a"], spec["b"]
        def touching_part(env):
            m = env.model
            def under(bid, top):
                while bid > 0:
                    if bid == top:
                        return True
                    bid = int(m.body_parentid[bid])
                return False
            ta, tb = m.body(a).id, m.body(b).id
            for b1, b2 in _contact_body_pairs(env):
                for x, y in ((b1, b2), (b2, b1)):
                    if under(x, ta) and under(y, tb):
                        return True
            return _subtrees_touching(env, a, b)
        return touching_part

    if pred == "grasped":
        obj = spec["obj"]
        return lambda env: _grasped(env, obj)

    if pred == "axis_dot":
        obj = spec["obj"]
        local = np.asarray(spec.get("axis", [0.0, 0.0, 1.0]), float)
        world = np.asarray(spec.get("world_axis", [0.0, 0.0, 1.0]), float)
        world = world / np.linalg.norm(world)
        min_dot = float(spec["min_dot"])
        max_dot = spec.get("max_dot")
        take_abs = bool(spec.get("abs"))
        def axis_dot(env):
            v = env.data.body(obj).xmat.reshape(3, 3) @ local
            dot = float(v @ world) / np.linalg.norm(v)
            if take_abs:
                dot = abs(dot)
            return bool(dot >= min_dot and (max_dot is None or dot <= max_dot))
        return axis_dot

    if pred == "press_count":
        return _PressCount(spec)

    if pred == "pose_within":
        obj, pos_tol = spec["obj"], spec["pos_tol"]
        pos_tol_xy = spec.get("pos_tol_xy")
        goal_p = np.asarray(spec["pos"])
        goal_quat_wxyz = np.asarray(spec["quat"])
        rel_to = spec.get("rel_to")
        axis = np.asarray(spec["axis"], float) if spec.get("axis") else None
        cos_tol = np.cos(np.radians(spec["rot_tol_deg"]) / (1 if axis is not None else 2))
        sym = max(1, int(spec.get("sym", 1)))
        sym_q = []
        for k in range(sym):
            q = np.empty(4)
            mujoco.mju_axisAngle2Quat(q, np.array([0.0, 0.0, 1.0]), 2 * np.pi * k / sym)
            sym_q.append(q)
        def pose_within(env):
            p, q = goal_p, goal_quat_wxyz
            if rel_to is not None:
                ref = env.data.body(rel_to)
                p = ref.xpos + ref.xmat.reshape(3, 3) @ goal_p
                q = np.empty(4)
                mujoco.mju_mulQuat(q, ref.xquat, goal_quat_wxyz)
            error = _body_pos(env, obj) - p
            if np.linalg.norm(error) > pos_tol:
                return False
            if pos_tol_xy is not None:
                error_in_ref_frame = ref.xmat.reshape(3, 3).T @ error if rel_to is not None else error
                if np.linalg.norm(error_in_ref_frame[:2]) > pos_tol_xy:
                    return False
            if axis is not None:
                goal_m = np.empty(9)
                mujoco.mju_quat2Mat(goal_m, q)
                actual = env.data.body(obj).xmat.reshape(3, 3) @ axis
                return bool(np.dot(actual, goal_m.reshape(3, 3) @ axis) >= cos_tol)
            xquat = env.data.body(obj).xquat
            for turn in sym_q:
                cand = np.empty(4)
                mujoco.mju_mulQuat(cand, q, turn)
                if abs(np.dot(xquat, cand)) >= cos_tol:
                    return True
            return False
        return pose_within

    if pred == "eef_near_start":
        pos_tol = spec["pos_tol"]
        cos_half_rot_tol = np.cos(np.radians(spec["rot_tol_deg"]) / 2)
        def eef_pose(env):
            site = env._site[env.arms[0]]
            quat = np.empty(4)
            mujoco.mju_mat2Quat(quat, env.data.site_xmat[site])
            return env.data.site_xpos[site].copy(), quat
        def near_start(env, start):
            pos, quat = eef_pose(env)
            return bool(np.all(np.abs(pos - start[0]) <= pos_tol)
                        and abs(np.dot(quat, start[1])) >= cos_half_rot_tol)
        return _Baseline(eef_pose, near_start)

    if pred == "in_bbox":
        a, b = spec["a"], spec["b"]
        def in_bbox(env):
            box = _bbox_world(env, b)
            p = _body_pos(env, a)
            return bool(box is not None and _hull_contains_xy(_footprint_hull(box), p[:2])
                        and box[:, 2].min() < p[2])
        return in_bbox

    if pred == "depth_in":
        a, b, z_threshold = spec["a"], spec["b"], spec["z_threshold"]
        def depth_in(env):
            ba, bb = _bbox_world(env, a), _bbox_world(env, b)
            return bool(ba is not None and bb is not None
                        and bb[:, 2].max() - ba[:, 2].min() > z_threshold)
        return depth_in

    if pred == "axis_up":
        return _build({"pred": "axis_dot", "obj": spec["obj"], "axis": spec["axis"],
                       "min_dot": float(np.cos(np.radians(spec["tol_deg"])))})

    if pred == "robot_clear":
        bodies = spec["bodies"]
        return lambda env: env.robot_clear(bodies)

    if pred == "gripper_open":
        lo = spec["min"]
        return lambda env: all(
            env.gripper_width(n) / env.gripper_max_width >= lo for n in env.arms)

    leaf = getattr(task_module(task, "grade"), pred, None) if task else None
    if leaf is None:
        raise ValueError(f"unknown success pred {pred!r}")
    return leaf(**{k: v for k, v in spec.items() if k != "pred"})


class _Combine:
    def __init__(self, checks, least, negate=False):
        self.checks, self.least, self.negate = checks, least, negate

    def __call__(self, env):
        return (sum(bool(c(env)) for c in self.checks) >= self.least) != self.negate


class _Baseline:
    def __init__(self, capture, check):
        self.capture, self.check, self.base = capture, check, None

    def __call__(self, env):
        if self.base is None:
            self.base = self.capture(env)
        return self.check(env, self.base)


class _Sequence:
    def __init__(self, stages, holds, invariant=None):
        self.stages, self.holds = stages, holds
        self.invariant, self.at, self.streak, self.dead = invariant, 0, 0, False
        self.broken = 0

    def __call__(self, env):
        broken = self.invariant is not None and not self.invariant(env)
        self.broken = self.broken + 1 if broken else 0
        if self.broken >= INVARIANT_KILL_STEPS:
            self.dead = True
        holds = [s(env) for s in self.stages]
        if self.dead or broken:
            return False
        if self.at < len(self.stages):
            if holds[self.at]:
                self.streak += 1
                if self.streak >= self.holds[self.at]:
                    self.at += 1
                    self.streak = 0
            else:
                self.streak = 0
        return self.at == len(self.stages) and bool(holds[-1])


INVARIANT_KILL_STEPS = 5

class PressEdge:
    def __init__(self, joint):
        self.joint, self.down = joint, False

    def update(self, env) -> bool:
        j = env.model.joint(self.joint)
        press_q = float(env.model.jnt_range[j.id][0]) * PRESS_DEPTH_FRACTION
        q = float(env.data.qpos[j.qposadr[0]])
        if q <= press_q and not self.down:
            self.down = True
            return True
        if q > press_q / 2:
            self.down = False
        return False


class _PressCount:
    def __init__(self, spec):
        self.edge, self.count, self.n = PressEdge(spec["joint"]), spec["count"], 0

    def __call__(self, env):
        self.n += self.edge.update(env)
        return self.n == self.count


def hold_steps(env, dwell=None):
    if dwell is None:
        dwell = make_checker(env.spec["success"], env.task)[1]
    return max(dwell, int(round(0.5 * env.control_freq)))


def make_checker(spec: dict, task=None):
    if spec["pred"] == "hold":
        return _build(spec["of"], task), spec["steps"]
    return _build(spec, task), 1


def nodes(spec):
    yield spec
    of = spec.get("of")
    for s in (of if isinstance(of, list) else [of] if of else []):
        yield from nodes(s)
    if spec.get("invariant"):
        yield from nodes(spec["invariant"])


def predicate_names(spec: dict) -> list[str]:
    return list(dict.fromkeys(n["pred"] for n in nodes(spec) if n.get("pred")))


def _rows(spec: dict, depth: int = 0, task=None):
    p = spec.get("pred", "?")
    cm = lambda m: f"{m * 100:g} cm"
    row = lambda t, node=spec: [(depth, t, node)]
    sub = lambda s: _rows(s, depth + 1, task)

    if p in ("all", "any"):
        return row("all of:" if p == "all" else "any of:") + [
            r for s in spec["of"] for r in sub(s)]
    if p == "not":
        return row("NOT:") + sub(spec["of"])
    if p == "sequence":
        rows = row("in this order, each step latching once reached:", None)
        for i, s in enumerate(spec["of"], 1):
            dwell = f" (hold {s['hold']} steps)" if s.get("hold", 1) != 1 else ""
            rows += [(depth + 1, f"stage {i}{dwell}:", None)] + _rows(s, depth + 2, task)
        if spec.get("invariant"):
            rows += [(depth + 1, "and throughout, never breaking:", None)] \
                + _rows(spec["invariant"], depth + 2, task)
        return rows
    if p == "count":
        return row(f"at least {spec['min']} of:") + [r for s in spec["of"] for r in sub(s)]
    if p == "hold":
        return row(f"held for {spec['steps']} steps:", None) + sub(spec["of"])

    leaf = {
        "lift_above": lambda s: f"{s['obj']} lifted above z={s['z_above']:.3f} m",
        "z_below": lambda s: f"{s['obj']} below z={s['z_below']:.3f} m",
        "above": lambda s: f"{s['obj']} at or above {s['base']}",
        "joint_in": lambda s: f"joint {s['joint']} within [{s['lo']:.3f}, {s['hi']:.3f}]",
        "robot_clear": lambda s: "robot clear of " + ", ".join(s["bodies"]),
        "in_site_box": lambda s: f"{s['obj']} released and at rest inside the {s['site']} box",
        "over_site": lambda s: f"{s['obj']} within the {s['site']} box",
        "on_site": lambda s: f"{s['obj']} released and at rest on {s['site']}"
                             + (f" and touching {s['parent']}" if s.get("parent") else ""),
        "within_dist": lambda s: f"{s['a']} within {cm(s['tol'])} of {s['b']}"
                                 + (" (xy only)" if s.get("axes") == "xy" else ""),
        "touching": lambda s: f"{s['a']} touching {s['b']}",
        "touching_part": lambda s: f"{s['a']} touching {s['b']}",
        "axis_dot": lambda s: f"{s['obj']}'s {list(s.get('axis', [0, 0, 1]))} axis dotted with "
                              f"{list(s.get('world_axis', [0, 0, 1]))} at least {s['min_dot']:g}"
                              + (f" and at most {s['max_dot']:g}" if s.get("max_dot") is not None else "")
                              + (" (either end)" if s.get("abs") else ""),
        "grasped": lambda s: f"{s['obj']} grasped",
        "pose_within": lambda s: f"{s['obj']} within {s['pos_tol'] * 1000:g} mm and "
                                 f"{s['rot_tol_deg']:g}° of its goal pose"
                                 + (f"; lateral error ≤ {s['pos_tol_xy'] * 1000:g} mm in the reference frame" if s.get("pos_tol_xy") is not None else "")
                                 + (" (axis only, spin free)" if s.get("axis") else ""),
        "axis_up": lambda s: f"{s['obj']}'s {list(s['axis'])} axis points up "
                             f"(within {s['tol_deg']:g}°)",
        "gripper_open": lambda s: f"gripper open at least {s['min'] * 100:g}%",
        "in_bbox": lambda s: f"{s['a']} inside {s['b']}'s footprint, above its floor",
        "press_count": lambda s: f"{s['joint']} pressed exactly {s['count']} times",
        "depth_in": lambda s: f"{s['a']} sunk at least {s['z_threshold'] * 100:g} cm "
                              f"below {s['b']}'s rim",
        "eef_near_start": lambda s: f"gripper back within {cm(s['pos_tol'])} of its start",
    }
    if p in leaf:
        text = leaf[p](spec)
    else:
        text = getattr(task_module(task, "grade"), p).__doc__.format(
            **{k: ", ".join(map(str, v)) if isinstance(v, list) else v for k, v in spec.items()})
    return row(text, spec if isinstance(_build(spec, task), types.FunctionType) else None)


def describe(spec: dict, task=None) -> list[tuple[int, str]]:
    return [(d, t) for d, t, _ in _rows(spec, task=task)]


def describe_state(spec: dict, env) -> list:
    return [None if node is None else bool(_build(node, env.task)(env)) for _d, _t, node in _rows(spec, task=env.task)]


class Scene:
    def __init__(self, env):
        self.env, self.step, self.frame, self._leaves = env, 0, 0, {}
        check, dwell = make_checker(env.spec["success"], env.task)
        self._success = [check, hold_steps(env, dwell), 0, None]

    def success(self):
        check, hold, streak, frame = self._success
        if frame != self.frame:
            streak = streak + 1 if check(self.env) else 0
            self._success[2:] = [streak, self.frame]
        return self._success[2] >= hold

    def tree(self, spec):
        key = ("tree", json.dumps(spec, sort_keys=True))
        if key not in self._leaves:
            self._leaves[key] = _build(spec, self.env.task)
        return bool(self._leaves[key](self.env))

    def __getattr__(self, pred):
        if pred.startswith("_"):
            raise AttributeError(pred)

        def leaf(**args):
            key = (pred, json.dumps(args, sort_keys=True))
            if key not in self._leaves:
                self._leaves[key] = _build(dict(pred=pred, **args), self.env.task)
            return bool(self._leaves[key](self.env))
        return leaf


@dataclass
class Subtask:
    label: str
    check: object
    ever: bool = False
    needs: tuple = ()
    end_needs: tuple = ()
    objects: tuple = ()
    presses: bool = False


def task_grade(task):
    return task_module(task, "grade")


def evaluate(module, scenes, timed_out):
    subs, invariant = module.subtasks(), getattr(module, "INVARIANT", None)
    earned, flags, killed = [False] * len(subs), [False] * len(subs), None
    for s in scenes:
        now = [bool(t.check(s)) for t in subs]
        if s.step == 0:
            continue
        if invariant is not None and not invariant(s):
            killed = s.step
            break
        flags = now
        for i, t in enumerate(subs):
            if not earned[i] and flags[i] and all(earned[j] for j in t.needs):
                earned[i] = True
    done = [earned[i] if t.ever else flags[i] for i, t in enumerate(subs)]
    for i, t in enumerate(subs):
        done[i] = done[i] and all(done[j] for j in t.end_needs)
    if timed_out:
        done[-1] = False
    score = sum(done) / len(subs)
    return dict(subtasks=[dict(label=t.label, done=d) for t, d in zip(subs, done)], score=score, solved=all(done),
                invariant_broken_at=killed)


def replay(path, scene_class=None, lenient=False):
    from core.sim_env import SimEnv
    roll = Path(path)
    episode = json.loads((roll / "episode.json").read_text())
    verdict = json.loads((roll / "verdict.json").read_text())
    task = episode["task"]
    env = SimEnv(task, seed=episode["seed"])
    data = np.load(roll / "cameras_states.npz")
    states = data["states"]
    if lenient and env.sim_state().size != states.shape[1]:
        for spec in sorted((ASSETS / "replay_packs").glob("*/*/task.json")):
            if json.loads(spec.read_text())["name"] == task:
                old = SimEnv(task, seed=episode["seed"], pack=spec.parent)
                if old.sim_state().size == states.shape[1]:
                    env = old
                    break
    if "static_names" in data.files:
        env.set_static_poses(data["static_names"].tolist(), data["static_poses"])
    elif env.static_poses()[0]:
        if not lenient:
            raise ValueError("recorded before fixture poses were saved: the fixtures cannot be put back where they were")
        env.reset(seed=episode["seed"])
    env.reset_to_sim_state(states[0])
    scene = (scene_class or Scene)(env)

    def scenes():
        previous, step = None, 0
        for frame in range(len(states)):
            tick = float(states[frame, 0])
            if previous is not None and abs(tick - previous) < 1e-7:
                continue
            if previous is not None:
                step += max(1, round((tick - previous) * env.control_freq))
            env.reset_to_sim_state(states[frame])
            scene.step, scene.frame = step, frame
            previous = tick
            yield scene
    return task, task_grade(task), verdict, scenes()


def grade_episode(path):
    task, module, verdict, scenes = replay(path)
    return dict(task=task, **evaluate(module, scenes, bool(verdict.get("timed_out"))))


def graded(roll):
    path = Path(roll) / "subtasks.json"
    if not path.is_file():
        write(roll, grade_episode(roll))
    return json.loads(path.read_text())


def write(roll, grade):
    (Path(roll) / "subtasks.json").write_text(json.dumps(dict(source="grade.py", score=grade["score"], solved=grade["solved"],
                                                              invariant_broken_at=grade["invariant_broken_at"],
                                                              subtasks=grade["subtasks"]), indent=1) + "\n")


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("episode", type=Path)
    ap.add_argument("--write", action="store_true", help="write subtasks.json into the episode folder")
    args = ap.parse_args()
    grade = grade_episode(args.episode)
    if args.write:
        write(args.episode, grade)
    print(json.dumps(grade, indent=1))


if __name__ == "__main__":
    main()
