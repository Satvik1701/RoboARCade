"""Subtasks of handover dominos.

1 to 5 blocks standing on end in one line, each the first time it holds, then the
task's own success (the block-to-block cascade after five stood)."""
import numpy as np

from core.grade_episode import Subtask
from core.grade_episode import _body_pos, _contact_body_pairs, _free_joint_qvel, _robot_touching

STANDING_DOT, STILL_M, MIN_GAP_M, MAX_GAP_M, LINE_TOL_M = 0.9, 0.002, 0.035, 0.105, 0.02
DOWN_DOT = 0.7
STABLE_LIN_VEL_M_S = 0.01
STABLE_ANG_VEL_RAD_S = 0.1
STANDING_Z_M = (0.0483, 0.0723)
BLOCKS = [f"jenga_{i}" for i in range(1, 19)]


def line_count(P):
    if len(P) < 2:
        return len(P)
    best = 1
    for i in range(len(P)):
        for j in range(i + 1, len(P)):
            d = P[j] - P[i]; L = np.linalg.norm(d)
            if L < 1e-6:
                continue
            d = d / L; off = P - P[i]
            on = np.abs(off[:, 0] * d[1] - off[:, 1] * d[0]) <= LINE_TOL_M
            run = 1
            for g in np.diff(np.sort(off[on] @ d)):
                run = run + 1 if MIN_GAP_M <= g <= MAX_GAP_M else 1
                best = max(best, run)
    return best


class DominoChain:
    def __init__(self, bodies, min, min_gap, max_gap, line_tol):
        self.bodies, self.minimum = tuple(bodies), int(min)
        self.min_gap, self.max_gap, self.line_tol = float(min_gap), float(max_gap), float(line_tol)
        self.members, self.seed, self.connected, self.robot_touched, self.dead = None, None, set(), set(), False

    @staticmethod
    def _axis_dot(env, body):
        return abs(float(env.data.body(body).xmat.reshape(3, 3)[:, 2] @ [0, 0, 1]))

    def _standing(self, env, body):
        qv = _free_joint_qvel(env, body)
        height = float(_body_pos(env, body)[2]) - float(env.spec["table_z"])
        return (self._axis_dot(env, body) >= STANDING_DOT
                and STANDING_Z_M[0] <= height <= STANDING_Z_M[1]
                and np.linalg.norm(qv[:3]) <= STABLE_LIN_VEL_M_S
                and np.linalg.norm(qv[3:]) <= STABLE_ANG_VEL_RAD_S)

    def _find_line(self, env):
        standing = [body for body in self.bodies if self._standing(env, body)]
        if len(standing) < self.minimum:
            return None
        pos = {body: _body_pos(env, body)[:2].copy() for body in standing}
        best = ()
        for i, first in enumerate(standing):
            for second in standing[i + 1:]:
                delta = pos[second] - pos[first]
                length = float(np.linalg.norm(delta))
                if length < self.min_gap:
                    continue
                axis = delta / length
                normal = np.array([-axis[1], axis[0]])
                aligned = [body for body in standing
                           if abs(float((pos[body] - pos[first]) @ normal))
                           <= self.line_tol]
                aligned.sort(key=lambda body: float(pos[body] @ axis))
                run = [aligned[0]]
                for body in aligned[1:]:
                    gap = float((pos[body] - pos[run[-1]]) @ axis)
                    if self.min_gap <= gap <= self.max_gap:
                        run.append(body)
                    else:
                        if len(run) > len(best):
                            best = tuple(run)
                        run = [body]
                if len(run) > len(best):
                    best = tuple(run)
        return best if len(best) >= self.minimum else None

    def _contacts(self, env):
        m = env.model
        owners = {m.body(name).id: name for name in self.members}
        edges = set()
        for b1, b2 in _contact_body_pairs(env):
            a, b = owners.get(b1), owners.get(b2)
            if a is not None and b is not None and a != b:
                edges.add(frozenset((a, b)))
        return edges

    def __call__(self, env):
        if self.dead:
            return False
        if self.members is None:
            line = self._find_line(env)
            if line is None or _robot_touching(env, line):
                return False
            self.members = tuple(line)
            return False

        touched = {body for body in self.members if _robot_touching(env, [body])}
        self.robot_touched |= touched
        if len(self.robot_touched) > 1:
            self.dead = True
            return False

        dot = {body: self._axis_dot(env, body) for body in self.members}
        tilted = {body for body in self.members if dot[body] < STANDING_DOT}
        if self.seed is None:
            if not tilted:
                return False
            if self.robot_touched:
                candidate = next(iter(self.robot_touched))
                if candidate not in tilted:
                    self.dead = True
                    return False
                self.seed = candidate
            else:
                self.seed = min(tilted, key=dot.get)
            self.connected = {self.seed}

        edges = self._contacts(env)
        changed = True
        while changed:
            changed = False
            for edge in edges:
                if edge & self.connected and not edge <= self.connected:
                    other = next(iter(edge - self.connected))
                    if other not in self.robot_touched:
                        self.connected.add(other)
                        changed = True

        return (len(self.connected) == len(self.members)
                and all(dot[body] <= DOWN_DOT for body in self.members))


def domino_chain(bodies, min, min_gap, max_gap, line_tol):
    """at least {min} of {bodies} latched as one stable domino line, then toppled through block-to-block contact after the robot knocks only one"""
    return DominoChain(bodies, min, min_gap, max_gap, line_tol)


def subtasks():
    state = dict(last=None, line=0, frame=-1, sampled=False)

    def standing(k):
        def check(s):
            if state["frame"] != s.frame:
                state["frame"], state["sampled"] = s.frame, s.frame % 2 == 0
                if state["sampled"]:
                    pos = np.array([s.env.data.body(b).xpos for b in BLOCKS])
                    zax = np.array([s.env.data.body(b).xmat.reshape(3, 3)[2, 2] for b in BLOCKS])
                    still = np.ones(len(pos), bool) if state["last"] is None else np.linalg.norm(pos - state["last"], axis=1) < STILL_M
                    state["last"] = pos.copy()
                    ok = [i for i, b in enumerate(BLOCKS) if abs(zax[i]) >= STANDING_DOT and still[i] and not _robot_touching(s.env, [b])]
                    state["line"] = line_count(pos[ok, :2])
            return state["sampled"] and state["line"] >= k
        return check

    return [Subtask(label, standing(k), ever=True, objects=tuple(BLOCKS)) for k, label in zip(range(1, 6), ['1 block standing on end', '2 blocks standing on end in one line', '3 blocks standing on end in one line', '4 blocks standing on end in one line', '5 blocks standing on end in one line'])] + \
           [Subtask('knock the first block', lambda s: s.success(), ever=True, needs=(4,))]
