"""Subtasks of shell game.

Uncover the cubes in rainbow order after the cover press and the shuffles, then line
them up in rainbow order."""
import numpy as np

from core.grade_episode import Subtask
from core.grade_episode import _body_pos

SETUP = [{"pred": "press_count", "joint": "buttonbox_joint_0", "count": 1}, {"pred": "press_count", "joint": "buttonbox_joint_1", "count": 5}]
REVEAL = {"pred": "uncover_order", "order": ["cube_red", "cube_orange", "cube_yellow", "cube_green", "cube_blue", "cube_indigo", "cube_violet"], "by": ["shell_0", "shell_1", "shell_2", "shell_3", "shell_4", "shell_5", "shell_6"]}


UNCOVER_TOL = 0.045
UNCOVER_DZ = 0.05


class UncoverOrder:
    def __init__(self, order, by):
        self.order, self.by = list(order), list(by)
        self.covered, self.revealed, self.broken = set(), set(), False

    def _hidden(self, env, name):
        p = _body_pos(env, name)
        return any(abs(p[2] - q[2]) < UNCOVER_DZ
                   and float(np.hypot(p[0] - q[0], p[1] - q[1])) < UNCOVER_TOL
                   for q in (_body_pos(env, o) for o in self.by))

    def __call__(self, env):
        for i, name in enumerate(self.order):
            if self._hidden(env, name):
                self.covered.add(name)
            elif name in self.covered and name not in self.revealed:
                if any(o not in self.revealed for o in self.order[:i]):
                    self.broken = True
                self.revealed.add(name)
        return not self.broken and len(self.revealed) == len(self.order)


def uncover_order(order, by):
    """cubes revealed in the order {order}"""
    return UncoverOrder(order, by)


def lateral_order(first, second, axis, min_gap, max_gap, align_tol):
    """{second} {min_gap} to {max_gap} m past {first} along {axis}, within {align_tol} m of its row"""
    axis = "xyz".index(axis)
    other = 1 if axis == 0 else 0

    def check(env):
        pa, pb = _body_pos(env, first), _body_pos(env, second)
        gap = float(pb[axis] - pa[axis])
        return bool(min_gap <= gap <= max_gap and abs(float(pb[other] - pa[other])) <= align_tol)
    return check


COLOURS = [c.removeprefix("cube_") for c in REVEAL["order"]]


def subtasks():
    setup_at = [0]
    firsts = [dict(REVEAL, order=REVEAL["order"][:k]) for k in range(1, 8)]

    def uncovered(k):
        def check(s):
            if k == 1 and setup_at[0] < 2 and s.tree(SETUP[setup_at[0]]):
                setup_at[0] += 1
            return setup_at[0] == 2 and s.tree(firsts[k - 1])
        return check
    lined = lambda k: lambda s: all(s.lateral_order(first=a, second=b, axis="y", min_gap=0.045, max_gap=0.15, align_tol=0.06)
                                    for a, b in zip(REVEAL["order"][:k - 1], REVEAL["order"][1:k])) \
        and s.robot_clear(bodies=REVEAL["order"][:k])
    return [Subtask(f'lift the shell off the {c} cube' + (f', after the {COLOURS[k - 2]} one' if k > 1 else ' first'),
                    uncovered(k), ever=True, objects=tuple(REVEAL["by"])) for k, c in enumerate(COLOURS, 1)] + \
           [Subtask(f'line up the cubes {COLOURS[0]} to {COLOURS[k - 1]} in rainbow order', lined(k), needs=(6,), end_needs=(6,),
                    objects=tuple(REVEAL["order"][:k] if k == 2 else REVEAL["order"][k - 1:k])) for k in range(2, 7)] + \
           [Subtask('line up all seven cubes in rainbow order', lambda s: s.success(), needs=(6,), end_needs=(6,), objects=(REVEAL["order"][6],))]
