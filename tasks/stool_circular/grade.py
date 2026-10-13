"""Subtasks of stool circular."""
import numpy as np

from core.grade_episode import Subtask
from core.grade_episode import _body_pos


def position_within(obj, pos, pos_tol, rel_to=None):
    """{obj} within {pos_tol} m of its goal position (rotation ignored)"""
    local_goal = np.asarray(pos)

    def check(env):
        goal = local_goal
        if rel_to is not None:
            ref = env.data.body(rel_to)
            goal = ref.xpos + ref.xmat.reshape(3, 3) @ local_goal
        return bool(np.linalg.norm(_body_pos(env, obj) - goal) <= pos_tol)
    return check


def position_within_goal_reader(env, obj, pos, pos_tol, rel_to=None):
    d, ob, ref, local_goal = env.data, env.model.body(obj).id, env.model.body(rel_to).id if rel_to else None, np.array(pos)

    def read():
        goal = d.xpos[ref] + d.xmat[ref].reshape(3, 3) @ local_goal if ref is not None else local_goal
        return float(np.linalg.norm(d.xpos[ob] - goal)), None
    return read


position_within.goal_reader = position_within_goal_reader


def subtasks():
    return [
        Subtask('pick the missing part among the loose parts and seat it on the structure, release',
                lambda s: s.success()),
    ]
