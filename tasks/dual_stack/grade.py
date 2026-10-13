"""Subtasks of dual stack.

The task's own success."""
import numpy as np

from core.grade_episode import Subtask
from core.grade_episode import _body_pos, _free_joint_qvel, _robot_touching


def dual_stack(base, greens, on_base_dz, dz_tol, xy_tol, same_level_tol, gripper_open_min, lin_vel_max, ang_vel_max):
    """both {greens} resting on {base}, level within {same_level_tol} m, gripper open and clear"""
    blocks = [base, *greens]

    def on_base(gp, rp):
        return abs((gp[2] - rp[2]) - on_base_dz) < dz_tol and np.linalg.norm(gp[:2] - rp[:2]) < xy_tol

    def check(env):
        rp = _body_pos(env, base)
        gp = [_body_pos(env, g) for g in greens]
        if not all(on_base(p, rp) for p in gp) or abs(gp[0][2] - gp[1][2]) >= same_level_tol:
            return False
        if env.gripper_width() <= gripper_open_min or _robot_touching(env, blocks):
            return False
        for b in blocks:
            qv = _free_joint_qvel(env, b)
            if np.linalg.norm(qv[:3]) > lin_vel_max or np.linalg.norm(qv[3:]) > ang_vel_max:
                return False
        return True
    return check


def subtasks():
    return [Subtask('Build a structure exactly two blocks tall: the blue block on the table, and BOTH green blocks resting directly on top of it, side by side at the same height', lambda s: s.success())]
