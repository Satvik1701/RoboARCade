"""Subtasks of square."""
from core.grade_episode import Subtask


def subtasks():
    return [
        Subtask('pick the nut and lower it onto the square peg, release',
                lambda s: s.success()),
    ]
