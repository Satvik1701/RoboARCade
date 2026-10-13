"""Subtasks of gamepad."""
from core.grade_episode import Subtask


def subtasks():
    return [
        Subtask('pick the missing part among the loose parts and seat it on the structure, release',
                lambda s: s.success()),
    ]
