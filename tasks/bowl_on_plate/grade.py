"""Subtasks of bowl on plate.

The task's own success."""
from core.grade_episode import Subtask


def subtasks():
    return [Subtask('Put the bowl on the plate, resting on it and centered, then release it', lambda s: s.success())]
