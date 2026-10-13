"""Subtasks of open middle drawer halfway."""
from core.grade_episode import Subtask


def subtasks():
    return [Subtask('open the middle drawer halfway', lambda s: s.success())]
