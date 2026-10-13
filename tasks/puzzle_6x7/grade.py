"""Subtasks of puzzle 6x7."""
from core.grade_episode import Subtask
from tasks.puzzle_4x6.grade import lights_out


def subtasks():
    return [
        Subtask('turn all 42 cells red', lambda s: s.success(), objects=tuple(f'button_{k}' for k in range(42)), presses=True),
    ]
