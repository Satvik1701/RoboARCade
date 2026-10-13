"""Subtasks of puzzle 4x6, and the Lights Out success leaf (puzzle_6x7 plays the same board)."""
from core.grade_episode import Subtask


def lights_out(rows, cols):
    """Lights Out {rows}x{cols}: every button red"""
    return lambda env: not env.data.userdata[:rows * cols].any()


def subtasks():
    return [Subtask('turn all 24 cells red', lambda s: s.success(), objects=tuple(f'button_{k}' for k in range(24)), presses=True)]
