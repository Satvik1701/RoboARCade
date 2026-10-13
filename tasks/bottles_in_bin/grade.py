"""Subtasks of bottles in bin."""
from core.grade_episode import Subtask

BOTTLES = ['bottle_slot_a', 'bottle_slot_b', 'bottle_slot_c', 'bottle_slot_d']


def count_in_bin(s):
    return sum(s.in_site_box(obj=b, site='bin_inside', size=[0.12, 0.12, 0.105]) for b in BOTTLES)


def subtasks():
    return [
        Subtask('1 bottle in the bin', lambda s: count_in_bin(s) >= 1),
        Subtask('2 bottles in the bin', lambda s: count_in_bin(s) >= 2),
        Subtask('3 bottles in the bin', lambda s: count_in_bin(s) >= 3),
        Subtask('4 bottles in the bin',
                lambda s: s.success()),
    ]
