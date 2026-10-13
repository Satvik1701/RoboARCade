"""Subtasks of sort lego bricks.

1 to 9 bricks in the bin of their colour, released."""
from core.grade_episode import Subtask

BRICKS = [(f"{c}_brick_{i}", f"{c}_bin") for c in ("red", "green", "yellow") for i in (1, 2, 3)]


def sorted_count(s):
    return sum(s.in_bbox(a=b, b=bin) and s.depth_in(a=b, b=bin, z_threshold=0.02) and s.robot_clear(bodies=[b])
               for b, bin in BRICKS)


def subtasks():
    return [Subtask("1 brick in the bin of its colour" if k == 1 else f"{k} bricks in the bins of their colour",
                    (lambda s: s.success()) if k == 9 else (lambda s, k=k: sorted_count(s) >= k)) for k in range(1, 10)]
