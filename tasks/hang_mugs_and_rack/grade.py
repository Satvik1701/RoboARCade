"""Subtasks of hang mugs and rack."""
from core.grade_episode import Subtask


def subtasks():
    return [
        Subtask('pick mug a and hang it on the tree',
                lambda s: ((s.within_dist(a='mug_slot_a', b='mug_tree_slot', tol=0.17, axes='xy')
                    and s.lift_above(obj='mug_slot_a', z_above=0.84921)
                    and s.touching_part(a='mug_slot_a', b='mug_tree_slot'))
                and s.robot_clear(bodies=['mug_slot_a']))),
        Subtask('pick mug d and hang it on the tree',
                lambda s: ((s.within_dist(a='mug_slot_d', b='mug_tree_slot', tol=0.17, axes='xy')
                    and s.lift_above(obj='mug_slot_d', z_above=0.84921)
                    and s.touching_part(a='mug_slot_d', b='mug_tree_slot'))
                and s.robot_clear(bodies=['mug_slot_d']))),
        Subtask('pick plate c and stand it in the rack',
                lambda s: ((s.within_dist(a='plate_slot_c', b='rack_slot', tol=0.24, axes='xy')
                    and s.axis_dot(obj='plate_slot_c', min_dot=0.0, max_dot=0.6947, abs=True)
                    and s.lift_above(obj='plate_slot_c', z_above=0.79921)
                    and s.in_bbox(a='plate_slot_c', b='rack_slot')
                    and s.touching_part(a='plate_slot_c', b='rack_slot'))
                and s.robot_clear(bodies=['plate_slot_c']))),
        Subtask('pick plate d and stand it in the rack',
                lambda s: s.success()),
    ]
