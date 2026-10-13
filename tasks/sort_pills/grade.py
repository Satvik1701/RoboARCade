"""Subtasks of sort pills.

Both caps off, Saturday and Sunday each get exactly two capsules and one tablet, every
organizer lid closed."""
from core.grade_episode import Subtask

DOSE = dict(capsule=2, tablet=1)


def cap_off(s, kind):
    return not s.within_dist(a=f'{kind}_lid', b=f'{kind}_bottle_lid_seat', tol=0.05) and not s.touching_part(a=f'{kind}_lid', b=f'{kind}_bottle')


def dosed(s, day, kind):
    box = f'{day}_compartment'
    n = sum(s.in_bbox(a=p, b=box) and s.depth_in(a=p, b=box, z_threshold=0.006) and not s.grasped(obj=p)
            for p in (f'bottle_{kind}_{i}' for i in range(1, 7)))
    return n == DOSE[kind]


def subtasks():
    return [Subtask('take the cap off the first container', lambda s: cap_off(s, 'capsule')),
            Subtask('take the cap off the second container', lambda s: cap_off(s, 'tablet'))] + [
            Subtask(f'fill the {d.title()} compartment, {nth} kind', lambda s, d=d, k=k: dosed(s, d, k))
            for d in ('saturday', 'sunday') for nth, k in (('first', 'capsule'), ('second', 'tablet'))] + [
            Subtask('close the organizer lids', lambda s: s.success())]
