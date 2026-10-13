"""Subtasks of soup and sauce.

Each can in the basket at the end."""
from core.grade_episode import Subtask

BOX = dict(site='basket_1_contain_region', size=[0.06108, 0.06108, 0.06949])


def subtasks():
    return [Subtask('soup in basket', lambda s: s.in_site_box(obj='alphabet_soup_1_main', **BOX)),
            Subtask('sauce in basket', lambda s: s.success())]
