"""Subtasks of start making breakfast."""
from core.grade_episode import Subtask


def subtasks():
    return [
        Subtask('turn the stove dial to the mark',
                lambda s: s.joint_in(joint='flat_stove_1_button', lo=0.95, hi=1.65)),
        Subtask('pick the first moka pot and place it on the cook region',
                lambda s: s.on_site(obj='moka_pot_1_main', site='flat_stove_1_cook_region', size=[0.075, 0.075, 0.0025], parent='flat_stove_1_main')),
        Subtask('pick the second moka pot and place it on the cook region',
                lambda s: s.on_site(obj='moka_pot_2_main', site='flat_stove_1_cook_region', size=[0.075, 0.075, 0.0025], parent='flat_stove_1_main')),
        Subtask('pick the bread and place it on the plate',
                lambda s: s.success()),
    ]
