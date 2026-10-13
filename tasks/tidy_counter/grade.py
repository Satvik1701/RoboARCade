"""Subtasks of tidy counter."""
from core.grade_episode import Subtask


def subtasks():
    return [
        Subtask('pick the wine bottle and place it on the cabinet top',
                lambda s: s.on_site(obj='wine_bottle_1_main', site='wooden_cabinet_1_top_side', size=[0.12534, 0.09438, 0.00147], parent='wooden_cabinet_1_main')
                and s.axis_up(obj='wine_bottle_1_main', axis=[0, 0, 1], tol_deg=20.0)),
        Subtask('pick the bowl and place it in the top drawer',
                lambda s: (s.in_site_box(obj='akita_black_bowl_1_main', site='wooden_cabinet_1_top_region', size=[0.02993, 0.07561, 0.10224])
                and (not s.grasped(obj='akita_black_bowl_1_main'))
                and s.joint_in(joint='wooden_cabinet_1_middle_level', lo=-0.01, hi=0.01)
                and s.joint_in(joint='wooden_cabinet_1_bottom_level', lo=-0.01, hi=0.01))),
        Subtask('close the drawer with the bottle upright on top and the bowl inside',
                lambda s: s.success(),
                needs=(1,), end_needs=(0,)),
    ]


def INVARIANT(s):
    return (not (s.grasped(obj='akita_black_bowl_1_main')
            and s.over_site(obj='akita_black_bowl_1_main', site='wooden_cabinet_1_body_region', lo=[-0.127, -0.1105, -0.109], hi=[0.127, 0.1105, 0.1])
            and (not s.over_site(obj='akita_black_bowl_1_main', site='wooden_cabinet_1_top_region', lo=[-0.2, -0.085, -0.11], hi=[0.055, 0.085, 0.11]))))
