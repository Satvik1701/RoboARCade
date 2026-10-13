"""Subtasks of lock the coins in the strongbox."""
from core.grade_episode import Subtask

COINS = [f'coin{i}' for i in range(6)]


def count_in_cavity(s):
    return sum(s.in_site_box(site='cavity', size=[0.064, 0.049, 0.018], obj=c) for c in COINS)


def subtasks():
    return [
        Subtask('pick the key and insert it in the keyway',
                lambda s: s.pose_within(obj='key', rel_to='case', pos=[0.07500000000000001, 0.0, 0.115], quat=[0.707107, 0.0, 0.707107, 0.0], pos_tol=0.006, rot_tol_deg=8.0),
                ever=True),
        Subtask('turn the key to unlock',
                lambda s: (s.joint_in(joint='j_rotor', lo=1.54, hi=1.62)
                and s.over_site(obj='key', site='keyway', lo=[-0.022, -0.011, -0.016], hi=[0.022, 0.011, 0.016])),
                ever=True,
                needs=(0,)),
        Subtask('1 coin in the cavity', lambda s: count_in_cavity(s) >= 1),
        Subtask('2 coins in the cavity', lambda s: count_in_cavity(s) >= 2),
        Subtask('3 coins in the cavity', lambda s: count_in_cavity(s) >= 3),
        Subtask('4 coins in the cavity', lambda s: count_in_cavity(s) >= 4),
        Subtask('close the lid',
                lambda s: (s.joint_in(joint='j_lid', lo=-0.01, hi=0.08)
                and count_in_cavity(s) >= 4),
                needs=(),
                end_needs=(1, 5)),
        Subtask('turn the key to relock',
                lambda s: s.success(),
                needs=(6,),
                end_needs=(6,)),
    ]
