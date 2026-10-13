"""Subtasks of set up pcr run."""
from core.grade_episode import Subtask


def subtasks():
    return [
        Subtask('pick the plate and seat it in the machine',
                lambda s: (s.over_site(obj='pcr_plate', site='plate_seat', lo=[-0.00125, -0.00125, -0.001], hi=[0.00125, 0.00125, 0.001])
                and s.axis_up(obj='pcr_plate', axis=[0, 0, 1], tol_deg=10.0)
                and s.axis_dot(obj='pcr_plate', axis=[1, 0, 0], world_axis=[1, 0, 0], min_dot=0.9848, abs=True))),
        Subtask('close the lid',
                lambda s: ((s.over_site(obj='pcr_plate', site='plate_seat', lo=[-0.00125, -0.00125, -0.001], hi=[0.00125, 0.00125, 0.001])
                    and s.axis_up(obj='pcr_plate', axis=[0, 0, 1], tol_deg=10.0)
                    and s.axis_dot(obj='pcr_plate', axis=[1, 0, 0], world_axis=[1, 0, 0], min_dot=0.9848, abs=True))
                and s.joint_in(joint='lid', lo=-1.95, hi=-1.85))),
        Subtask('latch the lever',
                lambda s: (((s.over_site(obj='pcr_plate', site='plate_seat', lo=[-0.00125, -0.00125, -0.001], hi=[0.00125, 0.00125, 0.001])
                        and s.axis_up(obj='pcr_plate', axis=[0, 0, 1], tol_deg=10.0)
                        and s.axis_dot(obj='pcr_plate', axis=[1, 0, 0], world_axis=[1, 0, 0], min_dot=0.9848, abs=True))
                    and s.joint_in(joint='lid', lo=-1.95, hi=-1.85))
                and s.joint_in(joint='lid_lever', lo=-0.95, hi=-0.85))),
        Subtask('tighten the force knob',
                lambda s: s.success()),
    ]


def INVARIANT(s):
    return s.axis_up(obj='pcr_plate', axis=[0, 0, 1], tol_deg=30.0)
