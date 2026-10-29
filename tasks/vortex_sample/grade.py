"""Subtasks of vortex sample.

Speed knob set, mixer on, tube held in the cup until mixed (in that order), then the
mixer off and the tube back in the rack."""
from core.grade_episode import Subtask

CHECKS = {"knob_set": {"pred": "joint_in", "joint": "knob", "lo": 2.07, "hi": 4.7}, "mixer_on": {"pred": "all", "of": [{"pred": "any", "of": [{"pred": "joint_in", "joint": "switch", "lo": 0.3, "hi": 0.85}, {"pred": "joint_in", "joint": "switch", "lo": -0.85, "hi": -0.3}]}, {"pred": "joint_in", "joint": "knob", "lo": 2.07, "hi": 4.7}]}, "mixing_complete": {"pred": "all", "of": [{"pred": "userdata_in", "index": 0, "lo": 5, "hi": 5}, {"pred": "joint_in", "joint": "knob", "lo": 2.07, "hi": 4.7}]}, "tube_returned": {"pred": "all", "of": [{"pred": "in_site_box", "obj": "tube", "site": "rack_wells", "size": [0.0947, 0.032, 0.008]}, {"pred": "axis_up", "obj": "tube", "axis": [0, 0, 1], "tol_deg": 15.0}, {"pred": "robot_clear", "bodies": ["tube"]}]}, "mixer_off": {"pred": "joint_in", "joint": "switch", "lo": -0.1, "hi": 0.1}}


def joint_speed_in(joint, lo, hi):
    """joint {joint} speed within [{lo}, {hi}] rad/s"""
    def check(env):
        j = env.model.joint(joint)
        return bool(lo <= float(env.data.qvel[j.dofadr[0]]) <= hi)
    return check


def userdata_in(index, lo, hi):
    """scene state {index} within [{lo}, {hi}]"""
    return lambda env: bool(lo <= env.data.userdata[index] <= hi)


def subtasks():
    c = lambda name: lambda s: s.tree(CHECKS[name])
    return [
        Subtask('turn the speed knob into range', c("knob_set")),
        Subtask('switch the mixer on', c("mixer_on"), ever=True, needs=(0,)),
        Subtask('pick the tube and hold it in the mixer cup until mixing completes', c("mixing_complete"), ever=True, needs=(1,)),
        Subtask('switch the mixer off', c("mixer_off"), needs=(2,), end_needs=(2,)),
        Subtask('return the tube to the rack', lambda s: s.success(), needs=(2,), end_needs=(2,)),
    ]


def INVARIANT(s):
    return s.tree({"pred": "all", "of": [{"pred": "axis_up", "obj": "tube", "axis": [0, 0, 1], "tol_deg": 15.0}, {"pred": "any", "of": [{"pred": "not", "of": {"pred": "robot_clear", "bodies": ["tube"]}}, {"pred": "touching_part", "a": "tube", "b": "tube_rack"}]}]})
