"""Subtasks of serve warm drinks.

Door open, mug inside, door closed on it, start pressed with the mug in and
the door closed, then served on the plate."""
from core.grade_episode import Subtask
from core.grade_episode import PressEdge

DOOR = {"joint": "microwave_1_microjoint", "comparison": "less_than", "value": -1.3}
MUG_INSIDE = {"pred": "over_site", "obj": "white_yellow_mug_1_main", "site": "microwave_1_heating_region", "lo": [-0.12, -0.0835, -0.09], "hi": [0.12, 0.0835, 0.08]}
DOOR_CLOSED = {"pred": "joint_in", "joint": "microwave_1_microjoint", "lo": -0.05, "hi": 0.0}
START = {"pred": "press_edge", "joint": "microwave_1_start_key"}


def press_edge(joint):
    """joint {joint} pressed (counts once, at a quarter of its travel)"""
    return PressEdge(joint).update


def subtasks():
    edge = PressEdge(START["joint"])

    def pressed(s):
        down = bool(edge.update(s.env))
        return down and s.tree(MUG_INSIDE) and s.tree(DOOR_CLOSED)
    door = lambda s: s.env.data.qpos[s.env.model.joint(DOOR["joint"]).qposadr[0]] < DOOR["value"]
    return [
        Subtask('open the microwave door', door, ever=True, objects=(DOOR["joint"],)),
        Subtask('pick the mug and place it inside the microwave', lambda s: s.tree(MUG_INSIDE), ever=True, needs=(0,)),
        Subtask('close the door', lambda s: s.tree(MUG_INSIDE) and s.tree(DOOR_CLOSED), ever=True, needs=(1,)),
        Subtask('press start', pressed, ever=True, needs=(2,), objects=(START["joint"],)),
        Subtask("pick the mug out of the microwave and place it on the plate to the robot's right", lambda s: s.success(), needs=(3,), end_needs=(3,)),
    ]


def INVARIANT(s):
    return s.tree({"pred": "axis_up", "obj": "white_yellow_mug_1_main", "axis": [0, 0, 1], "tol_deg": 30})
