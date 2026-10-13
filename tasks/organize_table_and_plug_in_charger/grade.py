"""Subtasks of organize table and plug in charger."""
from core.grade_episode import Subtask


def tips_in(a, b, tips, z_below):
    """both blade tips of {a} less than {z_below} m above the base of {b}"""
    def check(env):
        base = env.data.body(b)
        return all((base.xmat.reshape(3, 3).T @ (env.data.geom(t).xpos - base.xpos))[2] < z_below for t in tips)
    return check


def subtasks():
    return [
        Subtask('pick the headphones and place them on the stand',
                lambda s: ((s.in_bbox(a='headphones', b='headset_stand')) and s.robot_clear(bodies=['headphones']))),
        Subtask('pick the charger and plug it into the socket',
                lambda s: ((s.touching(a='charger', b='socket')
                    and s.in_bbox(a='charger', b='socket')
                    and s.tips_in(a='charger', b='socket', tips=['charger_tip_0', 'charger_tip_1'], z_below=0.013)
                    and s.axis_up(obj='charger', axis=[0, 1, 0], tol_deg=20))
                and s.robot_clear(bodies=['charger']))),
    ] + [Subtask(f'{k} pen{"s" * (k > 1)} in the holder', lambda s, k=k: pens(s) >= k) for k in (1, 2, 3)] + [
        Subtask('4 pens in the holder', lambda s: s.success()),
    ]


def pens(s):
    return sum(s.in_bbox(a=p, b='holder') and s.depth_in(a=p, b='holder', z_threshold=0.04) and s.robot_clear(bodies=[p])
               for p in ('pen0', 'pen1', 'pen2', 'pen3'))
