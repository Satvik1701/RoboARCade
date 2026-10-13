"""Tests of the stool_circular task.
"""
from types import SimpleNamespace
import mujoco
from core.grade_episode import _build


def test_position_only_ignores_inversion_but_checks_transformed_position():
    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <body name="ref" pos="1 2 3" quat="0.70710678 0 0 0.70710678"/>
      <body name="piece"><freejoint/><geom type="sphere" size=".01" mass="1"/></body>
      </worldbody></mujoco>''')
    data = mujoco.MjData(model)
    env = SimpleNamespace(model=model, data=data)
    spec = dict(pred='position_within', obj='piece', rel_to='ref', pos=[.1, 0, 0], pos_tol=.005)
    check = _build(spec, "stool_circular")
    for quat in ([1,0,0,0], [0,1,0,0], [.70710678,.70710678,0,0]):
        data.qpos[:] = [1,2.1,3,*quat]
        mujoco.mj_forward(model, data)
        assert check(env)
        assert not _build(dict(spec, pred='pose_within', quat=[1,0,0,0], rot_tol_deg=15))(env)
        data.qpos[0] += .006
        mujoco.mj_forward(model, data)
        assert not check(env)


