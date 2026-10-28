"""Geometry and saved-state contracts of the asset inspector."""
from types import SimpleNamespace

import mujoco
import numpy as np

from tasks.add_task.inspect_asset import collision_mesh, descendants, restore_state, state_array


def test_collision_mesh_is_convex_not_concave_visual_surface():
    m = mujoco.MjModel.from_xml_string('''<mujoco><asset><mesh name="shape"
        vertex="0 0 0  1 0 0  1 1 0  0 1 0  0 0 1  1 0 1  1 1 1  0 1 1  .5 .5 .2"
        face="0 2 1  0 3 2  0 1 5  0 5 4  1 2 6  1 6 5  2 3 7  2 7 6  3 0 4  3 4 7  4 5 8  5 6 8  6 7 8  7 4 8"/>
        </asset><worldbody><body name="parent"><geom type="mesh" mesh="shape"/>
        <body name="child" pos="0 0 2"><geom size=".1"/></body></body>
        <body name="other"><geom size=".1"/></body></worldbody></mujoco>''')
    vertices, faces = collision_mesh(m, 0)
    assert len(vertices) == 8  # The recessed visual vertex is absent from collision.
    assert faces.shape[1] == 3
    assert descendants(m, 'parent') == {m.body('parent').id, m.body('child').id}
    assert m.body('other').id not in descendants(m, 'parent')


def test_saved_state_restores_physics_and_calls_dynamics_restore():
    m = mujoco.MjModel.from_xml_string('''<mujoco><size nuserdata="2"/>
        <worldbody><body><joint type="slide"/><geom size=".1"/></body></worldbody></mujoco>''')
    d = mujoco.MjData(m)
    calls = []
    env = SimpleNamespace(model=m, data=d, _dynamics=calls.append)
    d.qpos[0] = .3; d.qvel[0] = .2; d.time = 4; d.userdata[:] = [7, 8]
    saved = state_array(env)
    d.qpos[0] = 9; d.qvel[0] = 0; d.time = 20; d.userdata[:] = 0
    restore_state(env, saved)
    assert d.qpos[0] == .3 and d.qvel[0] == .2 and d.time == 4
    np.testing.assert_array_equal(d.userdata, [7, 8])
    assert calls == ['restore']


def test_task_switch_does_not_replay_previous_geometry_rotation(monkeypatch):
    from tasks.add_task import inspect_asset

    def fake_env(task, seed):
        quat = '0.70710678 0.70710678 0 0' if task == 'beam' else '1 0 0 0'
        m = mujoco.MjModel.from_xml_string(f'''<mujoco><worldbody>
            <geom name="table_visual" type="box" size=".4 .4 .025"
                  group="1" pos="0 0 .8" quat="{quat}"/>
            </worldbody></mujoco>''')
        d = mujoco.MjData(m)
        mujoco.mj_forward(m, d)
        return SimpleNamespace(model=m, data=d, dynamics=None, cameras=[],
                               reset=lambda **kwargs: None)

    monkeypatch.setattr(inspect_asset, 'SimEnv', fake_env)
    viewer = inspect_asset.Inspector(SimpleNamespace(
        task='beam', seed=1, body=None, host='127.0.0.1', port=0))
    try:
        viewer.load_task('square')
        # Reconstruct the rotations a newly connected browser receives.
        messages = viewer.server.scene._websock_interface.get_message_buffer().message_from_id
        rotations = {msg.name: msg.wxyz for msg in tuple(messages.values())
                     if type(msg).__name__ == 'SetOrientationMessage'}
        for handle, gid, *_ in viewer.handles:
            displayed = np.asarray(rotations.get(handle.name, [1., 0., 0., 0.]))
            expected = viewer.pose(gid)['wxyz']
            assert abs(np.dot(displayed, expected)) > 1 - 1e-8
    finally:
        viewer.server.stop()
