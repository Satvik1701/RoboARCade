"""The visible well surfaces and contact geometry agree and support real tubes."""
import mujoco
import numpy as np
import pytest

from core.tasks import TASKS, load_scene


@pytest.fixture(scope='module')
def aligned_model():
    return load_scene(TASKS/'start_incubation').compile()


def ray(model, data, origin, group):
    mask = np.zeros(6, dtype=np.uint8); mask[group] = 1
    hit = np.array([-1], dtype=np.int32)
    distance = mujoco.mj_ray(model, data, origin, np.array([0., 0., -1.]), mask, True, -1, hit)
    assert distance >= 0
    return origin[2]-distance


@pytest.mark.parametrize('row,col', [(r, c) for r in range(4) for c in range(6)])
def test_well_visual_floor_matches_contact_surface_and_tube_seats(aligned_model, row, col):
    m = aligned_model; d = mujoco.MjData(m); mujoco.mj_forward(m, d)
    center = d.site(f'machine_well_r{row}_c{col}').xpos.copy()
    # Avoid exact shared triangle vertices, where the source mesh has tiny seams.
    for dx, dy in ((.0003, .0002), (-.0003, -.0002)):
        point = center + [dx, dy, .04]
        visual, contact = ray(m, d, point, 2), ray(m, d, point, 3)
        assert abs(visual-center[2]) < .0001
        assert 0 <= contact-visual < .0002
    body = m.body('tube_2').id
    a = int(m.jnt_qposadr[m.body_jntadr[body]])
    d.qpos[a:a+3] = center+[0, 0, .001]
    d.qpos[a+3:a+7] = [1, 0, 0, 0]
    mujoco.mj_forward(m, d)
    for _ in range(round(5/m.opt.timestep)): mujoco.mj_step(m, d)
    assert abs(d.xpos[body, 2]-center[2]) < .001
    assert np.linalg.norm(d.xpos[body, :2]-center[:2]) < .001
    assert d.xmat[body].reshape(3, 3)[2, 2] > np.cos(np.radians(12))
    block = m.body('thermoblock').id
    assert any({int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])} == {body, block}
               for c in d.contact)


def test_button_caps_and_colliders_have_identical_poses_and_sizes(aligned_model):
    m = aligned_model; d = mujoco.MjData(m); mujoco.mj_forward(m, d)
    for j in range(m.njnt):
        name = m.joint(j).name
        if not name.startswith('button_'): continue
        stem = name.removesuffix('_key')
        visual, contact = m.geom(stem+'_cap').id, m.geom(stem+'_contact').id
        np.testing.assert_array_equal(d.geom_xpos[visual], d.geom_xpos[contact])
        np.testing.assert_array_equal(d.geom_xmat[visual], d.geom_xmat[contact])
        np.testing.assert_array_equal(m.geom_size[visual], m.geom_size[contact])
