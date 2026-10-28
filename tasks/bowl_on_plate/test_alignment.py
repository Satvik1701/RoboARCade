"""Collision surfaces follow the visible bowl/plate, including their cavities."""
import mujoco
import numpy as np
import pytest

from core.tasks import TASKS, load_scene


@pytest.fixture(scope='module')
def model():
    return load_scene(TASKS/'bowl_on_plate').compile()


def hit(model, data, origin, direction, group):
    mask = np.zeros(6, dtype=np.uint8)
    mask[group] = 1
    return mujoco.mj_ray(model, data, origin, direction, mask, True, -1,
                         np.array([-1], dtype=np.int32))


@pytest.mark.parametrize('stem,radius,height', [
    ('akita_black_bowl_1', .05, .053), ('plate_1', .06, .017)])
def test_visible_and_contact_surfaces_agree(model, stem, radius, height):
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    body = model.body(stem+'_main').id
    groups, rgba = model.geom_group.copy(), model.geom_rgba.copy()
    try:
        model.geom_group[:] = 5
        for gid in range(model.ngeom):
            if model.geom_bodyid[gid] == body:
                model.geom_group[gid] = 3 if model.geom_contype[gid] or model.geom_conaffinity[gid] else 1
                model.geom_rgba[gid, 3] = 1
        errors = []
        # Top-down probes include the hollow bowl's floor and raised rim.
        for angle in np.linspace(0, 2*np.pi, 16, endpoint=False):
            direction = np.array([np.cos(angle), np.sin(angle), 0.])
            for r in [0., radius*.3, radius*.6, radius*.9]:
                origin = data.xpos[body]+r*direction+[0, 0, height+.03]
                visual = hit(model, data, origin, np.array([0., 0., -1.]), 1)
                contact = hit(model, data, origin, np.array([0., 0., -1.]), 3)
                assert visual >= 0 and contact >= 0
                errors.append(abs(visual-contact))
        assert max(errors) < .0015, f'Max contact/visual offset: {max(errors)*1000:.3f} mm'
        if stem.startswith('akita'):
            origin = data.xpos[body]+[0, 0, height+.03]
            floor_height = height+.03-hit(model, data, origin, np.array([0., 0., -1.]), 3)
            assert floor_height < .015  # A hull across the rim would seal the bowl.
    finally:
        model.geom_group[:] = groups
        model.geom_rgba[:] = rgba


def test_original_mass_and_inertia_are_retained(model):
    for stem, mass, inertia in [
        ('akita_black_bowl_1', .15, [.00017951291555296712, .00011100656646446331, .0001097981178608972]),
        ('plate_1', .030483259872438436, [6.499960924254833e-05, 3.380168298163339e-05, 3.19661970235976e-05]),
    ]:
        body = model.body(stem+'_main').id
        np.testing.assert_allclose(model.body_mass[body], mass, rtol=1e-9)
        np.testing.assert_allclose(model.body_inertia[body], inertia, rtol=1e-9)
