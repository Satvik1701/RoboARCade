"""Contact presses on every visible machine key, including spring return."""
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from core.grade_episode import PressEdge
from core.tasks import TASKS, load_scene

BUTTONS = (
    "prog_1", "prog_2", "prog_3", "prog_4", "prog_5",
    "menu_enter", "menu_up", "menu_down", "menu_left", "menu_right",
    "short", "start_stop", "time_up", "time_down", "temp_up", "temp_down",
    "speed_up", "speed_down",
)


@pytest.fixture(scope="module")
def panel_model():
    spec = load_scene(TASKS / "start_incubation")
    probe = spec.worldbody.add_body(name="test_probe", mocap=True)
    probe.add_geom(name="test_tip", type=mujoco.mjtGeom.mjGEOM_SPHERE,
                   size=[.003, 0, 0], contype=1, conaffinity=1)
    return spec.compile()


@pytest.mark.parametrize("button", BUTTONS)
def test_contact_press_and_release(panel_model, button):
    m, d = panel_model, mujoco.MjData(panel_model)
    name = f"button_{button}"
    joint = m.joint(name + "_key")
    d.mocap_pos[0] = [0, 0, 3]
    mujoco.mj_forward(m, d)
    body = m.body(name + "_key").id
    centre = d.xpos[body].copy()
    normal = d.xmat[body].reshape(3, 3)[:, 2]
    edge = PressEdge(name + "_key")
    env = SimpleNamespace(model=m, data=d)
    pair = {m.geom("test_tip").id, m.geom(name + "_contact").id}
    presses, touched = 0, False
    # Move a physical probe into the key, without writing the key's qpos or force.
    for offset in np.linspace(.015, .003, 120):
        d.mocap_pos[0] = centre + normal * offset
        for _ in range(5):
            mujoco.mj_step(m, d)
            presses += int(edge.update(env))
            touched |= any({c.geom1, c.geom2} == pair for c in d.contact)
    assert touched
    assert presses == 1
    assert d.qpos[joint.qposadr[0]] < -.0015
    for other in BUTTONS:
        if other != button:
            assert abs(d.qpos[m.joint(f"button_{other}_key").qposadr[0]]) < .0015
    d.mocap_pos[0] = centre + normal * .025
    for _ in range(500):
        mujoco.mj_step(m, d)
        assert not edge.update(env)
    assert d.qpos[joint.qposadr[0]] > -.00075
    assert not edge.down
