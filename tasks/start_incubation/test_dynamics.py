"""Instrument readout follows controls, simulation time and saved states."""
import json
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from core.tasks import TASKS, load_scene
from tasks.start_incubation.dynamics import Dynamics, LATCHES, SEGMENTS


@pytest.fixture(scope="module")
def model():
    return load_scene(TASKS / "start_incubation").compile()


@pytest.fixture
def env(model):
    e = SimpleNamespace(model=model, data=mujoco.MjData(model),
                        spec=json.loads((TASKS / "start_incubation/task.json").read_text()))
    e.dynamics = Dynamics(e)
    return e


def advance(env, seconds):
    env.data.time += seconds
    env.dynamics.physics_step(env)


def press(env, name, release=True):
    q = env.model.joint(f"button_{name}_key").qposadr[0]
    env.data.qpos[q] = -.003
    env.dynamics.physics_step(env)
    if release:
        env.data.qpos[q] = 0
        env.dynamics.physics_step(env)


def test_program_countdown_temperature_pause_and_completion(env):
    d = env.dynamics
    assert d.readout(env)["actual_celsius"] == 24
    press(env, "prog_1")
    advance(env, 5)
    r = d.readout(env)
    assert r["running"] and r["remaining_seconds"] == 145
    assert r["actual_celsius"] == 29
    press(env, "start_stop")
    advance(env, 10)
    assert d.readout(env)["remaining_seconds"] == 145
    assert d.readout(env)["actual_celsius"] == 27
    press(env, "start_stop")
    advance(env, 145)
    r = d.readout(env)
    assert r["done"] and not r["running"] and r["remaining_seconds"] == 0
    assert r["actual_celsius"] == 45


def test_held_buttons_do_not_repeat_and_presets_keep_edits(env):
    press(env, "time_up", release=False)
    for _ in range(10): advance(env, .1)
    assert env.dynamics.readout(env)["duration_seconds"] == 165
    env.data.qpos[env.model.joint("button_time_up_key").qposadr[0]] = 0
    advance(env, 0)
    press(env, "temp_up")
    press(env, "speed_up")
    press(env, "prog_2")
    assert env.dynamics.readout(env)["program"] == 2
    assert env.dynamics.readout(env)["duration_seconds"] == 150
    press(env, "prog_1")
    r = env.dynamics.readout(env)
    assert (r["duration_seconds"], r["target_celsius"], r["target_rpm"]) == (165, 46, 50)
    advance(env, 1)
    assert env.dynamics.readout(env)["actual_rpm"] == 50
    press(env, "start_stop")
    advance(env, .025)
    assert env.dynamics.readout(env)["actual_rpm"] == pytest.approx(25)
    advance(env, 1)
    assert env.dynamics.readout(env)["actual_rpm"] == 0


def test_short_is_momentary_and_menu_edits_selected_field(env):
    press(env, "menu_enter")
    press(env, "menu_right")  # Temperature.
    press(env, "menu_up")
    assert env.dynamics.readout(env)["target_celsius"] == 46
    press(env, "menu_down")
    press(env, "menu_left")  # Time.
    press(env, "menu_down")
    assert env.dynamics.readout(env)["duration_seconds"] == 135
    press(env, "short", release=False)
    advance(env, .5)
    assert env.dynamics.readout(env)["actual_rpm"] == 500
    assert env.dynamics.readout(env)["remaining_seconds"] == 135
    env.data.qpos[env.model.joint("button_short_key").qposadr[0]] = 0
    advance(env, 0)
    advance(env, .5)
    assert env.dynamics.readout(env)["actual_rpm"] == 0


def test_restore_reproduces_display_and_press_latches(env):
    press(env, "speed_up")
    press(env, "prog_1", release=False)
    advance(env, 3.5)
    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    state = np.empty(mujoco.mj_stateSize(env.model, spec))
    mujoco.mj_getState(env.model, env.data, state, spec)
    readout = env.dynamics.readout(env)
    colors = env.model.geom_rgba.copy()
    env.dynamics.reset(env)
    mujoco.mj_setState(env.model, env.data, state, spec)
    env.dynamics.restore(env)
    assert env.dynamics.readout(env) == readout
    np.testing.assert_array_equal(env.model.geom_rgba, colors)
    advance(env, .5)  # The still-held Program 1 key must not restart the timer.
    assert env.dynamics.readout(env)["remaining_seconds"] == 146


def test_digits_match_machine_state_and_setting_bounds(env):
    press(env, "prog_1")
    advance(env, 61)
    # 01:29 and 45/45 are actual segment states, not text in a screenshot.
    for field, text in [("time", "0129"), ("temp", "4545")]:
        for ids, char in zip(env.dynamics.digits[field], text):
            active = ''.join(letter for letter, gid in zip('abcdefg', ids)
                             if env.model.geom_rgba[gid, 0] < .5)
            assert active == SEGMENTS[int(char)]
    for field in range(3):
        s = env.dynamics.state(env)
        for _ in range(450): env.dynamics.adjust(s, field, -1)
        assert env.dynamics.settings(s)[field] == env.spec['incubation_panel']['limits'][field][0]
        for _ in range(450): env.dynamics.adjust(s, field, 1)
        assert env.dynamics.settings(s)[field] == env.spec['incubation_panel']['limits'][field][1]


def test_contact_press_starts_program_without_writing_key_pose():
    spec = load_scene(TASKS / "start_incubation")
    probe = spec.worldbody.add_body(mocap=True)
    probe.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[.003, 0, 0])
    m = spec.compile()
    e = SimpleNamespace(model=m, data=mujoco.MjData(m),
                        spec=json.loads((TASKS / "start_incubation/task.json").read_text()))
    e.dynamics = Dynamics(e)
    mujoco.mj_forward(m, e.data)
    b = m.body('button_prog_1_key').id
    centre = e.data.xpos[b].copy()
    normal = e.data.xmat[b].reshape(3, 3)[:, 2]
    for offset in np.linspace(.015, .003, 120):
        e.data.mocap_pos[0] = centre+normal*offset
        for _ in range(5):
            e.dynamics.physics_step(e)
            mujoco.mj_step(m, e.data)
    assert e.dynamics.readout(e)['running']
    assert 0 < e.dynamics.readout(e)['remaining_seconds'] < 150
