"""Tests of the vortex_sample task."""

import mujoco
import numpy as np
import pytest
from core.sim_env import SimEnv
from core import grade_episode as predicates
from core.grade_episode import make_checker



@pytest.fixture
def mixer():
    env = SimEnv("vortex_sample", seed=0)
    env.reset(seed=0)
    return env


def controls(env, n):
    for _ in range(n):
        for _ in range(env._substeps):
            env._step_physics()
        env._advance_episode_state()


def dial(env, detent, switch):
    env.data.joint("knob").qpos[0] = detent * 0.46
    env.data.joint("switch").qpos[0] = switch
    env.forward()


@pytest.mark.parametrize("detent", [0, 1, 5, 10])
def test_vortex_on_mapping_and_physical_speed(mixer, detent):
    dial(mixer, detent, 0.4)
    controls(mixer, 20)
    target = (600 + (detent - 1) * 2600 / 9) * np.pi / 30 if detent else 0
    assert mixer.data.actuator("platform_pivot").ctrl[0] == pytest.approx(target)
    assert mixer.data.joint("platform_pivot").qvel[0] == pytest.approx(target, abs=4)
    assert mixer.data.userdata[0] == 0


def test_vortex_touch_without_pressure_and_off(mixer):
    dial(mixer, 10, -0.4)
    controls(mixer, 20)
    assert mixer.data.actuator("platform_pivot").ctrl[0] == 0
    assert abs(mixer.data.joint("platform_pivot").qvel[0]) < 1
    dial(mixer, 10, 0.4)
    controls(mixer, 20)
    dial(mixer, 10, 0)
    controls(mixer, 1)
    assert mixer.data.actuator("platform_pivot").ctrl[0] == 0
    assert mixer.data.joint("platform_pivot").qvel[0] > 100
    controls(mixer, 14)
    assert abs(mixer.data.joint("platform_pivot").qvel[0]) < 1


def test_vortex_control_sampling_and_accumulated_lamp(mixer, monkeypatch):
    dial(mixer, 5, -0.4)
    mixer.dynamics.physics_step(mixer)
    driver = mixer.dynamics
    monkeypatch.setattr(driver, "forces", lambda env: (0.3, 0.3))
    monkeypatch.setattr("tasks.vortex_sample.dynamics._grasped", lambda env, obj: True)
    for i in range(5):
        mixer.num_step += 1
        mixer.data.joint("platform_pivot").qvel[0] = 180
        driver.physics_step(mixer)
        assert mixer.data.userdata[0] == i + 1
        for _ in range(200):
            driver.physics_step(mixer)
        assert mixer.data.userdata[0] == i + 1
        mixer.num_step += 1
        mixer.data.joint("platform_pivot").qvel[0] = 0
        driver.physics_step(mixer)
        assert mixer.data.userdata[0] == i + 1
    np.testing.assert_allclose(mixer.model.geom("mix_lamp").rgba, [0.03, 1, 0.08, 1])
    dial(mixer, 5, 0)
    mixer.num_step += 1
    driver.physics_step(mixer)
    assert mixer.data.userdata[0] == 0


def test_vortex_unheld_sample_cannot_mix(mixer, monkeypatch):
    dial(mixer, 5, 0.4)
    mixer.dynamics.physics_step(mixer)
    driver = mixer.dynamics
    monkeypatch.setattr(driver, "forces", lambda env: (0.3, 0.3))
    monkeypatch.setattr("tasks.vortex_sample.dynamics._grasped", lambda env, obj: False)
    for _ in range(10):
        mixer.num_step += 1
        mixer.data.joint("platform_pivot").qvel[0] = 180
        driver.physics_step(mixer)
    assert mixer.data.userdata[0] == 0


def test_vortex_geometry_orbit_and_state_serialization(mixer):
    points = []
    for angle in np.arange(4) * np.pi / 2:
        mixer.data.joint("platform_pivot").qpos[0] = angle
        mixer.data.joint("platform_orbit").qpos[0] = -angle
        mixer.forward()
        points.append(mixer.data.site("platform_function").xpos.copy())
    points = np.asarray(points)
    np.testing.assert_allclose(np.linalg.norm(points - points.mean(0), axis=1), 0.002)
    mixer.data.userdata[0] = 3
    saved = mixer.sim_state()
    mixer.data.userdata[0] = 5
    mixer.reset_to_sim_state(saved)
    assert mixer.data.userdata[0] == 3
    assert mixer.model.opt.timestep == 0.00025
    assert mixer.model.geom("mix_liquid_surface").contype == 0


def test_vortex_reset_randomize_and_rewind_repaint(mixer):
    driver = mixer.dynamics
    mixer.data.userdata[0] = 3
    mixer.data.userdata[2] = 0
    before = mixer.checkpoint()
    mixer.data.userdata[0] = 5
    mixer.data.userdata[2] = 1
    driver.paint(mixer)
    assert mixer.model.geom("mix_lamp").rgba[1] == 1
    assert mixer.model.geom("mix_liquid_surface").quat[1] != 0
    mixer.restore(before)
    assert mixer.data.userdata[0] == 3
    np.testing.assert_allclose(mixer.model.geom("mix_lamp").rgba, [0.08, 0.09, 0.08, 1])
    np.testing.assert_allclose(mixer.model.geom("mix_liquid_surface").quat, [1, 0, 0, 0])
    mixer.data.userdata[0] = 5
    mixer.redeal_episode()
    assert mixer.data.userdata[0] == 0
    assert mixer.model.geom("mix_lamp").rgba[1] < 0.1
    mixer.data.userdata[0] = 5
    mixer.reset(seed=0)
    assert mixer.data.userdata[0] == 0
    assert mixer.model.geom("mix_lamp").rgba[1] < 0.1


def test_vortex_grading_requires_mix_then_stopped_and_return(mixer):
    controls(mixer, 1)
    dial(mixer, 5, -0.4)
    check = mixer._check
    assert check(mixer) is False
    assert check(mixer) is False
    assert check.at == 2
    mixer.data.userdata[0] = 4
    assert check(mixer) is False
    assert check.at == 2
    mixer.data.userdata[0] = 5
    assert check(mixer) is False
    assert check.at == 3
    dial(mixer, 5, 0)
    mixer.data.userdata[0] = 0
    mixer.data.joint("platform_pivot").qvel[0] = 100
    assert check(mixer) is False
    assert check.at == 3
    mixer.data.joint("platform_pivot").qvel[0] = 0
    assert check(mixer) is False
    assert check.at == 4
    assert check(mixer) is True
    mixer.data.joint("platform_pivot").qvel[0] = 2
    assert check(mixer) is False


def test_vortex_tilt_invariant_latches_failure(mixer):
    angle = np.radians(16)
    a = mixer._obj_qadr["tube"]
    mixer.data.qpos[a + 3:a + 7] = [np.cos(angle / 2), np.sin(angle / 2), 0, 0]
    mixer.forward()
    for _ in range(5):
        assert mixer._check(mixer) is False
    assert mixer._check.dead
    mixer.data.qpos[a + 3:a + 7] = [1, 0, 0, 0]
    mixer.forward()
    assert mixer._check(mixer) is False
    assert mixer._check.dead


def test_vortex_liquid_relaxes_without_grasp_credit(mixer, monkeypatch):
    dial(mixer, 5, 0.4)
    driver = mixer.dynamics
    monkeypatch.setattr(driver, "forces", lambda env: (0.3, 0.3))
    monkeypatch.setattr("tasks.vortex_sample.dynamics._grasped", lambda env, obj: False)
    mixer.data.joint("platform_pivot").qvel[0] = 180
    driver.physics_step(mixer)
    response = mixer.data.userdata[2]
    assert 0 < response < 1
    assert mixer.data.userdata[0] == 0
    monkeypatch.setattr(driver, "forces", lambda env: (0, 0))
    mixer.num_step += 1
    driver.physics_step(mixer)
    assert 0 < mixer.data.userdata[2] < response
    assert mixer.data.userdata[0] == 0


def test_joint_speed_in_reads_velocity_not_position_or_control():
    from types import SimpleNamespace
    m = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><body><joint name="spin" type="hinge"/>'
        '<geom type="sphere" size="0.1"/></body></worldbody></mujoco>')
    d = mujoco.MjData(m)
    env = SimpleNamespace(model=m, data=d)
    check, _ = make_checker({"pred": "joint_speed_in", "joint": "spin", "lo": -1, "hi": 1}, "vortex_sample")
    d.qpos[0] = 200
    for velocity, expected in ((-1, True), (1, True), (1.001, False), (-1.001, False)):
        d.qvel[0] = velocity
        assert check(env) is expected
    assert "rad/s" in predicates.describe(
        {"pred": "joint_speed_in", "joint": "spin", "lo": -1, "hi": 1}, "vortex_sample")[0][1]


def test_userdata_in_inclusive_bounds_and_description():
    from types import SimpleNamespace
    env = SimpleNamespace(data=SimpleNamespace(userdata=np.array([0.0, 5.0])))
    spec = {"pred": "userdata_in", "index": 1, "lo": 5, "hi": 5}
    check, _ = make_checker(spec, "vortex_sample")
    assert check(env) is True
    env.data.userdata[1] = 4
    assert check(env) is False
    env.data.userdata[1] = 6
    assert check(env) is False
    assert "scene state 1" in predicates.describe(spec, "vortex_sample")[0][1]
