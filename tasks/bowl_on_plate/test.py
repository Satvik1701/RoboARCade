"""Tests of the bowl_on_plate task."""
import numpy as np
from core.grade_episode import _bbox_world
from helpers import (
    Episode,
    project,
)
import pytest
import mujoco
from core.grade_episode import make_checker
from core.tasks import TASKS
from core.sim_env import SimEnv
from helpers import (  # noqa: F401 (lift_env is a fixture)
    BOWL,
    PLATE,
    lift_env,
    set_free_body,
    settle_steps,
    zero_free_qvel,
)
from scipy.spatial.transform import Rotation as R
from core.grade_episode import _footprint_hull, _hull_contains_xy


pytestmark = pytest.mark.e2e


BOWL = "akita_black_bowl_1_main"


PLATE = "plate_1_main"


def _grasp_bowl_rim(ep):
    env, ui, k = ep.env, ep.ui, ep.k
    bowl = k.object_poses()[BOWL]
    grasp = bowl + [0, 0.06, 0.04]
    u, v = project(env, "ring1", grasp)
    pt = ui.gripper_teleport_via_click("ring1", u, v)
    ui.gripper_translate(*(grasp - pt))
    print(f"  click ({u},{v}) -> 3D {np.round(pt, 3)} (bowl {np.round(bowl, 3)})")
    ep.do("teleport above bowl rim")
    ui.gripper_advance_or_retreat(0.10)
    ep.do("descend to rim")
    ui.gripper_toggle()
    ep.do("close gripper")
    assert 0.002 < env.gripper_width() < 0.06, f"grasp missed (width {env.gripper_width():.3f})"
    return bowl


def test_bowl_grasp_and_lift():
    print("=== bowl grasp and lift (default interface) ===")
    ep = Episode("bowl_on_plate", seed=0)
    env, ui = ep.env, ep.ui

    start_z = float(_grasp_bowl_rim(ep)[2])
    ui.gripper_translate(0, 0, 0.15)
    ep.do("lift")

    assert env.data.body(BOWL).xpos[2] - start_z > 0.10
    assert not make_checker(env.spec["success"])[0](env) and not env.episode_over
    ep.save_video("bowl_grasp_and_lift")
    print(f"  PASS: physical lift completed in {env.num_step} steps\n")


def test_bowl_on_plate():
    print("=== bowl on plate (default interface) ===")
    ep = Episode("bowl_on_plate", seed=0)
    env, ui, k = ep.env, ep.ui, ep.k

    bowl = k.object_poses()[BOWL]
    bowl_bottom_offset = float(bowl[2] - _bbox_world(env, BOWL)[:, 2].min())
    _grasp_bowl_rim(ep)
    ui.gripper_translate(0, 0, 0.15)
    ep.do("lift")
    held_offset = k.object_poses()[BOWL] - k.proprio().eef_pos
    plate = k.object_poses()[PLATE]
    plate_top = float(_bbox_world(env, PLATE)[:, 2].max())
    target_body = np.array([plate[0], plate[1], plate_top + bowl_bottom_offset + 0.002])
    ui.target_pos = np.array([plate[0] - held_offset[0], plate[1] - held_offset[1],
                              ui.target_pos[2]])
    ep.do("carry over plate")
    ui.target_pos = target_body - held_offset
    ep.do("lower onto plate")
    ui.gripper_toggle()
    ep.do("open")

    assert make_checker(env.spec["success"])[0](env), "bowl on plate did not succeed"
    ep.save_video("bowl_on_plate")
    print(f"  PASS: steps={env.num_step}\n")


def test_bowl_on_plate_success_tree():
    env = SimEnv("bowl_on_plate", seed=0)
    env.reset(seed=0, randomize_init=False)
    check, hold = make_checker(env.spec["success"])
    assert hold == 1
    assert check(env) is False

    plate = env.data.body(PLATE).xpos.copy()
    set_free_body(env, BOWL, plate + [0, 0, 0.12])
    mujoco.mj_forward(env.model, env.data)
    assert check(env) is False

    set_free_body(env, BOWL, plate + [0, 0, 0.05])
    zero_free_qvel(env, BOWL)
    for _ in range(settle_steps(env, 0.6)):
        mujoco.mj_step(env.model, env.data)
    assert check(env) is True


def test_libero_on_success_tree():
    if not (TASKS / "bowl_on_plate" / "task.json").exists():
        pytest.skip("libero/bowl_on_plate pack not materialized")
    env = SimEnv("bowl_on_plate", seed=0)
    env.reset(seed=0, randomize_init=False)
    check, hold = make_checker(env.spec["success"])
    assert hold == 1 and check(env) is False

    plate = env.data.body("plate_1_main").xpos.copy()
    set_free_body(env, "akita_black_bowl_1_main", plate + [0, 0, 0.05])
    zero_free_qvel(env, "akita_black_bowl_1_main")
    for _ in range(settle_steps(env, 0.6)):
        mujoco.mj_step(env.model, env.data)
    assert check(env) is True


def _set_body_rot(env, name, rot: R, pos=(0.0, 0.0, 0.9)):
    qadr = env.model.joint(env.model.body(name).jntadr[0]).qposadr[0]
    env.data.qpos[qadr : qadr + 3] = pos
    q = rot.as_quat()
    env.data.qpos[qadr + 3 : qadr + 7] = [q[3], q[0], q[1], q[2]]
    mujoco.mj_forward(env.model, env.data)


def _seq_lift_spec(**extra):
    return {"pred": "sequence", "of": [
        {"pred": "lift_above", "obj": BOWL, "z_above": 1.0},
        {"pred": "z_below", "obj": BOWL, "z_below": 0.85},
    ], **extra}


def _step(check, env, z):
    set_free_body(env, BOWL, [0, 0, z])
    mujoco.mj_forward(env.model, env.data)
    return check(env)


def test_lift_above_threshold(lift_env):
    env = lift_env
    env.reset(seed=0)
    lift = {"pred": "lift_above", "obj": BOWL, "z_above": 1.0}
    check, hold = make_checker(lift)
    assert hold == 1
    z = lift["z_above"]
    for cube_z, expect in ((z - 0.02, False), (z, False), (z + 0.001, True)):
        set_free_body(env, BOWL, [0, 0, cube_z])
        mujoco.mj_forward(env.model, env.data)
        assert check(env) is expect, f"cube z={cube_z} vs threshold {z}"


def test_unknown_pred_rejected():
    with pytest.raises(ValueError):
        make_checker({"pred": "definitely_not_a_pred"})


def test_axis_up_is_direction_sensitive():
    env = SimEnv("bowl_on_plate", seed=0)
    env.reset(seed=0, randomize_init=False)
    obj = env._free_objects[0]
    up = make_checker({"pred": "axis_up", "obj": obj, "axis": [0, 0, 1], "tol_deg": 10})[0]
    down = make_checker({"pred": "axis_up", "obj": obj, "axis": [0, 0, -1], "tol_deg": 10})[0]

    _set_body_rot(env, obj, R.identity())
    assert up(env) is True and down(env) is False

    _set_body_rot(env, obj, R.from_euler("x", 180, degrees=True))
    assert up(env) is False and down(env) is True

    _set_body_rot(env, obj, R.from_euler("x", 9, degrees=True))
    assert up(env) is True
    _set_body_rot(env, obj, R.from_euler("x", 11, degrees=True))
    assert up(env) is False

    _set_body_rot(env, obj, R.from_euler("z", 90, degrees=True))
    assert up(env) is True


def test_gripper_open_fraction():
    env = SimEnv("bowl_on_plate", seed=0)
    env.reset(seed=0, randomize_init=False)
    check = make_checker({"pred": "gripper_open", "min": 0.8})[0]
    assert check(env) is True

    for qa in env._finger_qadr[env.arms[0]]:
        env.data.qpos[qa] = 0.005
    mujoco.mj_forward(env.model, env.data)
    assert check(env) is False


def test_hull_contains_xy_is_oriented():
    sq = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], float)
    assert _hull_contains_xy(_footprint_hull(sq), (0.0, 0.0)) is True
    assert _hull_contains_xy(_footprint_hull(sq), (0.99, 0.99)) is True
    assert _hull_contains_xy(_footprint_hull(sq), (1.01, 0.0)) is False
    assert _hull_contains_xy(_footprint_hull(sq), (1.0, 0.0)) is True

    c, s = np.cos(np.pi / 4), np.sin(np.pi / 4)
    diamond = sq @ np.array([[c, -s], [s, c]]).T
    assert _hull_contains_xy(_footprint_hull(diamond), (0.0, 0.0)) is True
    assert _hull_contains_xy(_footprint_hull(diamond), (1.3, 0.0)) is True
    assert _hull_contains_xy(_footprint_hull(diamond), (0.9, 0.9)) is False
    assert _hull_contains_xy(_footprint_hull(np.array([[0, 0], [1, 1]], float)), (0.5, 0.5)) is False


def test_in_bbox_and_depth_in():
    env = SimEnv("bowl_on_plate", seed=0)
    env.reset(seed=0, randomize_init=False)
    a, b = env._free_objects[0], env._free_objects[1]
    box_b = _bbox_world(env, b)
    bx, by = box_b[:, 0].mean(), box_b[:, 1].mean()
    b_top, b_bot = box_b[:, 2].max(), box_b[:, 2].min()

    inb = make_checker({"pred": "in_bbox", "a": a, "b": b})[0]
    set_free_body(env, a, [bx, by, b_top + 0.02])
    mujoco.mj_forward(env.model, env.data)
    assert inb(env) is True
    set_free_body(env, a, [bx + 0.5, by, b_top + 0.02])
    mujoco.mj_forward(env.model, env.data)
    assert inb(env) is False
    set_free_body(env, a, [bx, by, b_bot - 0.05])
    mujoco.mj_forward(env.model, env.data)
    assert inb(env) is False

    deep = make_checker({"pred": "depth_in", "a": a, "b": b, "z_threshold": 0.01})[0]
    set_free_body(env, a, [bx, by, b_top + 0.10])
    mujoco.mj_forward(env.model, env.data)
    assert deep(env) is False
    set_free_body(env, a, [bx, by, b_bot])
    mujoco.mj_forward(env.model, env.data)
    assert deep(env) is True


def test_sequence_requires_the_stages_in_order(lift_env):
    env = lift_env
    env.reset(seed=0)
    check, hold = make_checker(_seq_lift_spec())
    assert hold == 1
    assert _step(check, env, 0.83) is False
    assert _step(check, env, 1.05) is False
    assert _step(check, env, 0.83) is True
    assert _step(check, env, 1.05) is False
    assert _step(check, env, 0.83) is True


def test_sequence_rejects_the_end_state_alone(lift_env):
    env = lift_env
    env.reset(seed=0)
    check, _ = make_checker(_seq_lift_spec())
    for _ in range(5):
        assert _step(check, env, 0.83) is False


def test_sequence_resets_between_episodes(lift_env):
    env = lift_env
    env.reset(seed=0)
    check, _ = make_checker(_seq_lift_spec())
    _step(check, env, 1.05)
    assert _step(check, env, 0.83) is True
    check, _ = make_checker(_seq_lift_spec())
    assert _step(check, env, 0.83) is False


def test_sequence_invariant_kills_progress_for_good(lift_env):
    env = lift_env
    env.reset(seed=0)
    spec = _seq_lift_spec(
        invariant={"pred": "not", "of": {"pred": "z_below", "obj": BOWL,
                                         "z_below": 0.5}})
    check, _ = make_checker(spec)
    assert _step(check, env, 1.05) is False
    for _ in range(5):
        assert _step(check, env, 0.2) is False
    assert _step(check, env, 0.83) is False
    check, _ = make_checker(spec)
    assert _step(check, env, 1.05) is False and _step(check, env, 0.83) is True
    check, _ = make_checker(spec)
    assert _step(check, env, 1.05) is False
    for _ in range(4):
        assert _step(check, env, 0.2) is False
    assert _step(check, env, 0.83) is True


def test_sequence_advances_one_stage_per_step(lift_env):
    env = lift_env
    env.reset(seed=0)
    check, _ = make_checker({"pred": "sequence", "of": [
        {"pred": "z_below", "obj": BOWL, "z_below": 0.9},
        {"pred": "z_below", "obj": BOWL, "z_below": 0.9},
    ]})
    assert _step(check, env, 0.83) is False
    assert _step(check, env, 0.83) is True


def test_sequence_stage_hold_requires_consecutive_steps(lift_env):
    env = lift_env
    env.reset(seed=0)
    check, _ = make_checker({"pred": "sequence", "of": [
        {"pred": "lift_above", "obj": BOWL, "z_above": 1.0, "hold": 3},
        {"pred": "z_below", "obj": BOWL, "z_below": 0.85},
    ]})
    assert not _step(check, env, 1.05)
    assert not _step(check, env, 1.05)
    assert not _step(check, env, 0.83)
    assert check.at == 0 and check.streak == 0
    assert not _step(check, env, 1.05)
    assert not _step(check, env, 1.05)
    assert not _step(check, env, 1.05)
    assert check.at == 1 and check.streak == 0
    assert _step(check, env, 0.83)


@pytest.mark.parametrize("hold", [0, -1, 1.5, True])
def test_sequence_stage_hold_must_be_a_positive_integer(hold):
    with pytest.raises(ValueError, match="positive integer"):
        make_checker({"pred": "sequence", "of": [
            {"pred": "never", "hold": hold},
        ]})


def test_grading_state_rewinds_with_the_scene(lift_env):
    env = lift_env
    env.reset(seed=0)
    env._check = make_checker(_seq_lift_spec())[0]
    _step(env._check, env, 0.83)
    saved = env.checkpoint()
    assert _step(env._check, env, 1.05) is False
    env.restore(saved)
    assert _step(env._check, env, 0.83) is False


def test_describe_state_rows_align_and_turn_true_on_the_solved_scene():
    from core.grade_episode import describe, describe_state
    env = SimEnv("bowl_on_plate", seed=0)
    env.reset(seed=0, randomize_init=False)
    spec = env.spec["success"]
    before = describe_state(spec, env)
    assert len(before) == len(describe(spec))
    assert before[0] is False
    plate = env.data.body(PLATE).xpos.copy()
    set_free_body(env, BOWL, plate + [0, 0, 0.05])
    zero_free_qvel(env, BOWL)
    for _ in range(settle_steps(env, 0.6)):
        mujoco.mj_step(env.model, env.data)
    after = describe_state(spec, env)
    assert after[0] is True and all(v is True for v in after if v is not None)
